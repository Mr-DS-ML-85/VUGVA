//! V1 Unified Allocator — multi-GPU VRAM pool.
//!
//! Implements Algorithm 1 from the paper (§3.1):
//!
//! 1. Query local free memory on each GPU.
//! 2. If the allocation fits entirely in one GPU → fast-path native alloc.
//! 3. Otherwise → VMT_AllocateVirtualPool: shard across all GPUs.
//!
//! The allocator also exposes the `VugvaEngine` — the top-level public API
//! that ties together VMT, GPU cluster, streams, and prefetch.

use crate::ffi::cuda::*;
use crate::gpu::{GpuCluster, GpuInfo};
use crate::streams::StreamPool;
use crate::vmt::{Chunk, PageState, Tier, VirtualMemoryTable};
use crate::{check_cu, Result, VugvaError};

/// Maximum allocation that stays on a single GPU (128 MiB).
/// Allocations above this threshold are sharded across the cluster.
const SINGLE_GPU_LIMIT: usize = 128 * 1024 * 1024;

// ============================================================================
// V1 Unified Allocator
// ============================================================================

/// Low-level multi-GPU VRAM allocator. Manages per-GPU CUDA contexts and
/// raw `cuMemAlloc` calls.
pub struct UnifiedAllocator {
    /// Per-GPU CUDA contexts.
    contexts: Vec<CUcontext>,
    /// Device ordinals.
    #[allow(dead_code)]
    ordinals: Vec<i32>,
    /// GPU information.
    #[allow(dead_code)]
    infos: Vec<GpuInfo>,
}

impl UnifiedAllocator {
    /// Create and activate contexts for all GPUs.
    pub fn new(ordinals: &[i32], infos: &[GpuInfo]) -> Result<Self> {
        let mut contexts = Vec::with_capacity(ordinals.len());

        for &ord in ordinals {
            let dev = CUdevice(ord);
            let mut ctx = CUcontext(std::ptr::null_mut());
            unsafe {
                check_cu("cuCtxCreate_v2", cuCtxCreate_v2(&mut ctx, 0, dev))?;
            }
            contexts.push(ctx);
        }

        Ok(UnifiedAllocator {
            contexts,
            ordinals: ordinals.to_vec(),
            infos: infos.to_vec(),
        })
    }

    /// Switch to GPU `idx`'s context.
    pub fn set_device(&self, idx: usize) -> Result<()> {
        unsafe {
            check_cu("cuCtxSetCurrent", cuCtxSetCurrent(self.contexts[idx]))?;
        }
        Ok(())
    }

    /// Query free VRAM on GPU `idx` (bytes).
    pub fn free_vram(&self, idx: usize) -> Result<usize> {
        self.set_device(idx)?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            check_cu("cuMemGetInfo_v2", cuMemGetInfo_v2(&mut free, &mut total))?;
        }
        Ok(free)
    }

    /// Allocate raw device memory on GPU `idx`.
    pub fn alloc_device(&self, idx: usize, bytes: usize) -> Result<u64> {
        self.set_device(idx)?;
        let mut dptr = CUdeviceptr::NULL;
        unsafe {
            check_cu("cuMemAlloc_v2", cuMemAlloc_v2(&mut dptr, bytes))?;
        }
        Ok(dptr.0)
    }

    /// Free device memory on GPU `idx`.
    pub fn free_device(&self, idx: usize, ptr: u64) -> Result<()> {
        self.set_device(idx)?;
        unsafe {
            check_cu("cuMemFree_v2", cuMemFree_v2(CUdeviceptr(ptr)))?;
        }
        Ok(())
    }

    /// Peer-to-peer async copy: src_gpu → dst_gpu.
    pub fn peer_copy_async(
        &self,
        src_gpu: usize,
        src_ptr: u64,
        dst_gpu: usize,
        dst_ptr: u64,
        bytes: usize,
        stream: CUstream,
    ) -> Result<()> {
        // For simplicity, both contexts must be the destination context.
        self.set_device(dst_gpu)?;
        unsafe {
            check_cu(
                "cuMemcpyPeerAsync",
                cuMemcpyPeerAsync(
                    CUdeviceptr(dst_ptr),
                    self.contexts[dst_gpu],
                    CUdeviceptr(src_ptr),
                    self.contexts[src_gpu],
                    bytes,
                    stream,
                ),
            )?;
        }
        Ok(())
    }

    /// Destroy all CUDA contexts.
    pub fn destroy(&mut self) {
        // Destroy in reverse order so that later GPUs are cleaned up first.
        for ctx in self.contexts.drain(..).rev() {
            unsafe {
                cuCtxSetCurrent(ctx);
                cuCtxDestroy_v2(ctx);
            }
        }
    }
}

impl Drop for UnifiedAllocator {
    fn drop(&mut self) {
        self.destroy();
    }
}

// ============================================================================
// VugvaEngine — the top-level V1 API
// ============================================================================

/// The VUGVA V1 engine: unified multi-GPU VRAM allocator with VMT,
/// prefetch streams, and NUMA-aware routing.
///
/// ```ignore
/// let engine = VugvaEngine::builder()
///     .gpus(&[0, 1, 2, 3])
///     .build()?;
///
/// let name = engine.allocate("model.embed.weight", &[8192, 8192], 2)?;
/// let ptr = engine.access(&name, 0)?; // ensure resident on GPU 0
/// ```
pub struct VugvaEngine {
    /// GPU cluster topology and P2P matrix.
    pub cluster: GpuCluster,
    /// The virtual memory table.
    pub vmt: VirtualMemoryTable,
    /// Low-level CUDA allocator.
    allocator: UnifiedAllocator,
    /// Stream pool (compute + prefetch per GPU).
    pub streams: StreamPool,
}

/// Builder for `VugvaEngine`.
pub struct VugvaEngineBuilder {
    gpu_ordinals: Vec<i32>,
}

impl Default for VugvaEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VugvaEngineBuilder {
    pub fn new() -> Self {
        VugvaEngineBuilder {
            gpu_ordinals: Vec::new(),
        }
    }
    pub fn gpus(mut self, ordinals: &[i32]) -> Self {
        self.gpu_ordinals = ordinals.to_vec();
        self
    }
    pub fn build(self) -> Result<VugvaEngine> {
        VugvaEngine::new(&self.gpu_ordinals)
    }
}

impl VugvaEngine {
    pub fn builder() -> VugvaEngineBuilder {
        VugvaEngineBuilder::new()
    }

    /// Initialize the engine for the given GPU ordinals.
    pub fn new(gpu_ordinals: &[i32]) -> Result<Self> {
        if gpu_ordinals.is_empty() {
            return Err(VugvaError::InvalidGpu(0));
        }

        let mut cluster = GpuCluster::discover(gpu_ordinals)?;
        cluster.enable_peer_access()?;

        let allocator = UnifiedAllocator::new(&cluster.ordinals, &cluster.infos)?;
        let num_numa = cluster.numa.node_count;
        let vmt = VirtualMemoryTable::new(gpu_ordinals.len(), num_numa);
        let streams = StreamPool::new(gpu_ordinals.len())?;

        Ok(VugvaEngine {
            cluster,
            vmt,
            allocator,
            streams,
        })
    }

    /// Allocate a unified tensor spanning the VRAM cluster.
    ///
    /// Implements Algorithm 1 from the paper.
    pub fn allocate(&mut self, name: &str, shape: &[usize], element_size: usize) -> Result<String> {
        let total_bytes: usize = shape.iter().product::<usize>() * element_size;
        let num_gpus = self.cluster.ordinals.len();
        let chunk_bytes = total_bytes / num_gpus;

        // Step 1: Check if allocation fits on a single GPU (fast path)
        if total_bytes <= SINGLE_GPU_LIMIT {
            if let Ok(free) = self.allocator.free_vram(0) {
                if free >= total_bytes {
                    let alloc_name = self.vmt.allocate(name, shape, element_size)?;
                    let page = self.vmt.lookup_mut(&alloc_name).unwrap();
                    let ptr = self.allocator.alloc_device(0, total_bytes)?;
                    page.vram_chunks.push(Chunk {
                        gpu_ordinal: self.cluster.ordinals[0],
                        device_ptr: ptr,
                        size_bytes: total_bytes,
                        num_elements: total_bytes / element_size,
                    });
                    page.state = PageState::Resident;
                    page.tier = Tier::Vram;
                    return Ok(alloc_name);
                }
            }
        }

        // Step 2: Shard across all GPUs
        let alloc_name = self.vmt.allocate(name, shape, element_size)?;
        let page = self.vmt.lookup_mut(&alloc_name).unwrap();

        for (i, &ord) in self.cluster.ordinals.iter().enumerate() {
            let mut actual_chunk = chunk_bytes;
            // Last GPU gets the remainder
            if i == num_gpus - 1 {
                actual_chunk = total_bytes - chunk_bytes * (num_gpus - 1);
            }
            let ptr = self.allocator.alloc_device(i, actual_chunk)?;
            page.vram_chunks.push(Chunk {
                gpu_ordinal: ord,
                device_ptr: ptr,
                size_bytes: actual_chunk,
                num_elements: actual_chunk / element_size,
            });
        }

        page.state = PageState::Resident;
        page.tier = Tier::Vram;
        Ok(alloc_name)
    }

    /// Ensure a tensor page is resident on a specific GPU.
    ///
    /// If the page's chunk for `gpu_ordinal` is already in VRAM, return
    /// its device pointer. Otherwise, peer-copy from the owning GPU.
    pub fn access(&mut self, name: &str, gpu_ordinal: i32) -> Result<u64> {
        let gpu_idx = self
            .cluster
            .index_of(gpu_ordinal)
            .ok_or(VugvaError::InvalidGpu(gpu_ordinal as usize))?;

        let page = self
            .vmt
            .lookup(name)
            .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;

        // If resident and has a chunk for this GPU, fast return
        if page.state == PageState::Resident {
            if let Some(chunk) = page
                .vram_chunks
                .iter()
                .find(|c| c.gpu_ordinal == gpu_ordinal)
            {
                return Ok(chunk.device_ptr);
            }
        }

        // Otherwise: page fault — need to bring data to this GPU
        // Find the first chunk (on whatever GPU) and peer-copy
        let (src_ptr, src_size, src_ordinal) = {
            let page = self
                .vmt
                .lookup(name)
                .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
            let src_chunk = page
                .vram_chunks
                .first()
                .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
            (
                src_chunk.device_ptr,
                src_chunk.size_bytes,
                src_chunk.gpu_ordinal,
            )
        };

        if src_ordinal == gpu_ordinal {
            return Ok(src_ptr);
        }

        // Peer copy: src_gpu → dst_gpu
        let src_idx = self.cluster.index_of(src_ordinal).unwrap();

        // Allocate on destination
        let dst_ptr = self.allocator.alloc_device(gpu_idx, src_size)?;

        self.allocator.peer_copy_async(
            src_idx,
            src_ptr,
            gpu_idx,
            dst_ptr,
            src_size,
            self.streams.compute[gpu_idx].as_raw(),
        )?;
        self.streams.compute[gpu_idx].synchronize()?;

        // Register the new chunk
        let page_mut = self.vmt.lookup_mut(name).unwrap();
        page_mut.vram_chunks.push(Chunk {
            gpu_ordinal,
            device_ptr: dst_ptr,
            size_bytes: src_size,
            num_elements: src_size / page_mut.element_size,
        });

        Ok(dst_ptr)
    }

    /// Free an allocation across all GPUs.
    pub fn free(&mut self, name: &str) -> Result<()> {
        let page = self
            .vmt
            .remove(name)
            .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;

        for (i, chunk) in page.vram_chunks.iter().enumerate() {
            let gpu_idx = self.cluster.index_of(chunk.gpu_ordinal).unwrap_or(i);
            let _ = self.allocator.free_device(gpu_idx, chunk.device_ptr);
        }

        Ok(())
    }
}
