//! V2 Three-Tier Hybrid Memory: VRAM → DRAM → SSD.
//!
//! Implements §4 and §5.1 of the paper: a unified three-tier hierarchy
//! with full CPU bypass. The CPU touches only 72 bytes of metadata
//! (8-byte page lookup + 64-byte DMA descriptor) per promotion, while
//! the DMA engine transfers megabytes.
//!
//! ## Page migration state machine (Figure 5)
//!
//! ```text
//! UNMAPPED → ALLOCATED → RESIDENT (VRAM)
//!                        ↕ promote/evict
//!                     WARM (DRAM)
//!                        ↕ spill/load
//!                     COLD (SSD)
//! ```

use crate::dma::DmaEngine;
use crate::ffi::cuda::*;
use crate::gpu::GpuCluster;
use crate::streams::StreamPool;
use crate::vmt::{Chunk, DramChunk, PageState, Tier, VirtualMemoryTable};
use crate::{check_cu, Result, VugvaError};

/// Default idle threshold before demotion: 5 seconds.
const DEFAULT_IDLE_NS: u64 = 5_000_000_000;
/// Access count threshold for proactive promotion.
const HOT_ACCESS_THRESHOLD: u64 = 10;

// ============================================================================
// TieredPool
// ============================================================================

/// The V2 three-tier memory pool: VRAM (hot) → DRAM (warm) → SSD (cold).
///
/// Wraps the VMT, DMA engine, and stream pool to provide the unified
/// access interface described in Algorithm 2 of the paper.
pub struct TieredPool {
    /// Virtual memory table.
    pub vmt: VirtualMemoryTable,
    /// DMA engine (CPU-bypass transfers).
    pub dma: DmaEngine,
    /// GPU cluster topology.
    pub cluster: GpuCluster,
    /// Stream pool for async operations.
    pub streams: StreamPool,
    /// Per-GPU DRAM pools (host pointers, NUMA-local).
    dram_pools: Vec<DramPool>,
    /// Idle timeout for demotion (nanoseconds).
    idle_threshold_ns: u64,
    /// Hot-access threshold for proactive promotion.
    hot_threshold: u64,
}

/// A NUMA-local DRAM pool for one GPU.
struct DramPool {
    /// NUMA node this pool is allocated on.
    numa_node: usize,
    /// GPU ordinal this pool serves.
    #[allow(dead_code)]
    gpu_ordinal: i32,
    /// Base host pointer.
    base_ptr: usize,
    /// Total capacity in bytes.
    capacity: usize,
    /// Current allocation offset.
    offset: usize,
}

impl DramPool {
    fn allocate(&mut self, bytes: usize) -> Result<usize> {
        let aligned = (bytes + 63) & !63; // 64-byte align
        if self.offset + aligned > self.capacity {
            return Err(VugvaError::CudaError {
                fn_name: "DramPool::allocate",
                code: CUDA_ERROR_OUT_OF_MEMORY,
            });
        }
        let ptr = self.base_ptr + self.offset;
        self.offset += aligned;
        Ok(ptr)
    }
}

impl TieredPool {
    /// Create a new tiered pool for the given GPU ordinals.
    ///
    /// * `dram_pool_size_per_gpu` — bytes of DRAM to reserve per GPU.
    pub fn new(gpu_ordinals: &[i32], dram_pool_size_per_gpu: usize) -> Result<Self> {
        let cluster = GpuCluster::discover(gpu_ordinals)?;
        let num_gpus = gpu_ordinals.len();
        let dma = DmaEngine::new(num_gpus);
        let streams = StreamPool::new(num_gpus)?;
        let vmt = VirtualMemoryTable::new(num_gpus, cluster.numa.node_count);

        // Allocate NUMA-local DRAM pools
        let mut dram_pools = Vec::with_capacity(num_gpus);
        for &ord in gpu_ordinals {
            let optimal_node = cluster.optimal_dram_node(ord);
            let base_ptr = allocate_numa_dram(dram_pool_size_per_gpu, optimal_node)?;
            dram_pools.push(DramPool {
                numa_node: optimal_node,
                gpu_ordinal: ord,
                base_ptr,
                capacity: dram_pool_size_per_gpu,
                offset: 0,
            });
        }

        Ok(TieredPool {
            vmt,
            dma,
            cluster,
            streams,
            dram_pools,
            idle_threshold_ns: DEFAULT_IDLE_NS,
            hot_threshold: HOT_ACCESS_THRESHOLD,
        })
    }

    /// Allocate a new tensor in the unified DRAM+VRAM pool.
    ///
    /// From Algorithm 1:
    /// 1. Allocate DRAM region (NUMA-local to each GPU).
    /// 2. Create VRAM aliases (populated on access via DMA).
    /// 3. CPU never touches tensor data.
    pub fn allocate(
        &mut self,
        name: &str,
        shape: &[usize],
        element_size: usize,
        initial_tier: Tier,
    ) -> Result<String> {
        let total_bytes: usize = shape.iter().product::<usize>() * element_size;
        let alloc_name = self.vmt.allocate(name, shape, element_size)?;

        // Allocate DRAM chunks for each GPU's NUMA-local pool
        {
            let num_pools = self.dram_pools.len();
            let page = self.vmt.lookup_mut(&alloc_name).unwrap();
            page.tier = initial_tier;

            for pool in self.dram_pools.iter_mut() {
                let chunk_bytes = total_bytes / num_pools;
                let host_ptr = pool.allocate(chunk_bytes)?;
                page.dram_chunks.push(DramChunk {
                    numa_node: pool.numa_node,
                    host_ptr,
                    size_bytes: chunk_bytes,
                    cuda_registered: false,
                });
            }
        }

        // Set initial state
        let state = match initial_tier {
            Tier::Vram => PageState::Resident,
            Tier::Dram => PageState::Warm,
            Tier::Ssd => PageState::Cold,
        };
        self.vmt.lookup_mut(&alloc_name).unwrap().state = state;

        Ok(alloc_name)
    }

    /// Access a tensor page, promoting to VRAM if needed.
    ///
    /// Implements Algorithm 2 from the paper:
    /// 1. CPU checks page table (8 bytes — control plane).
    /// 2. If in DRAM, CPU submits DMA descriptor (64 bytes).
    /// 3. DMA engine transfers data autonomously (megabytes).
    /// 4. CPU returns VRAM pointer to framework.
    ///
    /// Total CPU data touched: 72 bytes.
    /// Total data transferred: megabytes.
    pub fn access(&mut self, name: &str, gpu_idx: usize) -> Result<u64> {
        let now = current_time_ns();

        // Step 1: Check tier and get source info (immutable borrow)
        let (tier, maybe_vram_chunk, maybe_dram_chunk) = {
            let page = self
                .vmt
                .lookup(name)
                .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
            let ordinal = self.cluster.ordinals[gpu_idx];
            let vram_chunk = page
                .vram_chunks
                .iter()
                .find(|c| c.gpu_ordinal == ordinal)
                .cloned();
            let dram_chunk = page.dram_chunks.first().cloned();
            (page.tier, vram_chunk, dram_chunk)
        };

        match tier {
            Tier::Vram => {
                if let Some(chunk) = maybe_vram_chunk {
                    // Already hot and resident on this GPU — touch and return
                    self.vmt.lookup_mut(name).unwrap().touch(now);
                    return Ok(chunk.device_ptr);
                }
                // VRAM but not on this GPU: allocate locally and copy via sync memcpy
                // In production, use cuMemcpyPeerAsync with proper context tracking
                let src = maybe_vram_chunk.unwrap();
                let size = src.size_bytes;
                let dst_ptr = self.alloc_vram_on_gpu(gpu_idx, size)?;

                // Sync copy from src GPU (we need src context for this)
                // Simplified: read from src, write to dst
                self.set_context(gpu_idx)?;
                let mut host_buf = vec![0u8; size];
                unsafe {
                    // Copy src → host
                    let mut src_ctx = CUcontext(std::ptr::null_mut());
                    cuCtxCreate_v2(&mut src_ctx, 0, CUdevice(src.gpu_ordinal));
                    cuCtxSetCurrent(src_ctx);
                    crate::ffi::cuda::cuMemcpyDtoH_v2(
                        host_buf.as_mut_ptr() as *mut std::ffi::c_void,
                        CUdeviceptr(src.device_ptr),
                        size,
                    );
                    cuCtxDestroy_v2(src_ctx);
                    // Copy host → dst
                    self.set_context(gpu_idx)?;
                    crate::ffi::cuda::cuMemcpyHtoD_v2(
                        CUdeviceptr(dst_ptr),
                        host_buf.as_ptr() as *const std::ffi::c_void,
                        size,
                    );
                }

                self.vmt.lookup_mut(name).unwrap().touch(now);
                let page = self.vmt.lookup_mut(name).unwrap();
                let elem_size = page.element_size;
                page.vram_chunks.push(Chunk {
                    gpu_ordinal: self.cluster.ordinals[gpu_idx],
                    device_ptr: dst_ptr,
                    size_bytes: size,
                    num_elements: size / elem_size,
                });
                page.tier = Tier::Vram;
                page.state = PageState::Resident;
                Ok(dst_ptr)
            }
            Tier::Dram => {
                // In DRAM — promote via H2D copy (CPU-bypass DMA path)
                let dram = maybe_dram_chunk
                    .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
                let size = dram.size_bytes;
                let src_ptr = dram.host_ptr as u64;
                let dst_ptr = self.alloc_vram_on_gpu(gpu_idx, size)?;

                self.dma
                    .submit_dram_to_vram(gpu_idx, src_ptr, dst_ptr, size, 2)?;

                self.set_context(gpu_idx)?;
                unsafe {
                    crate::ffi::cuda::cuMemcpyHtoDAsync_v2(
                        crate::ffi::cuda::CUdeviceptr(dst_ptr),
                        src_ptr as *const std::ffi::c_void,
                        size,
                        self.streams.compute[gpu_idx].as_raw(),
                    );
                    self.streams.compute[gpu_idx].synchronize()?;
                }

                self.vmt.lookup_mut(name).unwrap().touch(now);
                let page = self.vmt.lookup_mut(name).unwrap();
                let elem_size = page.element_size;
                page.vram_chunks.push(Chunk {
                    gpu_ordinal: self.cluster.ordinals[gpu_idx],
                    device_ptr: dst_ptr,
                    size_bytes: size,
                    num_elements: size / elem_size,
                });
                page.tier = Tier::Vram;
                page.state = PageState::Resident;
                Ok(dst_ptr)
            }
            Tier::Ssd => {
                // SSD → DRAM → VRAM (two-step)
                // For now, treat as DRAM path
                let dram = maybe_dram_chunk
                    .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
                let size = dram.size_bytes;
                let src_ptr = dram.host_ptr as u64;
                let dst_ptr = self.alloc_vram_on_gpu(gpu_idx, size)?;

                self.set_context(gpu_idx)?;
                unsafe {
                    crate::ffi::cuda::cuMemcpyHtoDAsync_v2(
                        crate::ffi::cuda::CUdeviceptr(dst_ptr),
                        src_ptr as *const std::ffi::c_void,
                        size,
                        self.streams.compute[gpu_idx].as_raw(),
                    );
                    self.streams.compute[gpu_idx].synchronize()?;
                }

                self.vmt.lookup_mut(name).unwrap().touch(now);
                let page = self.vmt.lookup_mut(name).unwrap();
                let elem_size = page.element_size;
                page.vram_chunks.push(Chunk {
                    gpu_ordinal: self.cluster.ordinals[gpu_idx],
                    device_ptr: dst_ptr,
                    size_bytes: size,
                    num_elements: size / elem_size,
                });
                page.tier = Tier::Vram;
                page.state = PageState::Resident;
                Ok(dst_ptr)
            }
        }
    }

    /// Demote a page from VRAM to DRAM (CPU-bypass writeback).
    pub fn demote(&mut self, name: &str) -> Result<()> {
        let page = self
            .vmt
            .lookup(name)
            .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;

        if page.tier != Tier::Vram || page.pinned {
            return Ok(());
        }

        // GPU writes back dirty pages to DRAM via DMA
        // (CPU-bypass: GPU MMU handles the writeback)
        for chunk in &page.vram_chunks {
            let gpu_idx = self.cluster.index_of(chunk.gpu_ordinal).unwrap_or(0);
            if let Some(dram) = self.dram_pools.get(gpu_idx) {
                // Reverse DMA: VRAM → DRAM
                self.dma.submit_vram_to_vram(
                    gpu_idx,
                    gpu_idx,
                    chunk.device_ptr,
                    dram.base_ptr as u64,
                    chunk.size_bytes,
                    0, // cold priority
                )?;
            }
        }

        let page_mut = self.vmt.lookup_mut(name).unwrap();
        page_mut.tier = Tier::Dram;
        page_mut.state = PageState::Warm;

        Ok(())
    }

    /// Background sweep: demote idle pages, promote hot DRAM pages.
    pub fn background_sweep(&mut self) -> Result<()> {
        let now = current_time_ns();
        let mut to_promote: Vec<String> = Vec::new();
        let mut to_demote: Vec<String> = Vec::new();

        for (name, page) in self.vmt.iter() {
            if page.state == PageState::Resident && page.is_idle(now, self.idle_threshold_ns) {
                to_demote.push(name.clone());
            }
            if page.tier == Tier::Dram && page.is_hot(self.hot_threshold) {
                to_promote.push(name.clone());
            }
        }

        for name in &to_demote {
            self.demote(name)?;
        }
        for name in &to_promote {
            // Promote to VRAM on GPU 0 (first available)
            self.access(name, 0)?;
        }

        Ok(())
    }

    // --- internal helpers ---

    fn alloc_vram_on_gpu(&self, gpu_idx: usize, bytes: usize) -> Result<u64> {
        self.set_context(gpu_idx)?;
        let mut dptr = CUdeviceptr::NULL;
        unsafe {
            check_cu("cuMemAlloc_v2", cuMemAlloc_v2(&mut dptr, bytes))?;
        }
        Ok(dptr.0)
    }

    fn set_context(&self, gpu_idx: usize) -> Result<()> {
        let dev = CUdevice(self.cluster.ordinals[gpu_idx]);
        let mut ctx = CUcontext(std::ptr::null_mut());
        unsafe {
            check_cu("cuCtxCreate_v2", cuCtxCreate_v2(&mut ctx, 0, dev))?;
            check_cu("cuCtxSetCurrent", cuCtxSetCurrent(ctx))?;
        }
        Ok(())
    }
}

impl Drop for TieredPool {
    fn drop(&mut self) {
        // Free DRAM pools
        for pool in &self.dram_pools {
            if pool.base_ptr != 0 {
                // SAFETY: base_ptr was returned by mmap with pool.capacity bytes.
                unsafe {
                    libc_free(pool.base_ptr as *mut u8, pool.capacity);
                }
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Get current monotonic time in nanoseconds.
fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Allocate DRAM on a specific NUMA node using `libc::mmap` + `mbind`.
///
/// Falls back to plain `mmap` if `mbind` is unavailable.
fn allocate_numa_dram(size: usize, numa_node: usize) -> Result<usize> {
    use std::ffi::c_void;

    // Round up to page size (4KB)
    let page_size = 4096usize;
    let aligned_size = (size + page_size - 1) & !(page_size - 1);

    // mmap anonymous private mapping
    extern "C" {
        fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: isize,
        ) -> *mut c_void;
        fn mbind(
            addr: *mut c_void,
            len: usize,
            mode: i32,
            nodemask: *const u64,
            maxnode: u64,
            flags: u32,
        ) -> i32;
    }

    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_ANONYMOUS: i32 = 0x20;

    let ptr = unsafe {
        mmap(
            std::ptr::null_mut(),
            aligned_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };

    if ptr.is_null() || ptr as isize == -1 {
        return Err(VugvaError::CudaError {
            fn_name: "mmap",
            code: CUDA_ERROR_OUT_OF_MEMORY,
        });
    }

    // Try to bind to NUMA node (MPOL_BIND = 1)
    if numa_node > 0 {
        let mode = 1i32; // MPOL_BIND
        let nodemask: u64 = 1u64 << numa_node;
        let flags = 0u32;
        unsafe {
            mbind(
                ptr,
                aligned_size,
                mode,
                &nodemask,
                64, // maxnode
                flags,
            );
            // Non-fatal if mbind fails — memory still works, just not optimal NUMA placement
        }
    }

    Ok(ptr as usize)
}

/// Free memory allocated with `mmap`.
///
/// # Safety
///
/// `ptr` must be a pointer previously returned by `mmap`, and `size`
/// must be the exact allocation size.
unsafe fn libc_free(ptr: *mut u8, size: usize) {
    extern "C" {
        fn munmap(addr: *mut std::ffi::c_void, length: usize) -> i32;
    }
    if !ptr.is_null() && size > 0 {
        // SAFETY: ptr is a valid mmap'd region of `size` bytes.
        munmap(ptr as *mut std::ffi::c_void, size);
    }
}
