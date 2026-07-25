//! CUDA stream management for asynchronous pipelines.
//!
//! Provides a thin safe wrapper around `CUstream` and `CUevent` handles.
//! Each GPU gets a dedicated **prefetch stream** that runs ahead of the
//! compute stream, overlapping DRAM→VRAM DMA transfers with Tensor Core work.

use crate::ffi::cuda::*;
use crate::{check_cu, Result};

// ============================================================================
// Safe stream wrapper
// ============================================================================

/// A managed CUDA stream with automatic cleanup.
#[derive(Debug)]
pub struct CudaStream {
    inner: CUstream,
    /// Priority: higher value = lower priority (CUDA convention).
    /// -1 is the highest priority on most hardware.
    #[allow(dead_code)]
    priority: i32,
}

impl CudaStream {
    /// Create a new stream with default priority.
    pub fn new() -> Result<Self> {
        let mut stream = CUstream::NULL;
        unsafe {
            check_cu(
                "cuStreamCreate",
                cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING),
            )?;
        }
        Ok(CudaStream {
            inner: stream,
            priority: 0,
        })
    }

    /// Create a stream with a specific priority.
    /// `priority` — higher = lower priority. -1 is highest.
    pub fn with_priority(priority: i32) -> Result<Self> {
        let mut stream = CUstream::NULL;
        unsafe {
            check_cu(
                "cuStreamCreateWithPriority",
                cuStreamCreateWithPriority(&mut stream, CU_STREAM_NON_BLOCKING, priority),
            )?;
        }
        Ok(CudaStream {
            inner: stream,
            priority,
        })
    }

    /// Synchronize: block until all work on this stream completes.
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check_cu("cuStreamSynchronize", cuStreamSynchronize(self.inner))?;
        }
        Ok(())
    }

    /// Query whether all work has completed (non-blocking).
    pub fn query_complete(&self) -> Result<bool> {
        unsafe {
            let res = cuStreamQuery(self.inner);
            if res == CUDA_SUCCESS {
                Ok(true)
            } else if res == 900 {
                // CUDA_ERROR_NOT_READY
                Ok(false)
            } else {
                check_cu("cuStreamQuery", res)?;
                unreachable!()
            }
        }
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> CUstream {
        self.inner
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.inner.is_default() {
            unsafe {
                cuStreamDestroy_v2(self.inner);
            }
        }
    }
}

// ============================================================================
// Safe event wrapper
// ============================================================================

/// A managed CUDA event with automatic cleanup.
#[derive(Debug)]
pub struct CudaEvent {
    inner: CUevent,
}

impl CudaEvent {
    /// Create a blocking-sync event (suitable for host-side waits).
    pub fn new_blocking() -> Result<Self> {
        let mut event = CUevent(std::ptr::null_mut());
        unsafe {
            check_cu(
                "cuEventCreate",
                cuEventCreate(&mut event, CU_EVENT_BLOCKING_SYNC),
            )?;
        }
        Ok(CudaEvent { inner: event })
    }

    /// Create a default (non-blocking) event.
    pub fn new_default() -> Result<Self> {
        let mut event = CUevent(std::ptr::null_mut());
        unsafe {
            check_cu("cuEventCreate", cuEventCreate(&mut event, CU_EVENT_DEFAULT))?;
        }
        Ok(CudaEvent { inner: event })
    }

    /// Record this event on the given stream.
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        unsafe {
            check_cu("cuEventRecord", cuEventRecord(self.inner, stream.as_raw()))?;
        }
        Ok(())
    }

    /// Block the calling thread until this event has been reached.
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check_cu("cuEventSynchronize", cuEventSynchronize(self.inner))?;
        }
        Ok(())
    }

    /// Check if the event has completed (non-blocking).
    pub fn query(&self) -> Result<bool> {
        unsafe {
            let res = cuEventQuery(self.inner);
            if res == CUDA_SUCCESS {
                Ok(true)
            } else if res == 900 {
                // CUDA_ERROR_NOT_READY
                Ok(false)
            } else {
                check_cu("cuEventQuery", res)?;
                unreachable!()
            }
        }
    }

    /// Get elapsed time in milliseconds between two events.
    pub fn elapsed(&self, other: &CudaEvent) -> Result<f32> {
        let mut ms = 0.0f32;
        unsafe {
            check_cu(
                "cuEventElapsedTime",
                cuEventElapsedTime(&mut ms, other.inner, self.inner),
            )?;
        }
        Ok(ms)
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> CUevent {
        self.inner
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.inner.0.is_null() {
            unsafe {
                cuEventDestroy_v2(self.inner);
            }
        }
    }
}

// ============================================================================
// Stream pool (one per GPU)
// ============================================================================

/// A set of streams: one compute stream and one prefetch stream per GPU.
pub struct StreamPool {
    /// Compute streams — one per GPU.
    pub compute: Vec<CudaStream>,
    /// Prefetch streams — highest priority, one per GPU.
    pub prefetch: Vec<CudaStream>,
    /// Number of GPUs.
    pub len: usize,
}

impl StreamPool {
    /// Create compute + prefetch stream pairs for `num_gpus` GPUs.
    pub fn new(num_gpus: usize) -> Result<Self> {
        let mut compute = Vec::with_capacity(num_gpus);
        let mut prefetch = Vec::with_capacity(num_gpus);

        for _ in 0..num_gpus {
            compute.push(CudaStream::new()?);
            // Prefetch stream at highest priority (-1) so it doesn't
            // contend with compute.
            prefetch.push(CudaStream::with_priority(-1)?);
        }

        Ok(StreamPool {
            compute,
            prefetch,
            len: num_gpus,
        })
    }

    /// Synchronize both streams on GPU `idx`.
    pub fn sync(&self, idx: usize) -> Result<()> {
        self.compute[idx].synchronize()?;
        self.prefetch[idx].synchronize()?;
        Ok(())
    }
}
