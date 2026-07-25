//! Raw FFI declarations for the **CUDA Driver API** subset used by VUGVA.
//!
//! Each function is resolved at runtime via `dlsym` through
//! [`super::cuda_module`]. The declarations below mirror the upstream
//! `cuda.h` header signatures.

use std::ffi::{c_char, c_int, c_void};

// ============================================================================
// Opaque types (mirror cuda.h)
// ============================================================================

/// `CUdevice` — integer device handle.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUdevice(pub i32);

/// `CUcontext` — pointer-sized context handle.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUcontext(pub *mut c_void);

/// `CUmodule` — loadable module.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUmodule(pub *mut c_void);

/// `CUfunction` — handle to a kernel function.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUfunction(pub *mut c_void);

/// `CUstream` — stream handle; null = default stream.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUstream(pub *mut c_void);

/// `CUdeviceptr` — pointer to device memory.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUdeviceptr(pub u64);

/// `CUevent` — CUDA event handle.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUevent(pub *mut c_void);

/// `CUresult` — return code from the driver API.
pub type CUresult = c_int;
/// `CUDA_SUCCESS`
pub const CUDA_SUCCESS: c_int = 0;
/// `CUresult::CUDA_ERROR_INVALID_VALUE`
pub const CUDA_ERROR_INVALID_VALUE: c_int = 1;
/// `CUresult::CUDA_ERROR_OUT_OF_MEMORY`
pub const CUDA_ERROR_OUT_OF_MEMORY: c_int = 2;

// ============================================================================
// Device attribute IDs (subset from CUdevice_attribute enum)
// ============================================================================

/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
/// `CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS`
pub const CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: c_int = 31;
/// `CU_DEVICE_ATTRIBUTE_PCI_BUS_ID`
pub const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: c_int = 33;
/// `CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID`
pub const CUDEVICE_ATTRIBUTE_PCI_DEVICE_ID: c_int = 34;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED`
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED: c_int = 62;
/// `CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED`
pub const CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED: c_int = 81;

// ============================================================================
// Stream / event flags
// ============================================================================

/// `CU_STREAM_DEFAULT` — default stream.
pub const CU_STREAM_DEFAULT: u32 = 0x00;
/// `CU_STREAM_NON_BLOCKING` — stream does not synchronize with null stream.
pub const CU_STREAM_NON_BLOCKING: u32 = 0x01;

/// `CU_EVENT_DEFAULT`
pub const CU_EVENT_DEFAULT: u32 = 0x00;
/// `CU_EVENT_BLOCKING_SYNC` — event uses blocking synchronization.
pub const CU_EVENT_BLOCKING_SYNC: u32 = 0x01;

// ============================================================================
// Memory flags
// ============================================================================

/// `CU_MEMHOSTALLOC_PORTABLE` — memory is portable across CUDA contexts.
pub const CU_MEMHOSTALLOC_PORTABLE: u32 = 0x01;
/// `CU_MEMHOSTALLOC_DEVICEMAP` — map allocation into device address space.
pub const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
/// `CU_MEMHOSTALLOC_WRITECOMBINED` — write-combined memory (faster host writes).
pub const CU_MEMHOSTALLOC_WRITECOMBINED: u32 = 0x04;

/// `CU_MEMHOSTREGISTER_PORTABLE`
pub const CU_MEMHOSTREGISTER_PORTABLE: u32 = 0x01;
/// `CU_MEMHOSTREGISTER_DEVICEMAP`
pub const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;
/// `CU_MEMHOSTREGISTER_IOMEMORY` — register mapped I/O memory.
pub const CU_MEMHOSTREGISTER_IOMEMORY: u32 = 0x04;

// ============================================================================
// External CUDA Driver API declarations
// ============================================================================

#[link(name = "cuda")]
extern "C" {
    // ---- Initialization & device query ----
    pub fn cuInit(flags: u32) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGet(dev: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceComputeCapability(
        major: *mut c_int,
        minor: *mut c_int,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;

    // ---- Context ----
    pub fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxGetCurrent(ctx: *mut CUcontext) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;
    pub fn cuCtxGetDevice(dev: *mut CUdevice) -> CUresult;

    // ---- Peer access ----
    pub fn cuDeviceCanAccessPeer(
        canAccessPeer: *mut c_int,
        dev: CUdevice,
        peerDev: CUdevice,
    ) -> CUresult;
    pub fn cuCtxEnablePeerAccess(peerDev: CUdevice, flags: u32) -> CUresult;
    pub fn cuCtxDisablePeerAccess(peerDev: CUdevice) -> CUresult;

    // ---- Memory: device allocation ----
    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;

    // ---- Memory: unified / managed allocation ----
    pub fn cuMemAllocManaged(dptr: *mut CUdeviceptr, bytesize: usize, flags: u32) -> CUresult;

    // ---- Memory: host pinned allocation ----
    pub fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, flags: u32) -> CUresult;
    pub fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    pub fn cuMemHostRegister(p: *mut c_void, bytesize: usize, flags: u32) -> CUresult;
    pub fn cuMemHostUnregister(p: *mut c_void) -> CUresult;

    // ---- Memory: host ↔ device copies (synchronous) ----
    pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytecount: usize) -> CUresult;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytecount: usize) -> CUresult;
    pub fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, bytecount: usize) -> CUresult;

    // ---- Memory: host ↔ device copies (asynchronous) ----
    pub fn cuMemcpyHtoDAsync_v2(
        dst: CUdeviceptr,
        src: *const c_void,
        bytecount: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoHAsync_v2(
        dst: *mut c_void,
        src: CUdeviceptr,
        bytecount: usize,
        stream: CUstream,
    ) -> CUresult;

    // ---- Memory: device ↔ device copies ----
    pub fn cuMemcpyDtoDAsync_v2(
        dst: CUdeviceptr,
        src: CUdeviceptr,
        bytecount: usize,
        stream: CUstream,
    ) -> CUresult;

    // ---- Memory: peer-to-peer copies ----
    pub fn cuMemcpyPeerAsync(
        dst: CUdeviceptr,
        dst_ctx: CUcontext,
        src: CUdeviceptr,
        src_ctx: CUcontext,
        bytecount: usize,
        stream: CUstream,
    ) -> CUresult;

    // ---- Memory: query ----
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    pub fn cuMemGetAddressRange(
        pbase: *mut CUdeviceptr,
        pbytesize: *mut usize,
        dptr: CUdeviceptr,
    ) -> CUresult;

    // ---- Memory: memset ----
    pub fn cuMemsetD32_v2(dptr: CUdeviceptr, value: u32, n: usize) -> CUresult;
    pub fn cuMemsetD8_v2(dptr: CUdeviceptr, value: u8, n: usize) -> CUresult;

    // ---- Streams ----
    pub fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> CUresult;
    pub fn cuStreamCreateWithPriority(
        stream: *mut CUstream,
        flags: u32,
        priority: c_int,
    ) -> CUresult;
    pub fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    pub fn cuStreamQuery(stream: CUstream) -> CUresult;
    pub fn cuStreamGetPriority(stream: CUstream, priority: *mut c_int) -> CUresult;

    // ---- Events ----
    pub fn cuEventCreate(event: *mut CUevent, flags: u32) -> CUresult;
    pub fn cuEventDestroy_v2(event: CUevent) -> CUresult;
    pub fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;
    pub fn cuEventSynchronize(event: CUevent) -> CUresult;
    pub fn cuEventQuery(event: CUevent) -> CUresult;
    pub fn cuEventElapsedTime(millis: *mut f32, start: CUevent, end: CUevent) -> CUresult;

    // ---- Modules & kernel launch ----
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleLoadDataEx(
        module: *mut CUmodule,
        image: *const c_void,
        numOptions: c_int,
        options: *mut *mut c_void,
        optionValues: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuModuleGetGlobal_v2(
        dptr: *mut CUdeviceptr,
        bytes: *mut usize,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    pub fn cuLaunchKernel(
        func: CUfunction,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        stream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
}

// ============================================================================
// Convenience impls
// ============================================================================

impl CUdevice {
    /// Integer device ordinal.
    pub fn ordinal(self) -> i32 {
        self.0
    }
}

impl CUdeviceptr {
    /// Null device pointer.
    pub const NULL: Self = CUdeviceptr(0);
    /// `true` if the device pointer is zero (unallocated).
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

impl CUstream {
    /// Default stream handle.
    pub const NULL: Self = CUstream(std::ptr::null_mut());

    /// `true` if this is the default (null) stream.
    pub fn is_default(self) -> bool {
        self.0.is_null()
    }
}
