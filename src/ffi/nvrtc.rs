//! Raw FFI declarations for **NVRTC** — the runtime CUDA kernel compiler.
//!
//! NVRTC lets us take a CUDA source string at runtime, compile it to PTX
//! or cubin (driver-loadable image), and launch kernels without a `.cu`
//! compile step in our build system. VUGVA uses this to compile a minimal
//! memcpy_peer kernel that respects peer-to-peer access flags without
//! relying on the CUDA Runtime's cudart library.

use std::ffi::{c_char, c_void};

/// `nvrtcResult` integer return code.
pub type nvrtcResult = i32;
/// `NVRTC_SUCCESS`
pub const NVRTC_SUCCESS: i32 = 0;

/// Opaque handle to a compiled program.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct nvrtcProgram(pub *mut c_void);

extern "C" {
    // ---- Version ----
    pub fn nvrtcVersion(major: *mut i32, minor: *mut i32) -> nvrtcResult;

    // ---- Compile ----
    pub fn nvrtcCreateProgram(
        prog: *mut nvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        numHeaders: i32,
        headers: *mut *const c_char,
        includeNames: *mut *const c_char,
    ) -> nvrtcResult;
    pub fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
    pub fn nvrtcCompileProgram(
        prog: nvrtcProgram,
        numOptions: i32,
        options: *mut *const c_char,
    ) -> nvrtcResult;
    pub fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;
    pub fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSize: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut c_char) -> nvrtcResult;
    pub fn nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSize: *mut usize) -> nvrtcResult;

    // ---- Error & log ----
    pub fn nvrtcGetErrorString(result: nvrtcResult) -> *const c_char;
    pub fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;
    pub fn nvrtcGetProgramLogSize(prog: nvrtcProgram, logSize: *mut usize) -> nvrtcResult;

    // ---- Specialization ----
    pub fn nvrtcAddNameExpression(prog: nvrtcProgram, name: *const c_char) -> nvrtcResult;
    pub fn nvrtcGetLoweredName(
        prog: nvrtcProgram,
        name: *const c_char,
        basename: *mut *const c_char,
    ) -> nvrtcResult;
}
