//! Raw FFI bindings to NVIDIA's CUDA Driver API and NVRTC runtime compiler.
//!
//! Everything in this module uses raw `dlsym`/`dlopen` — no external crates.
//! All symbols are resolved lazily on first use via
//! [`cuda_module`] / [`nvrtc_module`].

pub mod cuda;
pub mod nvrtc;

use crate::{Result, VugvaError, CUDA_SONAME, NVRTC_SONAME};
use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

// ============================================================================
// Raw dlopen: zero-dep ELF loader
// ============================================================================

extern "C" {
    fn dlopen(filename: *const c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, sym: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
    fn dlerror() -> *const c_char;
}

/// `RTLD_NOW` — resolve all symbols immediately.
const RTLD_NOW: i32 = 2;

// ============================================================================
// Lazy module loader
// ============================================================================

/// A handle to a single loaded shared library.
/// Stored in a `OnceLock` for thread-safe one-shot init.
pub struct LoadedLib {
    pub(crate) handle: *mut c_void,
}

// SAFETY: CUDA/NVRTC handles are process-local; Send is correct for
// transferring ownership across threads. Sync is not needed because
// library loading is serialized via OnceLock.
unsafe impl Send for LoadedLib {}
unsafe impl Sync for LoadedLib {}

/// Open a shared library by SONAME. Tries several common variants.
fn open_lib(soname: &str, fallbacks: &[&str]) -> Result<*mut c_void> {
    let candidates: Vec<&str> = std::iter::once(soname)
        .chain(fallbacks.iter().copied())
        .collect();
    for name in candidates {
        // Clear dlerror before each attempt so stale errors don't persist.
        // SAFETY: dlerror() is thread-safe and returns a valid pointer or null.
        unsafe {
            dlerror();
        }
        let bytes = match std::ffi::CString::new(name.as_bytes()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        // SAFETY: dlopen is called with a valid null-terminated string.
        let h = unsafe { dlopen(bytes.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            return Ok(h);
        }
    }
    // All candidates failed — return a single consolidated error.
    Err(VugvaError::LibLoad {
        library: "cuda/nvrtc",
        os_error: 0,
    })
}

/// Resolve a named symbol from a previously opened library.
///
/// # Safety
///
/// `handle` must be a valid `dlopen` handle. The returned pointer, if
/// non-null, must be transmuted to the correct function pointer type.
unsafe fn resolve(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
    let bytes = match std::ffi::CString::new(name) {
        Ok(c) => c,
        Err(_) => return None,
    };
    // SAFETY: handle is valid (checked by caller), name is null-terminated.
    let p = dlsym(handle, bytes.as_ptr());
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}

// ============================================================================
// Public re-exports
// ============================================================================

/// Singleton handle to `libcuda.so.1`. Resolved on first call.
pub fn cuda_module() -> Result<&'static LoadedLib> {
    static H: OnceLock<Result<LoadedLib>> = OnceLock::new();
    H.get_or_init(|| {
        let handle = open_lib(CUDA_SONAME, &["libcuda.so", "cuda"])?;
        // SAFETY: handle is non-null (checked by open_lib).
        Ok(LoadedLib { handle })
    })
    .as_ref()
    .map_err(|e| match e {
        VugvaError::LibLoad { .. } => VugvaError::LibLoad {
            library: CUDA_SONAME,
            os_error: 0,
        },
        _ => unreachable!(),
    })
}

/// Singleton handle to `libnvrtc.so`. Resolved on first call.
pub fn nvrtc_module() -> Result<&'static LoadedLib> {
    static H: OnceLock<Result<LoadedLib>> = OnceLock::new();
    H.get_or_init(|| {
        let handle = open_lib(
            NVRTC_SONAME,
            &[
                // CUDA 13
                "libnvrtc.so.13",
                // CUDA 12
                "libnvrtc.so.12",
                // CUDA 11
                "libnvrtc.so.11",
                // CUDA 10
                "libnvrtc.so.10",
                // Unversioned fallback
                "libnvrtc.so",
            ],
        )?;
        // SAFETY: handle is non-null (checked by open_lib).
        Ok(LoadedLib { handle })
    })
    .as_ref()
    .map_err(|e| match e {
        VugvaError::LibLoad { .. } => VugvaError::LibLoad {
            library: NVRTC_SONAME,
            os_error: 0,
        },
        _ => unreachable!(),
    })
}

/// Resolve a symbol from the **NVRTC** library and transmute to a typed
/// function pointer.
///
/// # Safety
///
/// The caller must ensure `T` matches the actual symbol signature loaded
/// from `libnvrtc.so`.
#[inline]
pub(crate) unsafe fn nvrtc_sym_to_fn<T>(name: &str) -> Result<T> {
    let lib = nvrtc_module()?;
    match resolve(lib.handle, name) {
        // SAFETY: transmute_copy is safe when T matches the symbol's type.
        Some(p) => Ok(std::mem::transmute_copy::<*mut c_void, T>(&p)),
        None => Err(VugvaError::LibLoad {
            library: "libnvrtc.so",
            os_error: 0,
        }),
    }
}
