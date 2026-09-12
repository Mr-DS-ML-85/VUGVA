# BUG.md — VUGVA Implementation Gaps, Bugs & Unimplemented Features

> Audit of `src/` against the paper (`paper/VUGVA_Paper.tex`).
> Generated 2026-07-26. No code modified.

---

## 1. PREFETCH DRAM→VRAM PATH IS A STUB

**File:** `src/prefetch.rs:144-150`

```rust
Tier::Dram => {
    // DRAM→VRAM: submit DMA descriptor (CPU-bypass)
    // The DMA engine handles the actual transfer.
    // In a full implementation this would write to the
    // DMA command ring; here we record the intent.
    // See dma.rs for the ring implementation.
}
```

**Paper §3.2 Algorithm 3, line 5:**
```
else if T_s = DRAM then
    DMA_SubmitDescriptor(DRAM_{G_s}, VRAM_{G_c})
```

**Bug:** The DRAM→VRAM prefetch path does nothing. When the Look-Ahead Attention Tracking engine encounters a future layer residing in DRAM, it silently skips the transfer. This means the paper's latency-hiding equation (§3.2, Eq. 1) cannot be satisfied for DRAM-resident weights — `T_transport` is effectively ∞ for those layers, breaking the `T_compute > T_transport` invariant.

**Impact:** Any model with weights split across VRAM and DRAM (the entire point of VUGVA v2) will have cold-cache misses at inference time instead of prefetched data.

---

## 2. PREFETCH SSD→VRAM PATH IS A STUB

**File:** `src/prefetch.rs:151-155`

```rust
Tier::Ssd => {
    // SSD→VRAM: GPUDirect Storage read
    // GDS would be initialized via dma.rs.
    // This path submits an async GDS read descriptor.
}
```

**Paper §3.3:**
> VUGVA extends NVIDIA's GPUDirect Storage (GDS) framework — originally designed for NVMe→VRAM transfers — to also handle DRAM→VRAM movements.

**Bug:** The SSD→VRAM prefetch path is a no-op. No GPUDirect Storage descriptors are submitted. When a page is in the Cold tier, the prefetcher ignores it entirely.

**Impact:** The three-tier hierarchy (VRAM→DRAM→SSD) promised in the paper is functionally two-tier. The SSD spill tier exists in the state machine but is never actively used for prefetch.

---

## 3. SSD TIER IN tiered.rs IS A DRAM COPY-PASTE

**File:** `src/tiered.rs:280-312`

```rust
Tier::Ssd => {
    // SSD → DRAM → VRAM (two-step)
    // For now, treat as DRAM path
    let dram = maybe_dram_chunk
        .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
    // ... same H2D copy as Tier::Dram ...
}
```

**Bug:** The SSD access path reads from `maybe_dram_chunk` (DRAM), not from an actual SSD offset. The `Page.ssd_offset` field is allocated but never read from in `access()`. This means:
1. Data spilled to SSD is never actually loaded from NVMe storage
2. The `Tier::Ssd` → `Tier::Warm` → `Tier::Resident` promotion chain is broken
3. No GPUDirect Storage API is called anywhere in the codebase

**Impact:** The SSD spill tier is pure fiction — it exists in the state machine diagram but has zero data-plane implementation.

---

## 4. demote() USES WRONG DMA METHOD

**File:** `src/tiered.rs:317-349`

```rust
pub fn demote(&mut self, name: &str) -> Result<()> {
    // ...
    for chunk in &page.vram_chunks {
        // ...
        self.dma.submit_vram_to_vram(
            gpu_idx, gpu_idx,  // <-- src and dst are the SAME GPU
            chunk.device_ptr,
            dram.base_ptr as u64,  // <-- host pointer treated as device pointer
            chunk.size_bytes,
            0,
        )?;
    }
}
```

**Bug 1:** `submit_vram_to_vram` is called with `src_gpu == dst_gpu` (same GPU), which is a self-copy, not a VRAM→DRAM writeback.

**Bug 2:** `dram.base_ptr` is a host pointer (from `mmap`), but it's cast to `u64` and passed as if it were a CUDA device pointer. The DMA descriptor's `dst_addr` will contain a host virtual address, which the GPU's DMA engine cannot address. This will either segfault or silently corrupt memory.

**Paper §4.1:** Demotion should write VRAM data back to DRAM via DMA. The correct path is `cuMemcpyDtoH` (device→host async) or a GDS write descriptor.

---

## 5. CUDA RUNTIME INTERCEPTOR IS NOT IMPLEMENTED

**Paper §3.1 Algorithm 1:**
> Intercepts `cudaMalloc`, `cudaMallocManaged`, and `cudaFree` commands using dynamic library preloading (`LD_PRELOAD`).

**File:** `src/` — no `interceptor.rs` exists.

**Missing:** The `LD_PRELOAD` shim library (`libvugva_preload.so`) that:
1. Interposes `cudaMalloc` / `cudaMallocManaged` / `cudaFree`
2. Forwards calls to VMT when allocation exceeds local VRAM
3. Returns virtual pointers to the framework transparently

**Impact:** Without this, VUGVA cannot be used as a drop-in replacement for CUDA memory management. Users must explicitly use the `VugvaEngine` API instead of PyTorch's default allocator.

---

## 6. DmaRing SUBMIT IS NOT TRULY LOCK-FREE

**File:** `src/dma.rs:163-188`

```rust
pub fn submit(&self, desc: &DmaDescriptor) -> Result<u32> {
    if self.is_full() { ... }
    let producer = self.completion.submitted.load(Ordering::Acquire);
    let slot = (producer & self.mask) as usize;
    unsafe {
        let ptr = self.ring.as_ptr().add(slot) as *mut DmaDescriptor;
        std::ptr::write_volatile(ptr, desc.clone());
    }
    std::sync::atomic::fence(Ordering::Release);
    self.completion.submitted.store(producer + 1, Ordering::Release);
    Ok(slot as u32)
}
```

**Bug:** This is a single-producer ring, but the producer index is read via `load` and then `store` (not `fetch_add`). If two threads call `submit` concurrently, they'll read the same `producer` value and write to the same slot, corrupting the ring. The comment says "lock-free" but it requires external synchronization.

**Also:** `is_full()` reads `submitted` then `consumer`, but another thread could drain between the two reads, making the full-check racy.

---

## 7. tiered.rs ACCESS() HAS USE-AFTER-MOVE BUGS

**STATUS: FIXED.** `access()` now resolves a *peer* chunk in step 1 and returns `UnknownAllocation` when the tier field and the VMT disagree, instead of `unwrap()`-ing the `None` that selected the branch.

**File:** `src/tiered.rs:196-204`

```rust
Tier::Vram => {
    if let Some(chunk) = maybe_vram_chunk {
        self.vmt.lookup_mut(name).unwrap().touch(now);
        return Ok(chunk.device_ptr);
    }
    // VRAM but not on this GPU: allocate locally and copy via sync memcpy
    let src = maybe_vram_chunk.unwrap();  // <-- BUG: already moved/consumed above
```

**Bug:** `maybe_vram_chunk` is moved in the `if let Some(chunk)` branch. If that branch doesn't execute (chunk is `None`), the code falls through and calls `.unwrap()` on the already-consumed `Option`, which will always panic.

In practice this is "reachable" only when the page is in `Tier::Vram` but has no chunk for the current GPU — a valid state in multi-GPU setups. The code will panic at `maybe_vram_chunk.unwrap()`.

---

## 8. tiered.rs ACCESS() CREATES AND DESTROYS CONTEXTS ON EVERY CALL

**STATUS: FIXED.** `src/context.rs` holds one retained *primary* context per GPU for the pool's lifetime; `set_context` is now a pointer store. Verified by `paper_repeated_access_does_not_leak_context`: 128 promotions cost 0.0 MB beyond the 128 MB of payload (measured 97.8 MB per leaked context before).

**File:** `src/tiered.rs:388-396`

```rust
fn set_context(&self, gpu_idx: usize) -> Result<()> {
    let dev = CUdevice(self.cluster.ordinals[gpu_idx]);
    let mut ctx = CUcontext(std::ptr::null_mut());
    unsafe {
        check_cu("cuCtxCreate_v2", cuCtxCreate_v2(&mut ctx, 0, dev))?;
        check_cu("cuCtxSetCurrent", cuCtxSetCurrent(ctx))?;
    }
    Ok(())
}
```

**Bug:** `cuCtxCreate_v2` allocates a new CUDA context every call, but the context is never stored or destroyed. This leaks CUDA contexts (each ~24MB of driver overhead). After enough `access()` calls, the process will hit the CUDA context limit and fail.

The V1 `UnifiedAllocator` correctly stores contexts in `self.contexts`, but `TieredPool` recreates them every time.

---

## 9. prefretch.rs INFLIGHT TRACKING IS DEAD CODE

**File:** `src/prefetch.rs:66-68, 162-174`

```rust
pub struct LookAheadPrefetcher {
    depth: usize,
    inflight: Vec<PrefetchJob>,  // <-- never populated in prefetch_ahead()
}
```

In `prefetch_ahead()`, jobs are dispatched but never added to `self.inflight`. The `sync_all()` method drains an always-empty vector.

**Bug:** There is no way to track or wait on in-flight prefetches. The prefetcher fires transfers and immediately returns without any completion tracking. `sync_all()` is a no-op.

---

## 10. prefetch.rs PEER COPY USES WRONG INDEX TYPE

**File:** `src/prefetch.rs:127-128`

```rust
let src_idx = src_chunk.gpu_ordinal.try_into().unwrap_or(0usize);
let _dst_idx = current_gpu.try_into().unwrap_or(0usize);
```

**Bug:** `gpu_ordinal` is `i32`, `try_into()` to `usize` is fine, but `current_gpu` is also `i32`. The `try_into().unwrap_or(0usize)` silently defaults to GPU 0 on negative ordinals instead of returning an error. If a CUDA device has ordinal -1 (error value), the prefetch silently targets the wrong GPU.

**Also:** `_dst_idx` is computed but never used — the dst context is passed directly as `dst_ctx`, making this variable completely dead code.

---

## 11. NO THREAD SAFETY (Send/Sync) ON TieredPool

**File:** `src/tiered.rs:38-53`

`TieredPool` contains `DramPool` which holds a raw `usize` pointer (`base_ptr`). Rust's auto-trait implementation will NOT make `TieredPool: Send` because of the raw pointer.

**Missing:** `unsafe impl Send for TieredPool {}` and `unsafe impl Sync for TieredPool {}` with documented safety invariants. This prevents using `TieredPool` across threads, which is required for the background sweep (`background_sweep`) to run on a dedicated thread while inference runs on another.

---

## 12. NO UNIT TESTS FOR 7 OF 10 MODULES

| Module | Unit Tests | Status |
|--------|-----------|--------|
| `vmt.rs` | 9 tests | ✅ Complete |
| `dma.rs` | 6 tests | ✅ Complete |
| `gpu.rs` | 6 tests | ✅ Complete |
| `allocator.rs` | 0 tests | ❌ Missing |
| `tiered.rs` | 0 tests | ❌ Missing |
| `prefetch.rs` | 0 tests | ❌ Missing |
| `streams.rs` | 0 tests | ❌ Missing |
| `nvrtc_kernel.rs` | 0 tests | ❌ Missing |
| `ffi/mod.rs` | 0 tests | ❌ Missing |
| `ffi/cuda.rs` | 0 tests | ❌ Missing |
| `ffi/nvrtc.rs` | 0 tests | ❌ Missing |

**Impact:** 23 unit tests exist, but the 19 integration tests in `tests/hardware.rs` require real GPU hardware. The CI pipeline (`ci.yml`) only runs `fmt`, `clippy`, and `check` — it never runs `cargo test` because there's no way to test without a GPU. Modules like `tiered.rs`, `prefetch.rs`, and `allocator.rs` have pure-logic code that could be tested without hardware (mock the FFI layer).

---

## 13. CI DOES NOT RUN TESTS

**File:** `.github/workflows/ci.yml`

```yaml
jobs:
  fmt: ...
  clippy: ...
  check: ...
  # No test job
```

The CI pipeline never runs `cargo test`. The `check` job only verifies compilation. This means regressions in unit test logic (e.g., state machine transitions, DMA ring logic) will never be caught in CI.

---

## 14. SAFETY COMMENTS MISSING ON ~30 UNSAFE BLOCKS

Across the codebase, approximately 30 `unsafe` blocks lack `// SAFETY:` documentation. Key offenders:

- `src/streams.rs` — 10 unsafe blocks, 0 safety comments
- `src/allocator.rs` — 7 unsafe blocks, 0 safety comments
- `src/nvrtc_kernel.rs` — 10 unsafe blocks, 0 safety comments
- `src/tiered.rs` — 7 unsafe blocks, 1 safety comment (line 510)

**Impact:** Violates Rust API guidelines and makes auditing unsafe correctness impossible.

---

## 15. lib.rs DECLARES `#![allow(dead_code)]` GLOBALLY

**File:** `src/lib.rs:34`

```rust
#![allow(dead_code)]
```

**Bug:** This suppresses dead code warnings for the entire crate, hiding:
- Unused fields in structs (e.g., `PrefetchJob.name`, `PrefetchJob.dst_gpu`)
- Unused imports
- Functions that are defined but never called

This makes it impossible to identify code that was written but never integrated.

---

## 16. DmaDescriptor._pad MAKES THE STRUCT NON-ZEROABLE

**File:** `src/dma.rs:43`

```rust
_pad: [u8; 28],
```

**Minor bug:** The `_pad` field prevents `#[derive(Default)]` from being derived (since `[u8; 28]` doesn't impl `Default`... actually it does). More importantly, the padding bytes are initialized to 0 in every allocation, which is correct but wasteful. The pad should use `MaybeUninit` or be validated at compile time to ensure the struct is truly 64 bytes without relying on runtime initialization.

---

## 17. GpuCluster::discover() LEAKS CUDA CONTEXTS

**STATUS: FIXED.** The free-VRAM probe uses a scoped `PrimaryContext` + guard, released on scope exit.

**File:** `src/gpu.rs:279-287`

```rust
let mut ctx = CUcontext(std::ptr::null_mut());
unsafe {
    cuCtxCreate_v2(&mut ctx, 0, dev);
    let mut free: usize = 0;
    let mut tot: usize = 0;
    cuMemGetInfo_v2(&mut free, &mut tot);
    info.free_vram = free;
    cuCtxDestroy_v2(ctx);
}
```

**Bug:** `cuCtxCreate_v2` return code is not checked. If context creation fails (e.g., GPU in exclusive mode), `cuMemGetInfo_v2` is called on a null context, which will return an error code but the error is silently ignored. `free_vram` will remain 0, making the GPU appear to have no VRAM.

---

## 18. VirtualMemoryTable HAS NO CAPACITY LIMIT

**File:** `src/vmt.rs:197-205`

The VMT uses an unbounded `HashMap<String, Page>`. There is no maximum page count or total memory budget.

**Impact:** A misbehaving framework could allocate millions of tiny pages, causing the VMT to consume unbounded host memory for metadata. The paper implies the VMT should have a fixed capacity matching the physical memory pool.

---

## 19. DramPool ALLOCATION NEVER FAILS GRACEFULLY

**STATUS: FIXED.** New `VugvaError::DramOom { requested, available, capacity }` replaces the device-OOM masquerade, and carries enough to separate exhaustion from fragmentation. Covered by `dram_allocate_reports_exhaustion_as_a_host_error` and `dram_oom_distinguishes_fragmentation_from_exhaustion`.

**Also fixed here (the larger problem behind it):** the pool was bump-only — `offset` never moved backwards and nothing freed device memory, so capacity was a *lifetime* budget. `DramPool` now has a coalescing best-fit free list with high-water retraction, `TieredPool` recycles device blocks, and `TieredPool::deallocate` returns both. `paper_allocations_are_reclaimed_and_reused` churns 512 MiB through a 64 MiB pool and retains 4.1 MB of VRAM.

**File:** `src/tiered.rs:71-83`

```rust
fn allocate(&mut self, bytes: usize) -> Result<usize> {
    let aligned = (bytes + 63) & !63;
    if self.offset + aligned > self.capacity {
        return Err(VugvaError::CudaError {
            fn_name: "DramPool::allocate",
            code: CUDA_ERROR_OUT_OF_MEMORY,
        });
    }
    // ...
}
```

**Minor:** Returns `CUDA_ERROR_OUT_OF_MEMORY` for a DRAM allocation failure. This is semantically misleading — the error is a host memory OOM, not a CUDA device OOM. The error code should be `std::io::Error` or a dedicated `VugvaError::DramOom` variant.

---

## 20. DmaCompletion DOES NOT HANDLE WRAP-AROUND CORRECTLY

**File:** `src/dma.rs:76-85`

```rust
pub fn is_done(&self) -> bool {
    self.submitted.load(Ordering::Acquire) == self.completed.load(Ordering::Acquire)
}
pub fn pending(&self) -> u32 {
    self.submitted.load(Ordering::Acquire)
        .saturating_sub(self.completed.load(Ordering::Acquire))
}
```

**Bug:** `submitted` and `completed` are `AtomicU32` and are never reset. After ~4 billion transfers, they wrap to 0. If `submitted` wraps before `completed` is checked, `pending()` will underflow to `u32::MAX` (via `saturating_sub`), falsely reporting billions of pending transfers.

The ring has 1024 slots, so this is unlikely in practice, but the design doesn't handle it. A proper SPSC ring should use sequence numbers or modular arithmetic.

---

## 21. CUDA FFI LINKS `libcuda.so` DIRECTLY

**STATUS: FIXED.** Both CUDA and NVRTC are `dlopen`'d at first use. `readelf -d` on the hardware test binary now shows only `libgcc_s.so.1`, `libc.so.6` and `ld-linux-x86-64.so.2` — no `libcuda`, no `libnvrtc` — and the driver is still reached at runtime.

**File:** `src/ffi/cuda.rs:114`

```rust
#[link(name = "cuda")]
extern "C" {
    pub fn cuInit(flags: u32) -> CUresult;
    // ...
}
```

**Bug:** Despite the paper claiming "zero external dependencies" and using `dlopen` for runtime loading, the FFI module has `#[link(name = "cuda")]` which creates a hard link-time dependency on `libcuda.so`. This means the library will fail to load on systems without CUDA installed, even if the CUDA functions are never called.

The `ffi/mod.rs` loader (`dlopen`/`dlsym`) is the correct approach, but `cuda.rs` contradicts it with `#[link(name = "cuda")]`. The two approaches conflict: either use `dlopen` OR static linking, not both.

---

## 22. DmaRing SUBMIT CONSUMES THE DESCRIPTOR (UNNECESSARY CLONE)

**File:** `src/dma.rs:163`

```rust
pub fn submit(&self, desc: &DmaDescriptor) -> Result<u32> {
    // ...
    std::ptr::write_volatile(ptr, desc.clone());
```

**Minor:** The descriptor is `Clone`, and every `submit` clones it into the ring. Since `DmaDescriptor` is `#[repr(C, align(64))]` and contains 64 bytes, this is a 64-byte memcpy. Not a bug, but the API accepts `&DmaDescriptor` (borrow) then clones internally — it should either take ownership or use `Copy` (which it can't due to alignment requirements).

---

## 23. NUMA BINDING USES WRONG MASK SIZE

**STATUS: FIXED.** The nodemask is sized to the node (`numa_node / 64 + 1` words) instead of `1u64 << numa_node`, which was UB for any node >= 64. `mbind` is also issued as a raw syscall now: glibc does not export it (it is a libnuma symbol), so the previous `extern "C"` declaration could never link.

**File:** `src/tiered.rs:483-493`

```rust
let nodemask: u64 = 1u64 << numa_node;
unsafe {
    mbind(
        ptr,
        aligned_size,
        mode,
        &nodemask,
        64, // maxnode
        flags,
    );
}
```

**Bug:** `maxnode` is hardcoded to 64, but `nodemask` is a `u64` (64 bits). The `mbind` man page says `maxnode` is the maximum node number + 1, and the nodemask bitmap has `(maxnode + 63) / 64` u64 elements. With `maxnode=64` and a single `u64`, the kernel reads exactly 8 bytes, which is correct for nodes 0-63. However, if `numa_node >= 64`, the shift `1u64 << numa_node` overflows (undefined behavior in Rust for shifts >= 64).

**Impact:** Systems with >64 NUMA nodes (rare but possible in large servers) will silently corrupt the nodemask.

---

## 24. prefetch_ahead DOES NOT CHECK schedule.len() BOUNDS

**File:** `src/prefetch.rs:109-116`

```rust
let depth = self
    .depth
    .min(schedule.len().saturating_sub(current_layer + 1));

for offset in 1..=depth {
    let future_layer = current_layer + offset;
    if future_layer >= schedule.len() {
        break;
    }
```

**Minor:** The `depth` calculation uses `saturating_sub` which returns 0 if `current_layer + 1 > schedule.len()`. This is correct, but the `break` inside the loop is redundant — `depth` already clamps the range. Not a bug, but dead logic.

---

## 25. VugvaEngine::free() DOES NOT FREE VRAM ON ALL GPUs

**STATUS: FIXED.** Errors are collected rather than discarded (all chunks still attempted, first error returned), and the `unwrap_or(i)` that substituted the loop counter for an unresolved GPU index — freeing a pointer against a foreign context, which is UB rather than an error — now returns `InvalidGpu`.

**File:** `src/allocator.rs:352-364`

```rust
pub fn free(&mut self, name: &str) -> Result<()> {
    let page = self.vmt.remove(name)
        .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;

    for (i, chunk) in page.vram_chunks.iter().enumerate() {
        let gpu_idx = self.cluster.index_of(chunk.gpu_ordinal).unwrap_or(i);
        let _ = self.allocator.free_device(gpu_idx, chunk.device_ptr);
    }
    Ok(())
}
```

**Bug:** The `_ = self.allocator.free_device(...)` silently discards the error from `free_device`. If CUDA reports an error during deallocation (e.g., double-free, invalid pointer), the error is swallowed and the caller believes the free succeeded.

---

## 26. NO BENCHMARK HARNESS

**Missing:** No `benches/` directory. The paper claims performance benchmarks (Table 5, 6, 7) but the codebase has no formal benchmarking infrastructure (criterion, divan, etc.). The demo (`examples/demo.rs`) prints timings but doesn't collect statistically rigorous measurements.

---

## 27. DmaEngine HAS NO ACTUAL DMA EXECUTION

**File:** `src/dma.rs:237-324`

The `DmaEngine` manages rings and provides `submit_*` methods, but there is no consumer thread or DMA execution engine. The `peek()` and `ack()` methods exist but are never called outside of tests.

**Bug:** Descriptors are submitted to the ring but nothing reads from the ring to execute the transfers. The `DmaEngine` is a write-only queue — data goes in but never comes out.

**Paper §3.3:** "the DMA engine autonomously transfers megabytes of tensor data without further CPU intervention." In reality, the DMA engine is simulated — actual data movement uses `cuMemcpyHtoDAsync_v2` (CPU-mediated) in `tiered.rs:258`.

---

## 28. TieredPool::access() VRAM→VRAM PATH USES HOST STAGING BUFFER

**STATUS: FIXED.** Replaced with `cuMemcpyPeerAsync`, which uses the P2P link when the pair is peer-enabled and lets the driver stage it otherwise. The old path bounced the whole tensor through a host `Vec` — two PCIe crossings plus a CPU copy of megabytes, in the module whose premise is 72 bytes of CPU involvement.

**File:** `src/tiered.rs:210-230`

```rust
// VRAM but not on this GPU: allocate locally and copy via sync memcpy
self.set_context(gpu_idx)?;
let mut host_buf = vec![0u8; size];
unsafe {
    // Copy src → host
    cuMemcpyDtoH_v2(...);
    // Copy host → dst
    cuMemcpyHtoD_v2(...);
}
```

**Bug:** The VRAM→VRAM peer copy path routes data through a CPU host buffer (`vec![0u8; size]`). This is the exact CPU-mediated path the paper claims to eliminate. The correct approach is `cuMemcpyPeerAsync` (which `VugvaEngine::access()` in `allocator.rs` does use correctly).

**Impact:** On multi-GPU setups, cross-GPU data movement goes through CPU memory, defeating the CPU-bypass guarantee.

---

## 29. compile_and_load() HARDCODES KERNEL NAME

**File:** `src/nvrtc_kernel.rs:59`

```rust
let c_name = std::ffi::CString::new("vugva_kernel").unwrap();
```

**Bug:** The program name passed to `nvrtcCreateProgram` is always `"vugva_kernel"` regardless of which kernel is being compiled. This makes NVRTC error messages misleading — both `memcpy_peer` and `tier_promote` will appear as "vugva_kernel" in compile logs.

---

## 30. CudaEvent::new_blocking() USES WRONG FLAG

**File:** `src/streams.rs:112-113`

```rust
check_cu(
    "cuEventCreate",
    cuEventCreate(&mut event, CU_EVENT_BLOCKING_SYNC),
)?;
```

**Minor:** `CU_EVENT_BLOCKING_SYNC` (0x01) means the host-side synchronize call blocks the calling thread. This is the correct flag for a blocking event, but the naming is confusing — "blocking sync" doesn't mean the event is "blocking" in the sense of preventing GPU progress. It means the host `cuEventSynchronize` call uses a blocking wait rather than spinning. This is correct behavior but the docstring "suitable for host-side waits" is imprecise.

---

## SUMMARY TABLE

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | **CRITICAL** | prefetch.rs | DRAM→VRAM prefetch is a stub |
| 2 | **CRITICAL** | prefetch.rs | SSD→VRAM prefetch is a stub |
| 3 | **CRITICAL** | tiered.rs | SSD tier is a DRAM copy-paste |
| 4 | **CRITICAL** | tiered.rs | demote() uses host ptr as device ptr |
| 5 | **CRITICAL** | (missing) | LD_PRELOAD interceptor not implemented |
| 6 | HIGH | dma.rs | DmaRing submit is not thread-safe |
| 7 | HIGH | tiered.rs | access() has use-after-move panic |
| 8 | HIGH | tiered.rs | set_context() leaks CUDA contexts |
| 9 | HIGH | prefetch.rs | Inflight tracking is dead code |
| 10 | MEDIUM | prefetch.rs | Silent default on bad ordinal |
| 11 | MEDIUM | tiered.rs | No Send/Sync impl |
| 12 | MEDIUM | (all) | 7/10 modules have zero unit tests |
| 13 | MEDIUM | ci.yml | CI never runs cargo test |
| 14 | MEDIUM | (all) | ~30 unsafe blocks lack SAFETY comments |
| 15 | LOW | lib.rs | Global allow(dead_code) hides issues |
| 16 | LOW | dma.rs | Padding init is wasteful |
| 17 | LOW | gpu.rs | discover() ignores cuCtxCreate errors |
| 18 | LOW | vmt.rs | VMT has no capacity limit |
| 19 | LOW | tiered.rs | DRAM OOM uses CUDA error code |
| 20 | LOW | dma.rs | DmaCompletion doesn't handle u32 wrap |
| 21 | HIGH | ffi/cuda.rs | #[link] contradicts dlopen approach |
| 22 | LOW | dma.rs | Unnecessary clone in submit |
| 23 | LOW | tiered.rs | NUMA mask overflow for node >= 64 |
| 24 | LOW | prefetch.rs | Redundant bounds check |
| 25 | LOW | allocator.rs | free() silently discards errors |
| 26 | MEDIUM | (missing) | No benchmark harness |
| 27 | **CRITICAL** | dma.rs | DMA engine has no consumer/executor |
| 28 | **CRITICAL** | tiered.rs | VRAM→VRAM goes through CPU buffer |
| 29 | LOW | nvrtc_kernel.rs | Hardcoded program name |
| 30 | LOW | streams.rs | Misleading docstring |

**Critical (7):** Issues that prevent the system from working as described in the paper.
**High (4):** Correctness bugs that cause panics, leaks, or silent data corruption.
**Medium (5):** Testing and quality gaps.
**Low (14):** Minor issues and code quality concerns.
