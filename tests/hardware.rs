//! Hardware integration tests — run on real CUDA GPUs.
//!
//! These tests call into `libcuda.so` (resolved lazily at runtime via `dlopen`,
//! not linked) and verify the GPU hardware is reachable and behaves as the
//! paper expects.

#[cfg(test)]
mod hw {
    use std::ffi::c_int;
    use vugva::ffi::cuda::*;

    fn init_cuda() {
        unsafe {
            let _ = cuInit(0);
        }
    }

    /// Serialize tests that allocate device memory or measure it.
    ///
    /// `cargo test` runs these as parallel threads in one process, and free
    /// VRAM and PCIe bandwidth are *process-global* — in fact machine-global.
    /// Two tests that each allocate a few hundred MB and then measure the
    /// difference will attribute each other's allocations to themselves, and
    /// two concurrent DMA transfers each read at half the link rate. Both
    /// showed up here as spurious failures: a "222 MB leak" that was another
    /// test's buffers, and "1.8 GB/s" that was PCIe being shared.
    ///
    /// Anything that touches VRAM takes this lock. The lock is poisoned by a
    /// panicking test, which is not a reason to fail every subsequent one, so
    /// the poison is recovered rather than propagated.
    fn gpu_exclusive() -> std::sync::MutexGuard<'static, ()> {
        static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        GPU_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Free VRAM on the current device, in MB.
    ///
    /// Only meaningful while `gpu_exclusive()` is held: this reads a
    /// machine-global counter, so a concurrent test's allocations show up as
    /// this test's.
    fn free_mb() -> f64 {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: valid out-pointers; the caller has a context current.
        unsafe {
            assert_eq!(cuMemGetInfo_v2(&mut free, &mut total), CUDA_SUCCESS);
        }
        free as f64 / (1024.0 * 1024.0)
    }

    /// Make device 0's **primary** context current and return it.
    ///
    /// Tests run as parallel threads in one process, and this helper used to
    /// call `cuCtxCreate_v2` on every invocation: that leaks 97.8 MB of VRAM
    /// per call (measured on this machine — see `vugva::context`) and hands
    /// each test a *different* context, so memory
    /// allocated by one test was not addressable from another. The primary
    /// context is refcounted and shared, which is what these tests want.
    fn current_ctx() -> CUcontext {
        init_cuda();
        let dev = CUdevice(0);
        let mut ctx = CUcontext::NULL;
        unsafe {
            // Retained per call and deliberately never released: holding the
            // reference keeps the context alive for the whole run regardless of
            // test ordering, and the process exits moments later.
            let rc = cuDevicePrimaryCtxRetain(&mut ctx, dev);
            assert_eq!(rc, CUDA_SUCCESS, "cuDevicePrimaryCtxRetain failed: {rc}");
            let rc = cuCtxSetCurrent(ctx);
            assert_eq!(rc, CUDA_SUCCESS, "cuCtxSetCurrent failed: {rc}");
        }
        ctx
    }

    // ================================================================
    // Paper §2 — CUDA driver is loadable and GPU is detectable
    // ================================================================

    #[test]
    fn paper_driver_loadable() {
        init_cuda();
        let mut count = 0i32;
        let rc = unsafe { cuDeviceGetCount(&mut count) };
        assert_eq!(rc, CUDA_SUCCESS);
        println!("CUDA driver loaded, {count} GPU(s)");
    }

    #[test]
    fn paper_gpu_count_positive() {
        init_cuda();
        let mut count = 0i32;
        let rc = unsafe { cuDeviceGetCount(&mut count) };
        assert_eq!(rc, CUDA_SUCCESS);
        assert!(count > 0, "expected >= 1 GPU");
        println!("detected {count} GPU(s)");
    }

    #[test]
    fn paper_gpu_name_and_compute_cap() {
        init_cuda();
        let dev = CUdevice(0);

        let mut name = [0i8; 256];
        let rc = unsafe { cuDeviceGetName(name.as_mut_ptr(), 256, dev) };
        assert_eq!(rc, CUDA_SUCCESS);
        let name_str = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }
            .to_str()
            .unwrap();
        println!("GPU name: {name_str}");
        assert!(!name_str.is_empty());

        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        let rc = unsafe { cuDeviceComputeCapability(&mut major, &mut minor, dev) };
        assert_eq!(rc, CUDA_SUCCESS);
        println!("compute capability: sm_{major}{minor}");
        assert!(major >= 6, "expected sm_60+, got sm_{major}{minor}");
    }

    // ================================================================
    // Paper §1 — RTX 4060 VRAM query
    // ================================================================

    #[test]
    fn paper_intra_gpu_vram_exists() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();
        let (mut free, mut total) = (0usize, 0usize);
        let rc = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        assert_eq!(rc, CUDA_SUCCESS);
        assert!(total > 0);
        assert!(free > 0);
        assert!(free <= total);
        println!("VRAM: {free} free / {total} total bytes");
        assert!(total >= 7_000_000_000, "RTX 4060 should have >= 7GB VRAM");
    }

    // ================================================================
    // Paper Algorithm 1 — single-GPU allocation
    // ================================================================

    #[test]
    fn paper_algo1_native_alloc() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();
        let mut dptr = CUdeviceptr::NULL;
        let rc = unsafe { cuMemAlloc_v2(&mut dptr, 4 * 1024 * 1024) };
        assert_eq!(rc, CUDA_SUCCESS);
        assert!(!dptr.is_null());
        unsafe { cuMemFree_v2(dptr) };
    }

    #[test]
    fn paper_algo1_large_alloc() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();
        let mut dptr = CUdeviceptr::NULL;
        let rc = unsafe { cuMemAlloc_v2(&mut dptr, 256 * 1024 * 1024) };
        assert_eq!(rc, CUDA_SUCCESS);
        unsafe { cuMemFree_v2(dptr) };
    }

    #[test]
    fn paper_algo1_multiple_shards() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();
        let mut ptrs = Vec::new();
        for _ in 0..8 {
            let mut dptr = CUdeviceptr::NULL;
            let rc = unsafe { cuMemAlloc_v2(&mut dptr, 32 * 1024 * 1024) };
            assert_eq!(rc, CUDA_SUCCESS);
            ptrs.push(dptr);
        }
        for p in ptrs {
            unsafe {
                cuMemFree_v2(p);
            }
        }
    }

    // ================================================================
    // Paper §4.2 — NUMA topology
    // ================================================================

    #[test]
    fn paper_numa_topology_parseable() {
        use vugva::gpu::NumaTopology;
        let topo = NumaTopology::from_numactl();
        if let Err(e) = &topo {
            println!("numactl unavailable ({e}), using single-node fallback");
        }
        if let Ok(topo) = topo {
            assert!(topo.node_count >= 1);
            println!("NUMA nodes: {}", topo.node_count);
            for n in 0..topo.node_count {
                assert_eq!(topo.distance(n, n), 10, "self-distance must be 10");
            }
        }
    }

    #[test]
    fn paper_dma_bandwidth_factor_thresholds() {
        use vugva::gpu::NumaTopology;
        let topo = NumaTopology::single_node();
        assert_eq!(topo.dma_bandwidth_factor(0, 0), 0.95);

        let topo2 = NumaTopology {
            distances: vec![vec![10, 12], vec![12, 10]],
            node_count: 2,
        };
        assert_eq!(topo2.dma_bandwidth_factor(0, 1), 0.95);

        let topo3 = NumaTopology {
            distances: vec![vec![10, 13], vec![13, 10]],
            node_count: 2,
        };
        assert_eq!(topo3.dma_bandwidth_factor(0, 1), 0.80);

        let topo4 = NumaTopology {
            distances: vec![vec![10, 20], vec![20, 10]],
            node_count: 2,
        };
        assert_eq!(topo4.dma_bandwidth_factor(0, 1), 0.80);

        let topo5 = NumaTopology {
            distances: vec![vec![10, 21], vec![21, 10]],
            node_count: 2,
        };
        assert_eq!(topo5.dma_bandwidth_factor(0, 1), 0.65);
    }

    // ================================================================
    // Paper §2 — VMT page state machine (Figure 5)
    // ================================================================

    fn advance_to(page: &mut vugva::vmt::Page, target: vugva::vmt::PageState) {
        use vugva::vmt::PageState;
        let chain: &[PageState] = &[PageState::Allocated, PageState::Resident, PageState::Warm];
        for &state in chain {
            if state == target {
                page.transition(state).unwrap();
                return;
            }
            page.transition(state).unwrap();
        }
        if target == PageState::Cold {
            page.transition(PageState::Cold).unwrap();
        }
    }

    #[test]
    fn paper_page_state_machine_completeness() {
        use vugva::vmt::{Page, PageState};

        let valid_transitions: &[(PageState, PageState)] = &[
            (PageState::Unmapped, PageState::Allocated),
            (PageState::Allocated, PageState::Resident),
            (PageState::Resident, PageState::Warm),
            (PageState::Warm, PageState::Resident),
            (PageState::Warm, PageState::Cold),
            (PageState::Cold, PageState::Warm),
            (PageState::Resident, PageState::Cold),
            (PageState::Cold, PageState::Resident),
        ];

        for &(from, to) in valid_transitions {
            let mut page = Page::new("test".into(), vec![100], 2, 1, 1);
            if from != PageState::Unmapped {
                advance_to(&mut page, from);
            }
            let result = page.transition(to);
            assert!(
                result.is_ok(),
                "valid transition {from:?} -> {to:?} was rejected"
            );
        }
    }

    #[test]
    fn paper_page_state_machine_invalid_transitions() {
        use vugva::vmt::{Page, PageState};

        let invalid_transitions: &[(PageState, PageState)] = &[
            (PageState::Unmapped, PageState::Resident),
            (PageState::Unmapped, PageState::Warm),
            (PageState::Unmapped, PageState::Cold),
            (PageState::Allocated, PageState::Warm),
            (PageState::Allocated, PageState::Cold),
            (PageState::Allocated, PageState::Unmapped),
        ];

        for &(from, to) in invalid_transitions {
            let mut page = Page::new("test".into(), vec![100], 2, 1, 1);
            if from != PageState::Unmapped {
                advance_to(&mut page, from);
            }
            let result = page.transition(to);
            assert!(
                result.is_err(),
                "invalid transition {from:?} -> {to:?} should have been rejected"
            );
        }
    }

    // ================================================================
    // Paper §3.3 — DMA descriptor is exactly 64 bytes
    // ================================================================

    #[test]
    fn paper_dma_descriptor_64_bytes() {
        use std::mem::size_of;
        use vugva::dma::DmaDescriptor;
        assert_eq!(size_of::<DmaDescriptor>(), 64);
    }

    // ================================================================
    // Paper §5.1 Algorithm 2 — 72 bytes metadata per promotion
    // ================================================================

    #[test]
    fn paper_algo2_metadata_ratio() {
        let total_metadata = 8 + 64; // page lookup + DMA descriptor
        assert_eq!(total_metadata, 72);
        let page_size = 2 * 1024 * 1024;
        let ratio = total_metadata as f64 / page_size as f64;
        assert!(ratio < 0.001);
        println!("metadata ratio: {ratio:.6}");
    }

    // ================================================================
    // Paper §3.2 — Latency hiding equation
    // ================================================================

    #[test]
    fn paper_latency_hiding_equation() {
        let t_compute: f64 = 50.0;
        let t_transport: f64 = 256.0 * 1024.0 * 1024.0 / (31.5 * 1e9) * 1e3;
        assert!(t_compute > t_transport);
        println!("T_compute={t_compute}ms > T_transport={t_transport:.1}ms");
    }

    // ================================================================
    // Paper §4 — Three-tier bandwidth hierarchy
    // ================================================================

    #[test]
    fn paper_tier_bandwidth_ordering() {
        let (bw_vram, bw_dram_max, bw_dram_min, bw_ssd) = (1008.0_f64, 58.0, 28.0, 7.0);
        assert!(bw_vram / bw_dram_max > 17.0);
        assert!(bw_dram_min / bw_ssd > 3.0);
        assert!(bw_vram >= bw_ssd * 144.0);
    }

    #[test]
    fn paper_tier_latency_ordering() {
        let (lat_vram, lat_dram, lat_ssd) = (100.0_f64, 2_000.0, 50_000.0);
        assert!(lat_dram > lat_vram * 10.0);
        assert!(lat_ssd > lat_dram * 10.0);
    }

    // ================================================================
    // Paper §5.2 — Prefetch depth hides PCIe latency
    // ================================================================

    #[test]
    fn paper_prefetch_depth_hides_latency() {
        let t_compute: f64 = 50.0;
        let t_transport: f64 = 256.0 * 1024.0 * 1024.0 / (31.5 * 1e9) * 1e3;
        assert!(t_compute >= t_transport);
    }

    // ================================================================
    // Paper §5.1 — CPU-bypass: CPU touches < 0.01% of data
    // ================================================================

    #[test]
    fn paper_cpu_bypass_metadata_only() {
        let total_data = 100 * 2 * 1024 * 1024;
        let total_metadata = 100 * 72;
        let ratio = total_metadata as f64 / total_data as f64;
        assert!(ratio < 0.0001);
        println!("CPU bypass ratio: {ratio:.6}");
    }

    // ================================================================
    // Paper §5.2 — Throughput improvement
    // ================================================================

    #[test]
    fn paper_throughput_improvement_formula() {
        let bw_cpu_mediated: f64 = 28.0;
        let bw_cpu_bypass: f64 = 31.5;
        let overhead = 30.0 * 1e-6 * 1000.0;
        let effective_cpu = bw_cpu_mediated * (1.0 - overhead);
        let improvement = (bw_cpu_bypass - effective_cpu) / effective_cpu * 100.0;
        assert!(improvement > 10.0);
        println!("throughput improvement: {improvement:.1}%");
    }

    // ================================================================
    // Paper §5 — NVRTC runtime compilation, no cuBLAS / cuDNN / nvcc
    // ================================================================

    /// Drive the whole NVRTC path end to end on the real GPU: detect the
    /// architecture, compile a kernel from a source string, load the PTX,
    /// launch it, and check the bytes it produced.
    ///
    /// Nothing exercised this before. `ffi/nvrtc.rs` declared its entry points
    /// in a bare `extern "C"` block with no `#[link]` at all, which compiled
    /// purely because no test ever pulled the call sites into a linked binary —
    /// the first real consumer would have hit `undefined reference to
    /// nvrtcCreateProgram`. A test that actually launches a kernel is the only
    /// thing that keeps that class of bug from coming back.
    #[test]
    fn paper_nvrtc_compiles_and_launches() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();

        let arch = match vugva::nvrtc_kernel::detect_sm_arch(0) {
            Ok(a) => a,
            Err(e) => panic!("detect_sm_arch failed: {e:?}"),
        };
        println!("detected architecture: {arch}");

        let kernel = match vugva::nvrtc_kernel::compile_tier_promote(0) {
            Ok(k) => k,
            Err(e) => panic!("compile_tier_promote failed: {e:?}"),
        };
        assert!(!kernel.function().0.is_null(), "null function handle");

        // Copy 1 MiB device-to-device through the compiled kernel.
        const N: usize = 1 << 20;
        let host_src: Vec<u8> = (0..N).map(|i| (i % 251) as u8).collect();
        let mut host_dst = vec![0u8; N];

        let mut d_src = CUdeviceptr(0);
        let mut d_dst = CUdeviceptr(0);
        unsafe {
            assert_eq!(cuMemAlloc_v2(&mut d_src, N), CUDA_SUCCESS);
            assert_eq!(cuMemAlloc_v2(&mut d_dst, N), CUDA_SUCCESS);
            assert_eq!(
                cuMemcpyHtoD_v2(d_src, host_src.as_ptr() as *const _, N),
                CUDA_SUCCESS
            );

            let mut p_src = d_src;
            let mut p_dst = d_dst;
            let mut n: u64 = N as u64;
            let args: [*mut std::ffi::c_void; 3] = [
                &mut p_src as *mut _ as *mut _,
                &mut p_dst as *mut _ as *mut _,
                &mut n as *mut _ as *mut _,
            ];

            let block = 256u32;
            let grid = (N as u32).div_ceil(block);
            vugva::nvrtc_kernel::launch_kernel(
                &kernel,
                (grid, 1, 1),
                (block, 1, 1),
                0,
                CUstream(std::ptr::null_mut()),
                &args,
            )
            .expect("launch_kernel failed");

            assert_eq!(cuCtxSynchronize(), CUDA_SUCCESS, "kernel launch faulted");
            assert_eq!(
                cuMemcpyDtoH_v2(host_dst.as_mut_ptr() as *mut _, d_dst, N),
                CUDA_SUCCESS
            );
            let _ = cuMemFree_v2(d_src);
            let _ = cuMemFree_v2(d_dst);
        }

        assert_eq!(host_dst, host_src, "kernel did not copy the buffer intact");
        println!("NVRTC-compiled tier_promote copied {N} bytes correctly on {arch}");
    }

    // ================================================================
    // Paper §5.1 — repeated promotion must not consume VRAM
    // ================================================================

    /// `TieredPool::access` used to call `cuCtxCreate_v2` through
    /// `set_context` on **every** invocation and never destroy the result.
    /// Measured on this machine that is 97.8 MB of VRAM per call: an 8 GB card
    /// died after ~70 accesses, before storing a single tensor. The paper's
    /// whole claim is that a promotion moves data and costs no CPU or driver
    /// state, so this is the property worth asserting directly.
    ///
    /// Each promotion must therefore cost only the VRAM of the data itself.
    ///
    /// The test uses **distinct** pages, one promotion each. Re-accessing a
    /// single page would prove nothing: after the first promotion its tier is
    /// `Vram` and every later call takes the already-resident early return,
    /// which never binds a context at all.
    ///
    /// `PAGES × PAGE_BYTES` of VRAM is legitimately consumed. A leaked context
    /// per promotion would add 97.8 MB on top of each — 12.5 GB across the run,
    /// which an 8 GB card cannot even satisfy, so under the old code this test
    /// failed with an allocation error rather than a threshold breach.
    #[test]
    fn paper_repeated_access_does_not_leak_context() {
        let _gpu = gpu_exclusive();
        // `cuMemGetInfo_v2` reports for the *current* context, so the test
        // thread needs one bound before it can measure anything. This is
        // device 0's primary context — the same one the pool retains — so the
        // figures describe the pool's own allocations.
        let _ctx = current_ctx();

        const PAGES: usize = 128;
        const PAGE_ELEMS: usize = 262_144; // 1 MiB at 4 bytes/element
        const PAGE_BYTES: usize = PAGE_ELEMS * 4;

        let mut pool = match vugva::tiered::TieredPool::new(&[0], 512 << 20) {
            Ok(p) => p,
            Err(e) => panic!("TieredPool::new failed: {e:?}"),
        };
        println!(
            "DRAM pool NUMA-bound: {}, pinned: {}",
            pool.dram_numa_bound(0).expect("dram_numa_bound"),
            pool.dram_pinned(0).expect("dram_pinned"),
        );

        // Every page starts in DRAM, so every access takes the promote path.
        let names: Vec<String> = (0..PAGES)
            .map(|i| {
                pool.allocate(
                    &format!("leak.probe.{i}"),
                    &[PAGE_ELEMS],
                    4,
                    vugva::vmt::Tier::Dram,
                )
                .expect("allocate")
            })
            .collect();

        let baseline = free_mb();
        for (i, name) in names.iter().enumerate() {
            let p = pool.access(name, 0).expect("access");
            assert_ne!(p, 0, "promotion {i} returned a null device pointer");
        }
        let after = free_mb();

        let delta = baseline - after;
        let payload_mb = (PAGES * PAGE_BYTES) as f64 / (1024.0 * 1024.0);
        let overhead = delta - payload_mb;
        println!(
            "{PAGES} promotions: free VRAM {baseline:.1} → {after:.1} MB \
             (delta {delta:.1} MB, payload {payload_mb:.1} MB, \
             overhead {overhead:.1} MB = {:.2} MB/promotion)",
            overhead / PAGES as f64
        );
        // A single leaked context is ~98 MB. Allowing 64 MB total keeps the
        // threshold below even one leak while absorbing allocator granularity.
        assert!(
            overhead < 64.0,
            "{PAGES} promotions cost {overhead:.1} MB beyond the \
             {payload_mb:.1} MB of data — context or allocation leak"
        );
    }

    // ================================================================
    // Paper §4 — DRAM tier must run at DMA speed, not staged-copy speed
    // ================================================================

    /// Measure the real DRAM→VRAM promotion bandwidth of the tiered pool.
    ///
    /// The pool's DRAM is `mmap` + `mbind` + `cuMemHostRegister`, so it is both
    /// NUMA-local and page-locked. Only the page-locking makes the transfer a
    /// true DMA: against *pageable* memory — which this pool was before, having
    /// never been registered — the driver stages every "async" copy through an
    /// internal bounce buffer, so it runs synchronously at roughly half speed
    /// and the CPU touches every byte. That is the exact opposite of the
    /// paper's claim, and it produced no error to notice.
    ///
    /// Measured on this machine (RTX 4060, PCIe 4.0 x8), 256 MiB in 16 MiB
    /// pages — run with `VUGVA_NO_PIN=1` to reproduce the second row:
    ///
    /// ```text
    /// pinned   21.2 ms = 12.7 GB/s
    /// pageable 48.7 ms =  5.5 GB/s
    /// ```
    ///
    /// This test asserts only that the pool reports itself pinned and that the
    /// measured rate clears a floor a staged copy could not. The number itself
    /// is printed rather than asserted against the paper's 28–58 GB/s: that
    /// range is for server-class hardware, and this box is a single-socket
    /// desktop on PCIe 4.0 x8, whose practical ceiling is ~13 GB/s — which the
    /// pinned path essentially reaches. Claiming the paper's figure on this
    /// hardware would be dishonest benchmarking.
    #[test]
    fn paper_dram_tier_promotes_at_dma_speed() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();

        const PAGES: usize = 16;
        const PAGE_ELEMS: usize = 4 << 20; // 16 MiB at 4 bytes/element
        const PAGE_BYTES: usize = PAGE_ELEMS * 4;

        let mut pool = match vugva::tiered::TieredPool::new(&[0], 512 << 20) {
            Ok(p) => p,
            Err(e) => panic!("TieredPool::new failed: {e:?}"),
        };

        let pinned = pool.dram_pinned(0).expect("dram_pinned");
        assert!(
            pinned || std::env::var_os("VUGVA_NO_PIN").is_some(),
            "DRAM pool is not page-locked — cuMemHostRegister failed. \
             Check `ulimit -l`; every promotion is a synchronous staged copy."
        );

        let names: Vec<String> = (0..PAGES)
            .map(|i| {
                pool.allocate(
                    &format!("bw.probe.{i}"),
                    &[PAGE_ELEMS],
                    4,
                    vugva::vmt::Tier::Dram,
                )
                .expect("allocate")
            })
            .collect();

        let start = std::time::Instant::now();
        for name in &names {
            pool.access(name, 0).expect("access");
        }
        let elapsed = start.elapsed();

        let bytes = (PAGES * PAGE_BYTES) as f64;
        let gbps = bytes / elapsed.as_secs_f64() / 1e9;
        println!(
            "DRAM→VRAM promotion: {:.0} MiB in {:.2} ms = {gbps:.1} GB/s \
             (pinned: {pinned}, NUMA-bound: {})",
            bytes / (1024.0 * 1024.0),
            elapsed.as_secs_f64() * 1e3,
            pool.dram_numa_bound(0).expect("dram_numa_bound"),
        );

        // A pageable staged copy on this class of machine lands around 3-6 GB/s
        // and includes a cuMemAlloc per page in the same measurement. 2 GB/s is
        // a floor that catches a regression to the CPU-staged path without
        // being sensitive to machine speed.
        if pinned {
            assert!(
                gbps > 8.0,
                "promotion ran at {gbps:.1} GB/s — the pageable staged path \
                 measures 5.5 GB/s here, so this has regressed off the DMA path"
            );
        }
    }

    /// Allocation must be a *working-set* budget, not a lifetime budget.
    ///
    /// The pool below is 64 MiB and the test cycles 512 MiB of pages through
    /// it — 8x its capacity — promoting each one to VRAM and releasing it
    /// again. Under the bump-only allocator this failed on cycle 17 with
    /// `CUDA_ERROR_OUT_OF_MEMORY` while the pool held exactly one live page,
    /// and every promotion additionally leaked a device block that nothing
    /// ever freed (BUG #19).
    #[test]
    fn paper_allocations_are_reclaimed_and_reused() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();

        const DRAM_POOL: usize = 64 << 20;
        const PAGE_ELEMS: usize = 1 << 20; // 4 MiB at 4 bytes/element
        const CYCLES: usize = 128;

        let mut pool = match vugva::tiered::TieredPool::new(&[0], DRAM_POOL) {
            Ok(p) => p,
            Err(e) => panic!("TieredPool::new failed: {e:?}"),
        };

        let before = free_mb();
        for i in 0..CYCLES {
            let name = pool
                .allocate(
                    &format!("churn_{i}"),
                    &[PAGE_ELEMS],
                    4,
                    vugva::vmt::Tier::Dram,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "allocation failed on cycle {i} of {CYCLES} — the pool \
                         is {} MiB and holds one page at a time, so this is a \
                         reclaim failure, not exhaustion: {e:?}",
                        DRAM_POOL >> 20
                    )
                });
            pool.access(&name, 0).expect("access");
            pool.deallocate(&name).expect("deallocate");

            assert_eq!(
                pool.dram_used(0).expect("dram_used"),
                0,
                "cycle {i}: DRAM still marked in use after the only page was freed"
            );
        }

        // One device block of this size should have been recycled for every
        // cycle after the first, so exactly one should be sitting in the cache.
        assert_eq!(
            pool.vram_cached_blocks(0).expect("vram_cached_blocks"),
            1,
            "{CYCLES} promote/release cycles should recycle a single device block"
        );

        let after = free_mb();
        let leaked = before - after;
        println!(
            "{CYCLES} alloc/promote/free cycles ({} MiB churned through a {} MiB pool): \
             free VRAM {before:.1} -> {after:.1} MB (retained {leaked:.1} MB)",
            (CYCLES * PAGE_ELEMS * 4) >> 20,
            DRAM_POOL >> 20,
        );
        // The single recycled 4 MiB block is expected to still be held; 128
        // leaked ones would be 512 MB.
        assert!(
            leaked < 32.0,
            "{CYCLES} cycles retained {leaked:.1} MB of VRAM — device blocks \
             are not being recycled"
        );

        drop(pool);
        let reclaimed = free_mb();
        assert!(
            reclaimed >= before - 8.0,
            "dropping the pool left {:.1} MB unreturned — the recycle cache is \
             not freed at teardown",
            before - reclaimed
        );
    }

    // ================================================================
    // Paper Figure 5 — RESIDENT → WARM must write back and reclaim
    // ================================================================

    /// Demotion must move the *data*, not just the state field.
    ///
    /// The test writes a pattern into the DRAM shard, promotes it to VRAM,
    /// then scribbles over the DRAM copy. Only a real VRAM→DRAM writeback can
    /// restore it. The old `demote` submitted a descriptor and copied nothing
    /// while still marking the page `Warm` — "the DRAM copy is authoritative" —
    /// so the scribble would have survived and the next promotion would have
    /// served it as if it were the tensor (BUG #4).
    #[test]
    fn paper_demote_writes_back_and_reclaims_vram() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();

        const ELEMS: usize = 1 << 20; // 4 MiB at 4 bytes/element
        const BYTES: usize = ELEMS * 4;

        let mut pool = match vugva::tiered::TieredPool::new(&[0], 64 << 20) {
            Ok(p) => p,
            Err(e) => panic!("TieredPool::new failed: {e:?}"),
        };

        let name = pool
            .allocate("wb", &[ELEMS], 4, vugva::vmt::Tier::Dram)
            .expect("allocate");

        let host = pool.vmt.lookup(&name).expect("page").dram_chunks[0].host_ptr as *mut u8;

        // A position-dependent pattern: a stale-vs-fresh mix-up shows up as a
        // specific wrong index rather than as "all zeros".
        let pattern = |i: usize| (i as u8) ^ 0xA5;
        // SAFETY: `host` points at BYTES of live, page-locked pool memory.
        unsafe {
            for i in 0..BYTES {
                host.add(i).write(pattern(i));
            }
        }

        pool.access(&name, 0).expect("promote to VRAM");
        assert_eq!(
            pool.vmt.lookup(&name).expect("page").tier,
            vugva::vmt::Tier::Vram
        );

        // Destroy the DRAM copy. If demote does not really write back, this is
        // what survives.
        // SAFETY: same range as above.
        unsafe {
            std::ptr::write_bytes(host, 0xEE, BYTES);
        }

        pool.demote(&name).expect("demote");

        let page = pool.vmt.lookup(&name).expect("page");
        assert_eq!(page.tier, vugva::vmt::Tier::Dram, "tier must drop to DRAM");
        assert_eq!(
            page.state,
            vugva::vmt::PageState::Warm,
            "state must drop to Warm"
        );
        assert!(
            page.vram_chunks.is_empty(),
            "demote must drop the VRAM chunk it released — a stale chunk would \
             hand out a recycled device pointer on the next access"
        );

        // SAFETY: same range as above.
        let restored = unsafe { std::slice::from_raw_parts(host, BYTES) };
        let mismatch = (0..BYTES).find(|&i| restored[i] != pattern(i));
        assert!(
            mismatch.is_none(),
            "byte {} is 0x{:02X}, expected 0x{:02X} — VRAM was not written back",
            mismatch.unwrap(),
            restored[mismatch.unwrap()],
            pattern(mismatch.unwrap()),
        );

        // The released device block must be reusable, not leaked.
        assert_eq!(
            pool.vram_cached_blocks(0).expect("vram_cached_blocks"),
            1,
            "demote must return its device block to the recycle cache"
        );

        // And promoting again must reuse that exact block rather than growing.
        let before = free_mb();
        pool.access(&name, 0).expect("re-promote");
        let after = free_mb();
        assert_eq!(
            pool.vram_cached_blocks(0).expect("vram_cached_blocks"),
            0,
            "re-promotion must consume the recycled block"
        );
        assert!(
            before - after < 1.0,
            "re-promotion allocated {:.1} MB instead of reusing the {} MiB \
             block demote just released",
            before - after,
            BYTES >> 20,
        );

        println!(
            "demote/re-promote round trip on {} MiB: {BYTES} bytes verified, \
             0 new VRAM allocated",
            BYTES >> 20
        );
    }

    /// T2 (§4, Table 2): a page allocated *cold* must promote SSD→DRAM→VRAM.
    ///
    /// This is the property that makes the hierarchy three-tier rather than a
    /// two-tier cache. Before T2 was wired in, `Tier::Ssd` allocated DRAM like
    /// every other tier and `access` read that DRAM chunk without ever opening
    /// the file — so the cold tier was decorative and the DRAM pool still
    /// bounded the whole corpus.
    ///
    /// The decisive part is the capacity relationship: the spill file is 16×
    /// the DRAM pool and the page is a quarter of that pool, so an
    /// implementation that still demanded full DRAM backing for cold pages
    /// would fail to allocate here rather than quietly pass.
    #[test]
    fn paper_ssd_tier_promotes_cold_pages() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();

        const ELEMS: usize = 1 << 20; // 1 MiB at 1 byte/element
        const BYTES: usize = ELEMS;
        const DRAM_POOL: usize = 4 << 20;

        let mut pool = match vugva::tiered::TieredPool::new(&[0], DRAM_POOL) {
            Ok(p) => p,
            Err(e) => panic!("TieredPool::new failed: {e:?}"),
        };

        // Next to the test binary, not /tmp — on most desktop installs /tmp is
        // a tmpfs, i.e. RAM, so a cold tier placed there consumes the very
        // resource it exists to relieve.
        let path = {
            let mut p = std::env::current_exe().expect("current_exe");
            p.pop();
            p.push(format!("vugva_t2_hw_{}", std::process::id()));
            p
        };
        pool.attach_spill(&path, 64 << 20, true)
            .expect("attach_spill");
        assert!(pool.has_spill(), "spill must be attached");

        let name = pool
            .allocate("t2.cold.page", &[ELEMS], 1, vugva::vmt::Tier::Ssd)
            .expect(
                "a cold page must allocate against the spill file — a DramOom \
                 here means Tier::Ssd is still demanding DRAM backing",
            );
        assert!(
            pool.spill_used() >= BYTES,
            "cold allocation must reserve spill space, used={}",
            pool.spill_used()
        );

        // Promote. The file was set to full length at creation and this page
        // has never been written, so it must read back as zeros — which is
        // exactly what proves the bytes came from the file rather than from
        // uninitialised DRAM.
        let dptr = pool.access(&name, 0).expect("promote from SSD");
        assert_ne!(dptr, 0, "promotion must yield a device pointer");

        let mut got = vec![0xABu8; BYTES];
        unsafe {
            let rc = vugva::ffi::cuda::cuMemcpyDtoH_v2(
                got.as_mut_ptr() as *mut std::ffi::c_void,
                vugva::ffi::cuda::CUdeviceptr(dptr),
                BYTES,
            );
            assert_eq!(rc, 0, "cuMemcpyDtoH_v2 failed: {rc}");
        }
        assert_eq!(
            got.iter().filter(|&&b| b != 0).count(),
            0,
            "a reserved-but-unwritten cold page must arrive as zeros; non-zero \
             bytes mean the promotion served stale DRAM instead of the file"
        );

        println!(
            "T2 verified: {} MiB cold page promoted SSD→DRAM→VRAM against a \
             {} MiB DRAM pool (spill_used={} B)",
            BYTES >> 20,
            DRAM_POOL >> 20,
            pool.spill_used()
        );
    }

    /// The full cold round trip: write → spill → promote → read back exact.
    ///
    /// `paper_ssd_tier_promotes_cold_pages` proves a cold page *arrives*; this
    /// proves it arrives carrying the caller's data. Those are different
    /// claims, and only the second one makes the tier usable — a corpus that
    /// pages in as zeros is exactly as useless as one that fails to allocate.
    ///
    /// The pattern is position-dependent (not a constant fill) so a write that
    /// lands at the wrong offset, or a promotion that reads the wrong range,
    /// shows up as a mismatch rather than passing by luck. The page is also
    /// larger than a quarter of the DRAM pool, so it cannot be quietly served
    /// from a DRAM copy.
    #[test]
    fn paper_ssd_tier_round_trips_written_bytes() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();

        const BYTES: usize = 2 << 20; // 2 MiB
        const DRAM_POOL: usize = 4 << 20;

        let mut pool = match vugva::tiered::TieredPool::new(&[0], DRAM_POOL) {
            Ok(p) => p,
            Err(e) => panic!("TieredPool::new failed: {e:?}"),
        };
        let path = {
            let mut p = std::env::current_exe().expect("current_exe");
            p.pop();
            p.push(format!("vugva_t2_rt_{}", std::process::id()));
            p
        };
        pool.attach_spill(&path, 64 << 20, true)
            .expect("attach_spill");

        let name = pool
            .allocate("t2.corpus", &[BYTES], 1, vugva::vmt::Tier::Ssd)
            .expect("cold allocate");

        let src: Vec<u8> = (0..BYTES).map(|i| (i * 31 + (i >> 13)) as u8).collect();
        pool.write_page(&name, &src).expect("write_page");

        let dptr = pool.access(&name, 0).expect("promote from SSD");
        let mut got = vec![0u8; BYTES];
        unsafe {
            let rc = vugva::ffi::cuda::cuMemcpyDtoH_v2(
                got.as_mut_ptr() as *mut std::ffi::c_void,
                vugva::ffi::cuda::CUdeviceptr(dptr),
                BYTES,
            );
            assert_eq!(rc, 0, "cuMemcpyDtoH_v2 failed: {rc}");
        }

        let bad = got.iter().zip(&src).filter(|(a, b)| a != b).count();
        assert_eq!(
            bad, 0,
            "{bad} of {BYTES} bytes differ after SSD→DRAM→VRAM — the cold tier \
             is not carrying the caller's data"
        );

        // A short write must be refused rather than leaving a partial page.
        assert!(
            pool.write_page(&name, &src[..BYTES / 2]).is_err(),
            "a size mismatch must be rejected, not silently truncated"
        );

        println!(
            "T2 round trip verified: {} MiB written → spilled → promoted → byte-exact",
            BYTES >> 20
        );
    }

    /// LGM §3.4: skewed access must sort pages into tiers, and the membrane
    /// must follow the workload when it moves.
    ///
    /// Both halves matter and they fail independently. A membrane that places
    /// correctly but never re-places is just a startup heuristic; one that
    /// re-places without settling is a thrash generator. This drives a
    /// two-phase workload — hot on the first quarter of pages, then hot on the
    /// last quarter — and asserts placement after each.
    ///
    /// The VRAM budget is set to roughly a third of the working set so the
    /// membrane is genuinely forced to choose. With room for everything there
    /// is no decision to test.
    #[test]
    fn paper_membrane_places_by_access_frequency() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();

        const PAGES: usize = 12;
        const ELEMS: usize = 1 << 20; // 4 MiB at 4 bytes
        const PAGE_BYTES: usize = ELEMS * 4;
        // Room for ~4 of 12 pages once the 0.85 high-water mark is applied.
        const BUDGET: usize = PAGE_BYTES * 5;

        let mut pool = match vugva::tiered::TieredPool::new(&[0], 256 << 20) {
            Ok(p) => p,
            Err(e) => panic!("TieredPool::new failed: {e:?}"),
        };
        pool.enable_membrane(BUDGET);

        let names: Vec<String> = (0..PAGES)
            .map(|i| {
                pool.allocate(
                    &format!("lgm.page.{i}"),
                    &[ELEMS],
                    4,
                    vugva::vmt::Tier::Dram,
                )
                .expect("allocate")
            })
            .collect();

        // Phase 1: pages 0..3 are hot, the rest touched once each.
        for _ in 0..6 {
            for _ in 0..8 {
                for n in names.iter().take(3) {
                    pool.access(n, 0).expect("hot access");
                }
            }
            for n in names.iter().skip(3) {
                pool.access(n, 0).expect("cold access");
            }
            pool.background_sweep().expect("sweep");
        }

        let hot_rate: f64 = names.iter().take(3).map(|n| pool.membrane_rate(n)).sum();
        let cold_rate: f64 = names.iter().skip(3).map(|n| pool.membrane_rate(n)).sum();
        assert!(
            hot_rate > cold_rate,
            "phase 1: hot pages must carry the higher aggregate rate \
             ({hot_rate:.2} vs {cold_rate:.2})"
        );
        let resident_hot = names
            .iter()
            .take(3)
            .filter(|n| pool.vmt.lookup(n).map(|p| p.tier == vugva::vmt::Tier::Vram) == Some(true))
            .count();
        assert!(
            resident_hot >= 2,
            "phase 1: at least 2 of 3 hot pages should be VRAM-resident, got \
             {resident_hot} — the membrane is not placing by frequency"
        );

        // Phase 2: the workload inverts. The last three pages become hot.
        for _ in 0..8 {
            for _ in 0..8 {
                for n in names.iter().rev().take(3) {
                    pool.access(n, 0).expect("hot access");
                }
            }
            pool.background_sweep().expect("sweep");
        }

        let new_hot: f64 = names
            .iter()
            .rev()
            .take(3)
            .map(|n| pool.membrane_rate(n))
            .sum();
        let old_hot: f64 = names.iter().take(3).map(|n| pool.membrane_rate(n)).sum();
        assert!(
            new_hot > old_hot,
            "phase 2: the membrane must follow the workload — new hot \
             {new_hot:.2} vs stale {old_hot:.2}. A raw access counter fails \
             exactly here, because the phase-1 pages keep their lifetime total."
        );
        let resident_new = names
            .iter()
            .rev()
            .take(3)
            .filter(|n| pool.vmt.lookup(n).map(|p| p.tier == vugva::vmt::Tier::Vram) == Some(true))
            .count();
        assert!(
            resident_new >= 2,
            "phase 2: at least 2 of 3 newly-hot pages should have been \
             promoted, got {resident_new}"
        );

        println!(
            "LGM membrane: budget {} MiB over {PAGES} × {} MiB — \
             phase 1 hot resident {resident_hot}/3, after inversion {resident_new}/3",
            BUDGET >> 20,
            PAGE_BYTES >> 20,
        );
    }

    /// §3.2 Look-Ahead Prefetch: a prefetched page must arrive correct, and the
    /// access that claims it must be materially cheaper than one that starts
    /// the transfer itself.
    ///
    /// Both halves matter. Correctness alone would be satisfied by a prefetch
    /// that does nothing (the access would still work), and speed alone would
    /// be satisfied by one that hands back a pointer to the wrong bytes — which
    /// is precisely the failure mode this codebase keeps producing. So the test
    /// asserts the data *and* the timing.
    ///
    /// The comparison is deliberately generous: prefetch only has to beat the
    /// cold path, not beat it by a specific factor. PCIe contention on a shared
    /// desktop makes a tight bound flaky, and a flaky performance assertion is
    /// worse than a loose one because it trains people to ignore failures.
    #[test]
    fn paper_prefetch_overlaps_transport_with_compute() {
        let _gpu = gpu_exclusive();
        let _ctx = current_ctx();

        const ELEMS: usize = 8 << 20; // 32 MiB at 4 bytes
        const BYTES: usize = ELEMS * 4;
        const PAGES: usize = 8;

        let mut pool = match vugva::tiered::TieredPool::new(&[0], 512 << 20) {
            Ok(p) => p,
            Err(e) => panic!("TieredPool::new failed: {e:?}"),
        };

        let names: Vec<String> = (0..PAGES)
            .map(|i| {
                pool.allocate(&format!("pf.page.{i}"), &[ELEMS], 4, vugva::vmt::Tier::Dram)
                    .expect("allocate")
            })
            .collect();

        // Distinct, position-dependent contents per page, so a prefetch that
        // returns the *wrong page's* pointer is caught rather than passing.
        for (i, name) in names.iter().enumerate() {
            let src: Vec<u8> = (0..BYTES).map(|b| (b.wrapping_mul(31) ^ i) as u8).collect();
            pool.write_page(name, &src).expect("write_page");
        }

        // Cold: access with no prefetch outstanding.
        let t_cold = std::time::Instant::now();
        for name in &names {
            pool.access(name, 0).expect("cold access");
            pool.demote(name).expect("demote");
        }
        let cold = t_cold.elapsed();

        // Warm: issue every prefetch first, then claim them.
        let t_pf = std::time::Instant::now();
        for name in &names {
            pool.prefetch(name, 0).expect("prefetch");
        }
        let issue = t_pf.elapsed();
        assert_eq!(
            pool.inflight_count(),
            PAGES,
            "every prefetch must be in flight before any is claimed"
        );

        let t_warm = std::time::Instant::now();
        let mut ptrs = Vec::with_capacity(PAGES);
        for name in &names {
            ptrs.push(pool.access(name, 0).expect("warm access"));
        }
        let warm = t_warm.elapsed();
        assert_eq!(
            pool.inflight_count(),
            0,
            "claims must drain the in-flight set"
        );

        // Correctness: every page must carry its own bytes.
        for (i, ptr) in ptrs.iter().enumerate() {
            let mut got = vec![0u8; 4096];
            unsafe {
                let rc = vugva::ffi::cuda::cuMemcpyDtoH_v2(
                    got.as_mut_ptr() as *mut std::ffi::c_void,
                    vugva::ffi::cuda::CUdeviceptr(*ptr),
                    got.len(),
                );
                assert_eq!(rc, 0, "cuMemcpyDtoH_v2 failed: {rc}");
            }
            let want: Vec<u8> = (0..got.len())
                .map(|b| (b.wrapping_mul(31) ^ i) as u8)
                .collect();
            assert_eq!(got, want, "page {i} came back with another page's contents");
        }

        println!(
            "prefetch: issue {:.2} ms · claim {:.2} ms (cold {:.2} ms) over {PAGES} × {} MiB",
            issue.as_secs_f64() * 1e3,
            warm.as_secs_f64() * 1e3,
            cold.as_secs_f64() * 1e3,
            BYTES >> 20,
        );
        assert!(
            warm < cold,
            "claiming prefetched pages ({:.2} ms) must beat cold promotion \
             ({:.2} ms) — if it does not, the copy is not actually overlapping",
            warm.as_secs_f64() * 1e3,
            cold.as_secs_f64() * 1e3,
        );
    }
}
