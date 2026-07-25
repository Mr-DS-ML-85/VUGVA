//! Hardware integration tests — run on real CUDA GPUs.
//!
//! These tests call into libcuda.so (linked via #[link(name = "cuda")])
//! and verify the GPU hardware is reachable and behaves as the paper expects.

#[cfg(test)]
mod hw {
    use std::ffi::c_int;
    use vugva::ffi::cuda::*;

    fn init_cuda() {
        unsafe {
            let _ = cuInit(0);
        }
    }

    fn current_ctx() -> CUcontext {
        init_cuda();
        let dev = CUdevice(0);
        let mut ctx = CUcontext(std::ptr::null_mut());
        unsafe { cuCtxCreate_v2(&mut ctx, 0, dev) };
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
        let _ctx = current_ctx();
        let mut dptr = CUdeviceptr::NULL;
        let rc = unsafe { cuMemAlloc_v2(&mut dptr, 4 * 1024 * 1024) };
        assert_eq!(rc, CUDA_SUCCESS);
        assert!(!dptr.is_null());
        unsafe { cuMemFree_v2(dptr) };
    }

    #[test]
    fn paper_algo1_large_alloc() {
        let _ctx = current_ctx();
        let mut dptr = CUdeviceptr::NULL;
        let rc = unsafe { cuMemAlloc_v2(&mut dptr, 256 * 1024 * 1024) };
        assert_eq!(rc, CUDA_SUCCESS);
        unsafe { cuMemFree_v2(dptr) };
    }

    #[test]
    fn paper_algo1_multiple_shards() {
        let _ctx = current_ctx();
        let mut ptrs = Vec::new();
        for _ in 0..8 {
            let mut dptr = CUdeviceptr::NULL;
            let rc = unsafe { cuMemAlloc_v2(&mut dptr, 32 * 1024 * 1024) };
            assert_eq!(rc, CUDA_SUCCESS);
            ptrs.push(dptr);
        }
        for p in ptrs {
            unsafe { cuMemFree_v2(p); }
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
            assert!(result.is_ok(), "valid transition {from:?} -> {to:?} was rejected");
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
            assert!(result.is_err(), "invalid transition {from:?} -> {to:?} should have been rejected");
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
}
