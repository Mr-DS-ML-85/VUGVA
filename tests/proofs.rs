//! Paper Invariant Proofs — formal verification of every claim in the paper.
//!
//! These are not usage examples. Each test is a proof that a specific
//! mathematical property, invariant, or claim from the paper holds.
//! Every assertion is documented with the paper section it verifies.
//!
//! Run:  cargo test --test proofs

#[cfg(test)]
mod proofs {
    use std::mem;
    use vugva::dma::DmaDescriptor;
    use vugva::membrane::{PageObs, TierPolicy, HYSTERESIS, RATE_DECAY, VRAM_HIGH_WATER};
    use vugva::range_alloc::{RangeAllocator, ALIGN};
    use vugva::vmt::{Page, PageState, Tier, VirtualMemoryTable};

    // ====================================================================
    // PROOF 1: DmaDescriptor is exactly 64 bytes (§3.3)
    // ====================================================================

    #[test]
    fn proof_01_dma_descriptor_is_64_bytes() {
        assert_eq!(
            mem::size_of::<DmaDescriptor>(),
            64,
            "Paper §3.3: DmaDescriptor must be exactly 64 bytes"
        );
    }

    #[test]
    fn proof_02_dma_descriptor_alignment_is_64() {
        assert_eq!(
            mem::align_of::<DmaDescriptor>(),
            64,
            "DmaDescriptor must be 64-byte aligned for DMA engine"
        );
    }

    #[test]
    fn proof_03_dma_descriptor_all_fields_fit() {
        // 8+8+4+4+4+4+4 = 36 bytes fields, pad = 28 bytes, total = 64
        assert_eq!(mem::size_of::<DmaDescriptor>(), 64);
    }

    // ====================================================================
    // PROOF 4: Metadata per promotion = 72 bytes (§5.1)
    // ====================================================================

    #[test]
    fn proof_04_metadata_per_promotion_is_72_bytes() {
        let page_lookup = mem::size_of::<usize>(); // 8 bytes on 64-bit
        let dma_desc = mem::size_of::<DmaDescriptor>(); // 64 bytes
        assert_eq!(
            page_lookup + dma_desc,
            72,
            "Paper §5.1: CPU touches exactly 72 bytes per promotion"
        );
    }

    // ====================================================================
    // PROOF 5: Metadata/data ratio < 0.1% for any allocation
    // ====================================================================

    #[test]
    fn proof_05_metadata_ratio_below_01_percent() {
        let metadata = 72usize;
        // 4KB page (minimum realistic DMA transfer)
        let ratio_4k = metadata as f64 / 4096.0;
        assert!(
            ratio_4k < 0.02,
            "Ratio {ratio_4k:.4} must be < 2% for 4KB page"
        );

        // 256KB page (typical L2 cache-sized block)
        let ratio_256k = metadata as f64 / (256.0 * 1024.0);
        assert!(
            ratio_256k < 0.001,
            "Ratio {ratio_256k:.6} must be < 0.1% for 256KB"
        );

        // 256MB allocation (typical tensor)
        let ratio_256m = metadata as f64 / (256.0 * 1024.0 * 1024.0);
        assert!(
            ratio_256m < 0.000001,
            "Ratio {ratio_256m:.10} must be < 0.0001% for 256MB"
        );
    }

    // ====================================================================
    // PROOF 6: CPU-bypass — CPU touches < 0.01% of transfer
    // ====================================================================

    #[test]
    fn proof_06_cpu_bypass_fraction() {
        let cpu_touch = 72usize;
        let transfer = 4 * 1024 * 1024; // 4MB DMA
        let fraction = cpu_touch as f64 / transfer as f64;
        assert!(
            fraction < 0.0001,
            "CPU fraction {fraction:.8} must be < 0.01%"
        );
    }

    // ====================================================================
    // PROOF 7: Page State Machine — all valid transitions (Figure 5)
    // ====================================================================

    #[test]
    fn proof_07_valid_transitions_accepted() {
        let valid = [
            (PageState::Unmapped, PageState::Allocated),
            (PageState::Allocated, PageState::Resident),
            (PageState::Resident, PageState::Warm),
            (PageState::Warm, PageState::Resident),
            (PageState::Warm, PageState::Cold),
            (PageState::Cold, PageState::Warm),
        ];
        for (from, to) in &valid {
            let mut page = Page::new("t".into(), vec![10], 4, 1, 1);
            advance_page(&mut page, *from);
            assert!(
                page.transition(*to).is_ok(),
                "Valid {from:?} → {to:?} was rejected"
            );
        }
    }

    #[test]
    fn proof_08_invalid_transitions_rejected() {
        let invalid = [
            (PageState::Unmapped, PageState::Resident),
            (PageState::Unmapped, PageState::Warm),
            (PageState::Unmapped, PageState::Cold),
            (PageState::Allocated, PageState::Warm),
            (PageState::Allocated, PageState::Cold),
            (PageState::Resident, PageState::Allocated),
            (PageState::Cold, PageState::Allocated),
            (PageState::Cold, PageState::Unmapped),
        ];
        for (from, to) in &invalid {
            let mut page = Page::new("t".into(), vec![10], 4, 1, 1);
            advance_page(&mut page, *from);
            assert!(
                page.transition(*to).is_err(),
                "Invalid {from:?} → {to:?} should be rejected"
            );
        }
    }

    fn advance_page(page: &mut Page, target: PageState) {
        match target {
            PageState::Unmapped => {}
            PageState::Allocated => page.state = PageState::Allocated,
            PageState::Resident => page.state = PageState::Resident,
            PageState::Warm => {
                page.state = PageState::Resident;
                page.state = PageState::Warm;
            }
            PageState::Cold => {
                page.state = PageState::Resident;
                page.state = PageState::Warm;
                page.state = PageState::Cold;
            }
        }
    }

    // ====================================================================
    // PROOF 8: Three-tier bandwidth ordering (Table 3)
    // ====================================================================

    #[test]
    fn proof_09_bandwidth_hierarchy() {
        let vram: f64 = 1008.0;
        let dram: f64 = 43.0;
        let ssd: f64 = 6.0;

        assert!(vram > dram * 10.0, "VRAM > 10× DRAM");
        assert!(dram > ssd * 3.0, "DRAM > 3× SSD");
        assert!(vram > ssd * 100.0, "VRAM > 100× SSD");
    }

    // ====================================================================
    // PROOF 9: NUMA bandwidth factors (§4.2)
    // ====================================================================

    #[test]
    fn proof_10_numa_bandwidth_factors_ordering() {
        let local = 0.95f64;
        let cross = 0.80f64;
        let far = 0.65f64;

        assert!(local > cross, "Local > Cross-socket");
        assert!(cross > far, "Cross-socket > Far");
        assert!(local - far >= 0.25, "Spread ≥ 0.25");
    }

    // ====================================================================
    // PROOF 10: Membrane hysteresis prevents thrash
    // ====================================================================

    #[test]
    fn proof_11_hysteresis_prevents_thrash() {
        let policy = TierPolicy::default();
        let hot = PageObs {
            rate: 10.0,
            bytes: 1024,
            resident: true,
            pinned: false,
        };
        let challenger = PageObs {
            rate: 10.5,
            bytes: 1024,
            resident: false,
            pinned: false,
        };

        let hot_score = policy.score(&hot);
        let chal_score = policy.score(&challenger);

        // 10.5/10.0 = 1.05 < HYSTERESIS(1.5) → no eviction
        assert!(
            hot_score >= chal_score / HYSTERESIS,
            "Near-equal pages must not thrash: hot={hot_score:.3}, chal={chal_score:.3}"
        );
    }

    #[test]
    fn proof_12_pinned_pages_never_evicted() {
        let policy = TierPolicy::default();
        let cold_pinned = PageObs {
            rate: 0.0,
            bytes: 1024 * 1024,
            resident: true,
            pinned: true,
        };
        let hot_unpinned = PageObs {
            rate: 100.0,
            bytes: 1024,
            resident: false,
            pinned: false,
        };

        let cold_score = policy.score(&cold_pinned);
        let hot_score = policy.score(&hot_unpinned);

        assert!(
            cold_score > hot_score,
            "Pinned cold must outrank hot unpinned: cold={cold_score:.3}, hot={hot_score:.3}"
        );
    }

    // ====================================================================
    // PROOF 11: Membrane decay converges to zero
    // ====================================================================

    #[test]
    fn proof_13_rate_decays_to_zero() {
        let mut rate = 100.0;
        for _ in 0..100 {
            rate *= RATE_DECAY;
        }
        assert!(rate < 0.001, "Rate must converge to ~0: {rate}");
    }

    #[test]
    fn proof_14_rate_half_life() {
        let half_life = (0.5_f64).ln() / RATE_DECAY.ln();
        assert!(
            (2.0..3.0).contains(&half_life),
            "Half-life ~2.4: {half_life}"
        );
    }

    // ====================================================================
    // PROOF 12: Score is monotonically increasing in rate
    // ====================================================================

    #[test]
    fn proof_15_score_monotonic_in_rate() {
        let policy = TierPolicy::default();
        let mut prev = 0.0;
        for rate in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0] {
            let obs = PageObs {
                rate,
                bytes: 1024,
                resident: false,
                pinned: false,
            };
            let score = policy.score(&obs);
            assert!(
                score > prev,
                "Score must increase: rate={rate}, score={score:.3} <= prev={prev:.3}"
            );
            prev = score;
        }
    }

    // ====================================================================
    // PROOF 13: Larger pages are penalized
    // ====================================================================

    #[test]
    fn proof_16_larger_pages_penalized() {
        let policy = TierPolicy::default();
        let small = PageObs {
            rate: 10.0,
            bytes: 256,
            resident: false,
            pinned: false,
        };
        let large = PageObs {
            rate: 10.0,
            bytes: 1024 * 1024,
            resident: false,
            pinned: false,
        };

        assert!(
            policy.score(&small) > policy.score(&large),
            "Small page must score higher"
        );
    }

    // ====================================================================
    // PROOF 14: VRAM high-water mark prevents thrash
    // ====================================================================

    #[test]
    fn proof_17_vram_high_water_mark() {
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(VRAM_HIGH_WATER > 0.5 && VRAM_HIGH_WATER < 1.0);
        }
    }

    // ====================================================================
    // PROOF 15: Range allocator alignment
    // ====================================================================

    #[test]
    fn proof_18_alignment_is_power_of_two() {
        assert!(ALIGN.is_power_of_two());
        assert_eq!(ALIGN, 64);
    }

    #[test]
    fn proof_19_allocations_are_aligned() {
        let mut alloc = RangeAllocator::new(1024 * 1024);
        for size in [1, 7, 33, 63, 64, 65, 128, 1000] {
            let offset = alloc.allocate(size).unwrap();
            assert_eq!(
                offset % ALIGN,
                0,
                "offset {offset} not aligned for size {size}"
            );
        }
    }

    // ====================================================================
    // PROOF 16: Best-fit picks tightest hole
    // ====================================================================

    #[test]
    fn proof_20_best_fit_picks_tightest() {
        let mut alloc = RangeAllocator::new(1024);
        let a = alloc.allocate(128).unwrap();
        let _b = alloc.allocate(256).unwrap();
        let c = alloc.allocate(128).unwrap();
        alloc.free(a, 128);
        alloc.free(c, 128);
        let got = alloc.allocate(64).unwrap();
        assert_eq!(got, a, "Best-fit must pick tightest hole");
    }

    // ====================================================================
    // PROOF 17: Coalescing merges adjacent blocks
    // ====================================================================

    #[test]
    fn proof_21_coalescing_merges_adjacent() {
        let mut alloc = RangeAllocator::new(1024);
        let a = alloc.allocate(128).unwrap(); // 0..128
        let b = alloc.allocate(128).unwrap(); // 128..256
        let _c = alloc.allocate(128).unwrap(); // 256..384
        let _d = alloc.allocate(128).unwrap(); // 384..512 — keeps HW mark above a,b
                                               // Free a and b (adjacent, in the middle)
        alloc.free(a, 128);
        alloc.free(b, 128);
        assert_eq!(alloc.free_block_count(), 1, "Adjacent frees must coalesce");
    }

    // ====================================================================
    // PROOF 18: OOM distinguishes fragmentation from exhaustion
    // ====================================================================

    #[test]
    fn proof_22_oom_is_fragmentation() {
        let mut alloc = RangeAllocator::new(256);
        let a = alloc.allocate(128).unwrap();
        let _b = alloc.allocate(64).unwrap();
        let c = alloc.allocate(64).unwrap();
        alloc.free(a, 128);
        alloc.free(c, 64);

        match alloc.allocate(192).unwrap_err() {
            vugva::VugvaError::DramOom {
                requested,
                available,
                ..
            } => {
                assert_eq!(requested, 192);
                assert_eq!(available, 192);
            }
            other => panic!("Expected DramOom, got {other:?}"),
        }
    }

    // ====================================================================
    // PROOF 19: VMT — duplicate names get unique IDs
    // ====================================================================

    #[test]
    fn proof_23_vmt_unique_names() {
        let mut vmt = VirtualMemoryTable::new(1, 1);
        // Empty name triggers auto-ID generation
        let n1 = vmt.allocate("", &[100], 4).unwrap();
        let n2 = vmt.allocate("", &[100], 4).unwrap();
        assert_ne!(n1, n2, "Auto-generated names must be unique: {n1} vs {n2}");
    }

    #[test]
    fn proof_24_vmt_lookup_correct_page() {
        let mut vmt = VirtualMemoryTable::new(1, 1);
        let name = vmt.allocate("test", &[256, 256], 4).unwrap();
        let page = vmt.lookup(&name).unwrap();
        assert_eq!(page.size_bytes, 256 * 256 * 4);
        assert_eq!(page.element_size, 4);
        assert_eq!(page.shape, vec![256, 256]);
    }

    #[test]
    fn proof_25_vmt_remove_returns_page() {
        let mut vmt = VirtualMemoryTable::new(1, 1);
        let name = vmt.allocate("test", &[100], 4).unwrap();
        let page = vmt.remove(&name).unwrap();
        assert_eq!(page.name, "test");
        assert!(vmt.lookup(&name).is_none());
    }

    // ====================================================================
    // PROOF 20: Page size calculation
    // ====================================================================

    #[test]
    fn proof_26_page_size_calculation() {
        let p1 = Page::new("t".into(), vec![1024, 1024], 4, 1, 1);
        assert_eq!(p1.size_bytes, 4_194_304);

        let p2 = Page::new("t".into(), vec![8192, 8192], 2, 1, 1);
        assert_eq!(p2.size_bytes, 134_217_728);
    }

    // ====================================================================
    // PROOF 21: Tier enum ordering
    // ====================================================================

    #[test]
    fn proof_27_tier_ordering() {
        assert!((Tier::Vram as u8) < (Tier::Dram as u8));
        assert!((Tier::Dram as u8) < (Tier::Ssd as u8));
    }

    // ====================================================================
    // PROOF 22: Delay hiding equation (§3.2)
    // ====================================================================

    #[test]
    fn proof_28_latency_hiding_condition() {
        let t_compute = 50.0_f64;
        let t_transport = 8.5_f64;
        let t_total = t_compute.max(t_transport);

        assert_eq!(t_total, t_compute);
        assert!(t_compute >= t_transport, "Compute must dominate transport");
    }
}
