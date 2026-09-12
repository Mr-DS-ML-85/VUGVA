//! Error handling patterns — graceful degradation without CUDA.
//!
//! Run:  cargo run --example error_handling
//! (works with or without GPU)

use vugva::allocator::VugvaEngine;
use vugva::gpu::GpuCluster;
use vugva::VugvaError;

fn main() {
    println!("=== VUGVA Error Handling ===\n");

    // 1. GPU discovery — graceful failure
    println!("[1] GPU Discovery");
    match GpuCluster::discover(&[0]) {
        Ok(cluster) => {
            let info = &cluster.infos[0];
            println!(
                "  Found: {} (sm_{}{})",
                info.name, info.compute_capability.0, info.compute_capability.1
            );
            println!("  VRAM:  {} MB", info.total_vram / 1024 / 1024);
        }
        Err(e) => println!("  No GPU: {e}"),
    }

    // 2. Engine creation — graceful failure
    println!("\n[2] Engine Creation");
    let mut engine = match VugvaEngine::new(&[0]) {
        Ok(e) => {
            println!("  Engine ready");
            e
        }
        Err(VugvaError::LibLoad { library, .. }) => {
            println!("  CUDA not available ({library} missing) — CPU-only mode");
            println!("  This is expected on machines without NVIDIA drivers.");
            return;
        }
        Err(e) => {
            println!("  Unexpected error: {e}");
            return;
        }
    };

    // 3. Allocation
    println!("\n[3] Allocation");
    let name = match engine.allocate("test", &[256, 256], 4) {
        Ok(n) => {
            println!("  Allocated: {n}");
            n
        }
        Err(e) => {
            println!("  Failed: {e}");
            return;
        }
    };

    // 4. Access
    println!("\n[4] Access");
    match engine.access(&name, 0) {
        Ok(ptr) => println!("  Device ptr: 0x{ptr:x}"),
        Err(e) => println!("  Failed: {e}"),
    }

    // 5. Free
    println!("\n[5] Free");
    match engine.free(&name) {
        Ok(()) => println!("  Freed OK"),
        Err(e) => println!("  Failed: {e}"),
    }

    // 6. Error type inspection
    println!("\n[6] Error Types");
    let err = VugvaError::DramOom {
        requested: 1024,
        available: 512,
        capacity: 4096,
    };
    println!("  {err}");

    let err = VugvaError::InvalidGpu(99);
    println!("  {err}");

    let err = VugvaError::UnknownAllocation("missing.tensor".into());
    println!("  {err}");

    println!("\nAll error paths tested!");
}
