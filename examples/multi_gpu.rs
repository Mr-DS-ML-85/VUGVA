//! Multi-GPU sharding example — large allocation splits across GPUs.
//!
//! Run:  cargo run --example multi_gpu
//! (requires 2+ GPUs, or uses single-GPU fast path)

use vugva::allocator::VugvaEngine;

fn main() -> vugva::Result<()> {
    println!("=== VUGVA Multi-GPU Sharding ===\n");

    let gpus: Vec<i32> = (0..4).collect();
    let mut engine = VugvaEngine::new(&gpus)?;

    let gpu_count = engine.cluster.ordinals.len();
    println!("Using {gpu_count} GPU(s): {:?}", engine.cluster.ordinals);

    // Allocate 256MB — above SINGLE_GPU_LIMIT, so it shards
    let shape = &[1024, 1024, 64]; // 64M elements × 4 bytes = 256MB
    let name = engine.allocate("big_tensor", shape, 4)?;
    let page = engine.vmt.lookup(&name).unwrap();

    println!("\nAllocated: {name}");
    println!(
        "  Total:   {} bytes ({:.1} MB)",
        page.size_bytes,
        page.size_bytes as f64 / 1048576.0
    );
    println!("  Chunks:  {}", page.vram_chunks.len());
    for (i, chunk) in page.vram_chunks.iter().enumerate() {
        println!(
            "    [{i}] GPU {} — {} bytes",
            chunk.gpu_ordinal, chunk.size_bytes
        );
    }

    // Access on each GPU — collect ordinals first to avoid borrow conflict
    let ordinals: Vec<i32> = engine.cluster.ordinals.clone();
    for ord in &ordinals {
        let ptr = engine.access(&name, *ord)?;
        println!("  GPU {ord} ptr: 0x{ptr:x}");
    }

    engine.free(&name)?;
    println!("\nFreed successfully.");

    Ok(())
}
