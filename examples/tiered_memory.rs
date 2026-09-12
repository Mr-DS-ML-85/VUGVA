//! Tiered memory example — VRAM → DRAM → SSD hierarchy.
//!
//! Run:  cargo run --example tiered_memory

use vugva::allocator::VugvaEngine;

fn main() -> vugva::Result<()> {
    println!("=== VUGVA Tiered Memory (V2) ===\n");

    let mut engine = VugvaEngine::new(&[0])?;
    println!("Engine ready");

    let tensors = [
        ("layer.0.weight", vec![512, 512], 4usize),   // 1MB
        ("layer.1.weight", vec![1024, 1024], 4usize), // 4MB
        ("layer.2.weight", vec![2048, 2048], 4usize), // 16MB
        ("embed.table", vec![8192, 8192], 2usize),    // 128MB (f16)
    ];

    let mut names = Vec::new();
    for (name, shape, elem) in &tensors {
        let n = engine.allocate(name, shape, *elem)?;
        let page = engine.vmt.lookup(&n).unwrap();
        println!(
            "Allocated: {n} — {} bytes, state={:?}, tier={:?}",
            page.size_bytes, page.state, page.tier
        );
        names.push(n);
    }

    println!("\nVMT has {} pages", engine.vmt.len());

    let start = std::time::Instant::now();
    let ptr = engine.access(&names[0], 0)?;
    println!("Access layer.0: 0x{ptr:x} in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let ptr = engine.access(&names[1], 0)?;
    println!("Access layer.1: 0x{ptr:x} in {:?}", start.elapsed());

    println!("\nFreeing tensors...");
    for name in names.iter().rev() {
        engine.free(name)?;
        println!("  Freed: {name}");
    }

    println!("VMT has {} pages (should be 0)", engine.vmt.len());
    assert_eq!(engine.vmt.len(), 0);

    println!("\nTiered memory demo complete!");
    Ok(())
}
