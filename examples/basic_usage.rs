//! Basic usage example — allocate, access, free a tensor.
//!
//! Run:  cargo run --example basic_usage

use vugva::allocator::VugvaEngine;

fn main() -> vugva::Result<()> {
    println!("=== VUGVA Basic Usage ===\n");

    // 1. Create engine with GPU 0
    let mut engine = VugvaEngine::new(&[0])?;
    println!("Engine ready, VMT has {} pages", engine.vmt.len());

    // 2. Allocate a 4MB tensor (1024x1024 f32)
    let name = engine.allocate("embed.weight", &[1024, 1024], 4)?;
    println!("Allocated: {name}");

    // 3. Check the page in VMT
    let page = engine.vmt.lookup(&name).unwrap();
    println!("  State:  {:?}", page.state);
    println!("  Tier:   {:?}", page.tier);
    println!("  Size:   {} bytes", page.size_bytes);
    println!("  Chunks: {}", page.vram_chunks.len());

    // 4. Get device pointer on GPU 0
    let ptr = engine.access(&name, 0)?;
    println!("  Device ptr: 0x{ptr:x}");
    assert!(ptr > 0);

    // 5. Free it
    engine.free(&name)?;
    println!("Freed: {name}");
    assert!(engine.vmt.lookup(&name).is_none());

    println!("\nAll good!");
    Ok(())
}
