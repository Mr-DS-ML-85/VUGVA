//! Page state machine example — demonstrates valid/invalid transitions.
//!
//! Run:  cargo run --example page_states

use vugva::vmt::{Page, PageState, VirtualMemoryTable};

fn main() {
    println!("=== VUGVA Page State Machine ===\n");

    let mut vmt = VirtualMemoryTable::new(1, 1);

    // Allocate a page
    let name = vmt.allocate("test.page", &[1024], 4).unwrap();
    let page = vmt.lookup(&name).unwrap();
    println!("Initial state: {:?}", page.state);
    assert_eq!(page.state, PageState::Allocated);

    // Valid: Allocated → Resident
    let page = vmt.lookup_mut(&name).unwrap();
    page.state = PageState::Resident;
    println!("After activation: {:?}", page.state);

    // Valid: Resident → Warm (eviction)
    let page = vmt.lookup_mut(&name).unwrap();
    page.state = PageState::Warm;
    println!("After eviction:   {:?}", page.state);

    // Valid: Warm → Resident (promotion)
    let page = vmt.lookup_mut(&name).unwrap();
    page.state = PageState::Resident;
    println!("After promotion:  {:?}", page.state);

    // Test Page::transition
    println!("\n--- Testing Page::transition ---");
    let mut page = Page::new("t".into(), vec![10], 4, 1, 1);
    assert_eq!(
        page.state,
        PageState::Unmapped,
        "Page::new starts as Unmapped"
    );

    assert!(page.transition(PageState::Allocated).is_ok());
    println!("Unmapped → Allocated: OK");

    assert!(page.transition(PageState::Resident).is_ok());
    println!("Allocated → Resident: OK");

    assert!(page.transition(PageState::Warm).is_ok());
    println!("Resident → Warm:     OK");

    assert!(page.transition(PageState::Resident).is_ok());
    println!("Warm → Resident:     OK");

    page.transition(PageState::Resident).unwrap();
    page.transition(PageState::Warm).unwrap();
    assert!(page.transition(PageState::Cold).is_ok());
    println!("Warm → Cold:         OK");

    // Cold → Resident is valid (skip-warm promotion)
    assert!(page.transition(PageState::Resident).is_ok());
    println!("Cold → Resident:     OK (skip-warm)");

    // Invalid: Resident → Allocated
    let mut page3 = Page::new("t3".into(), vec![10], 4, 1, 1);
    page3.transition(PageState::Allocated).unwrap();
    page3.transition(PageState::Resident).unwrap();
    assert!(page3.transition(PageState::Allocated).is_err());
    println!("Resident → Allocated: REJECTED (correct)");

    // Invalid: Unmapped → Warm (must go through Allocated → Resident)
    let mut page4 = Page::new("t4".into(), vec![10], 4, 1, 1);
    assert!(page4.transition(PageState::Warm).is_err());
    println!("Unmapped → Warm:     REJECTED (correct)");

    println!("\nAll transitions verified!");
}
