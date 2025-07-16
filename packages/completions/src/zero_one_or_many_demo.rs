{% if include_examples %}
//! ZeroOneOrMany examples - DELETE THIS MODULE IF NOT NEEDED
//!
//! Shows usage of ZeroOneOrMany for flexible collections with zero allocations.

use cyrup_sugars::ZeroOneOrMany;

pub fn run() {
    println!("📦 ZeroOneOrMany Examples:");
    
    // Start empty
    let mut collection = ZeroOneOrMany::none();
    println!("  Empty: {:?}", collection);
    
    // Add items (transformations)
    collection = collection.with_pushed("middleware1");
    println!("  After first push: {:?}", collection);
    
    collection = collection.with_pushed("middleware2");
    println!("  After second push: {:?}", collection);
    
    // Pattern matching for safe handling
    match &collection {
        ZeroOneOrMany::None => println!("  No items"),
        ZeroOneOrMany::One(item) => println!("  Single item: {}", item),
        ZeroOneOrMany::Many(items) => println!("  Multiple items: {:?}", items),
    }
    
    // Merge collections
    let other = ZeroOneOrMany::one("middleware3");
    let merged = ZeroOneOrMany::merge(vec![collection, other]);
    println!("  Merged: {:?}", merged);
    
    println!();
}
{% endif %}