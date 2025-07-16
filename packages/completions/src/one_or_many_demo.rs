{% if include_examples %}
//! OneOrMany examples - DELETE THIS MODULE IF NOT NEEDED
//!
//! Shows usage of OneOrMany for non-empty collections with type safety guarantees.

use cyrup_sugars::OneOrMany;

pub fn run() {
    println!("📦 OneOrMany Examples:");
    
    // Single item
    let single = OneOrMany::one("primary-server");
    println!("  Single: {:?}", single);
    println!("  First: {}", single.first());
    
    // Multiple items (guaranteed non-empty)
    let multiple = OneOrMany::many(vec!["server1", "server2", "server3"]).unwrap();
    println!("  Multiple: {:?}", multiple);
    println!("  First: {}", multiple.first());
    let rest = multiple.rest();
    println!("  Rest: {:?}", rest);
    
    // Transform operations
    let doubled = multiple.map(|s| format!("{}-v2", s));
    println!("  Mapped: {:?}", doubled);
    
    println!();
}
{% endif %}