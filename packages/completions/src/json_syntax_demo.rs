{% if include_examples %}
//! JSON syntax examples - DELETE THIS MODULE IF NOT NEEDED
//!
//! Shows usage of JSON object syntax in builder patterns.

pub fn run() {
    println!("🔧 JSON Syntax Examples:");
    
    // This demonstrates the {"key" => "value"} syntax
    // Note: This is a conceptual example - actual implementation depends on your builder
    
    // Example builder pattern that would use JSON syntax:
    println!("  JSON syntax works with builder patterns like:");
    println!("  .additional_params({{\"beta\" => \"true\"}})");
    println!("  .metadata({{\"key\" => \"val\", \"foo\" => \"bar\"}})");
    println!("  .config({{\"timeout\" => \"30\", \"retries\" => \"3\"}})");
    
    println!();
}
{% endif %}