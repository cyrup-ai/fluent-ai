use fluent_ai_http3::json_path::parser::JsonPathParser;

fn test_pattern(pattern: &str) {
    println!("Testing pattern: '{}'", pattern);
    match JsonPathParser::compile(pattern) {
        Ok(_) => println!("✅ Pattern '{}' compiled successfully", pattern),
        Err(e) => println!("❌ Pattern '{}' failed: {}", pattern, e),
    }
    println!("---");
}

fn main() {
    // Test simple patterns first
    test_pattern("$[1:2]");
    test_pattern("$[::2]");
    test_pattern("$[1::2]"); // This might be the problematic one
}