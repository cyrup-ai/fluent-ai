// Temporary debug test for @ parsing

use fluent_ai_http3::json_path::JsonPathParser;

fn main() {
    println!("Testing @ parsing in JSONPath expressions...");
    
    let test_expressions = vec![
        "$.store.books[?@.active]",
        "$.store.books[?@.id > 1]", 
        "$.store.books[?@.value >= 15.0]",
    ];
    
    for expr in test_expressions {
        println!("\nTesting: {}", expr);
        match JsonPathParser::compile(expr) {
            Ok(_) => println!("  ✓ Compiled successfully"),
            Err(e) => println!("  ✗ Failed: {:?}", e),
        }
    }
}