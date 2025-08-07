use fluent_ai_http3::json_path::JsonPathParser;

fn main() {
    let test_expressions = vec![
        "$.store.book[?@.author]", // This should work
        "$.store.book[?@.author.length]", // This is failing
    ];

    for expr in test_expressions {
        println!("\n=== Testing: {} ===", expr);
        match JsonPathParser::compile(expr) {
            Ok(_) => println!("✓ SUCCESS: Expression compiled"),
            Err(e) => println!("✗ ERROR: {}", e),
        }
    }
}