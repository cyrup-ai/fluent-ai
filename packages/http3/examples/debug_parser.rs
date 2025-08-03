use fluent_ai_http3::json_path::JsonPathParser;

fn main() {
    println!("Testing basic filter expression parsing...");
    
    let test_expressions = vec![
        "$.store.books[?@.active]",
        "$.store.books[?@.value >= 15.0]", 
        "@",
        "$.store.@",
        "$.store.books[*]",
    ];
    
    for expr in test_expressions {
        println!("\n--- Testing: {} ---", expr);
        match JsonPathParser::compile(expr) {
            Ok(parsed) => {
                println!("✅ Successfully compiled: {}", expr);
                println!("   Selectors: {:?}", parsed.selectors());
            }
            Err(e) => {
                println!("❌ Failed to compile: {}", expr);
                println!("   Error: {}", e);
            }
        }
    }
}