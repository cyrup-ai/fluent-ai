use fluent_ai_http3::json_path::JsonPathParser;

fn main() {
    let expressions = vec![
        "$.@name",
        "$.@",
        "@.store",
        "$['@']",
    ];
    
    for expr in expressions {
        println!("\n=== Testing: {} ===", expr);
        match JsonPathParser::compile(expr) {
            Ok(_) => println!("✅ SUCCESS: Compiled successfully"),
            Err(e) => println!("❌ ERROR: {}", e),
        }
    }
}