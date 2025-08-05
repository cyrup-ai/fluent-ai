use fluent_ai_http3::json_path::parser::JsonPathParser;

fn main() {
    println!("Testing JSONPath parser patterns:");
    
    let test_patterns = vec![
        "$",
        "$..*",
        "$..level1..*",
        "$..*..*",
    ];
    
    for pattern in test_patterns {
        println!("\nTesting pattern: {}", pattern);
        match JsonPathParser::compile(pattern) {
            Ok(_parser) => {
                println!("✅ Successfully compiled: {}", pattern);
            }
            Err(e) => {
                println!("❌ Failed to compile: {}", pattern);
                println!("Error: {}", e);
            }
        }
    }
}