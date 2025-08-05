use fluent_ai_http3::json_path::parser::JsonPathParser;

fn main() {
    println!("Testing JSONPath parser for pattern: $..*");
    
    match JsonPathParser::compile("$..*") {
        Ok(_parser) => {
            println!("✅ Successfully compiled: $..*");
        }
        Err(e) => {
            println!("❌ Failed to compile: $..*");
            println!("Error: {}", e);
        }
    }
}