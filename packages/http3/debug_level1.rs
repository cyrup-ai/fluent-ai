fn main() { 
    use fluent_ai_http3::json_path::compiler::JsonPathParser;
    println!("Testing $..level1:");
    match JsonPathParser::compile("$..level1") {
        Ok(_) => println!("✓ Compiled successfully"),
        Err(e) => println!("✗ Error: {}", e),
    }
}
