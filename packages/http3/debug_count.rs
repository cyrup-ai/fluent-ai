use fluent_ai_http3::json_path::compiler::JsonPathParser;
fn main() {
    let expr = "$[?count(@.missing[*]) == 999]";
    match JsonPathParser::compile(expr) {
        Ok(_) => println!("✓ Expression compiled successfully"),
        Err(e) => println!("✗ Error: {}", e),
    }
}
