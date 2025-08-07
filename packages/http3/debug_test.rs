
use fluent_ai_http3::json_path::JsonPathParser;

#[test]
fn test_problematic_expression() {
    let expr = "$.store.book[?@.author.length]";
    let result = JsonPathParser::compile(expr);
    println!("Expression: {}", expr);
    match &result {
        Ok(_) => println!("SUCCESS: Compiled"),
        Err(e) => println!("ERROR: {}", e),
    }
    assert!(result.is_ok(), "Expression should compile: {}", expr);
}

