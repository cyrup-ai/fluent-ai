use fluent_ai_http3::json_path::JsonPathParser;

fn main() {
    let test_expressions = vec![
        "$.store.book[?@.author.length]",  // This should compile but currently fails
        "$.store.book[?@.author]",         // This should compile 
        "$.store.book[?@.price > 10]",     // This should compile
    ];
    
    for expr in test_expressions {
        println!("Testing: {}", expr);
        match JsonPathParser::compile(expr) {
            Ok(_) => println!("  ✓ COMPILED SUCCESSFULLY"),
            Err(e) => println!("  ✗ COMPILATION ERROR: {}", e),
        }
        println!();
    }
}