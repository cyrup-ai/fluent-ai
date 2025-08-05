use fluent_ai_http3::json_path::JsonPathParser;

fn main() {
    println!("Testing function parsing:");
    
    let expressions = vec![
        "$..book",
        "$[?count($..book)]",
        "$[?count(@..book)]",
        "$[?length(@)]",
    ];
    
    for expr in expressions {
        println!("\nTesting: '{}'", expr);
        match JsonPathParser::compile(expr) {
            Ok(_) => println!("  ✓ Success"),
            Err(e) => println!("  ✗ Error: {:?}", e),
        }
    }
}