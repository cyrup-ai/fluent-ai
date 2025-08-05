use fluent_ai_http3::json_path::tokenizer::ExpressionParser;

fn main() {
    let expressions = vec![
        "$..book", 
        "$[?count($..book)]"
    ];
    
    for expr in expressions {
        println!("Expression: {}", expr);
        match ExpressionParser::new(expr).parse() {
            Ok(selectors) => {
                for (i, selector) in selectors.iter().enumerate() {
                    println!("  {}: {:?}", i, selector);
                }
            }
            Err(e) => {
                println!("  Error: {:?}", e);
            }
        }
        println!();
    }
}