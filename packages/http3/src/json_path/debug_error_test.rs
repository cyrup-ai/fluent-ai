//! Debug test for @ error messages

#[cfg(test)]
mod at_error_debug {
    use crate::json_path::JsonPathParser;
    
    #[test]
    fn debug_at_error_messages() {
        println!("Testing @ error messages in JSONPath expressions...");
        
        let invalid_expressions = vec![
            "@",                        // Bare @ as root
            "$.@",                      // @ as segment
            "$.store[@]",              // @ as selector (not in filter)
        ];
        
        for expr in invalid_expressions {
            println!("\nTesting: {}", expr);
            match JsonPathParser::compile(expr) {
                Ok(_) => println!("  ✓ Compiled successfully (unexpected!)"),
                Err(e) => println!("  ✗ Failed: {:?}", e),
            }
        }
    }
}