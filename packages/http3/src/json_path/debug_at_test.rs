//! Debug test for @ parsing

#[cfg(test)]
mod at_parsing_debug {
    use crate::json_path::JsonPathParser;

    #[test]
    fn debug_at_parsing() {
        println!("Testing @ parsing in JSONPath expressions...");

        let test_expressions = vec![
            "$.store.books[?@.active]",
            "$.store.books[?@.id > 1]",
            "$.store.books[?@.value >= 15.0]",
        ];

        for expr in test_expressions {
            println!("\nTesting: {}", expr);
            match JsonPathParser::compile(expr) {
                Ok(_) => println!("  ✓ Compiled successfully"),
                Err(e) => println!("  ✗ Failed: {:?}", e),
            }
        }
    }
}
