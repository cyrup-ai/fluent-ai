#[cfg(test)]
mod tests {
    use fluent_ai_http3::json_path::JsonPathParser;

    #[test]
    fn debug_function_parsing() {
        let test_cases = vec![
            "$..book",                    // Basic recursive descent
            "$[?@.price < 10]",          // Simple filter
            "$[?count(@..book)]",        // Function with JSONPath argument
        ];

        for case in test_cases {
            println!("Testing: '{}'", case);
            match JsonPathParser::compile(case) {
                Ok(_) => println!("  ✓ Success"),
                Err(e) => println!("  ✗ Error: {:?}", e),
            }
        }
    }
}