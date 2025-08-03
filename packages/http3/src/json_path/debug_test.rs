//! Debug test for JSONPath function parsing infinite loop
//!
//! This is a minimal test to isolate the parsing issue with function calls

use super::parser::JsonPathParser;

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn debug_core_function_call() {
        // Test the simplest function call first
        let result = JsonPathParser::compile("$.test[?length() == 0]");
        println!("Core function result: {:?}", result);

        // This should work if the basic parsing is correct
        assert!(
            result.is_ok(),
            "Core function call should parse successfully"
        );
    }

    #[test]
    fn debug_function_with_property() {
        // Test function with property argument - this is the problematic case
        println!("Starting compilation of problematic expression...");
        let result = JsonPathParser::compile("$.items[?length(@.name) == 5]");
        println!("Property function result: {:?}", result);

        if result.is_err() {
            println!("Error details: {}", result.err().unwrap());
        }
    }

    #[test]
    fn debug_basic_property_access() {
        // Test basic property access without function
        let result = JsonPathParser::compile("$.items[?@.name == 'test']");
        println!("Basic property access result: {:?}", result);

        assert!(result.is_ok(), "Basic property access should work");
    }
}
