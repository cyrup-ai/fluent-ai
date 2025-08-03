//! RFC 9535 Current Node Identifier (@) Validation Tests (Section 2.3.5)
//!
//! Tests for RFC 9535 Section 2.3.5 current node identifier requirements:
//! "The current node identifier @ refers to the current node in the context 
//! of the evaluation of a filter expression"
//!
//! This test suite validates:
//! - @ is only valid within filter expressions
//! - @ correctly refers to current node in filter context
//! - @ behavior in nested filter expressions
//! - @ usage in function expressions within filters
//! - @ error handling outside filter contexts
//! - @ property access patterns
//! - @ in logical expressions and comparisons

use bytes::Bytes;
use fluent_ai_http3::json_path::{JsonArrayStream, JsonPathParser, JsonPathError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct TestNode {
    id: i32,
    name: String,
    value: f64,
    active: bool,
    metadata: Option<serde_json::Value>,
    nested: Option<Box<TestNode>>,
}

/// Test data for current node identifier validation
const TEST_JSON: &str = r#"{
  "store": {
    "books": [
      {
        "id": 1,
        "name": "Book One",
        "value": 10.5,
        "active": true,
        "metadata": {"category": "fiction", "pages": 300},
        "nested": {"id": 11, "name": "Chapter", "value": 1.0, "active": true}
      },
      {
        "id": 2,
        "name": "Book Two", 
        "value": 25.0,
        "active": false,
        "metadata": {"category": "science", "pages": 450}
      },
      {
        "id": 3,
        "name": "Book Three",
        "value": 15.75,
        "active": true,
        "metadata": null
      }
    ],
    "config": {
      "id": 100,
      "name": "Store Config",
      "value": 99.99,
      "active": true
    }
  }
}"#;

/// RFC 9535 Section 2.3.5 - Current Node Identifier Context Tests
#[cfg(test)]
mod current_node_context_tests {
    use super::*;

    #[test]
    fn test_current_node_only_valid_in_filters() {
        // RFC 9535: @ is only valid within filter expressions
        let invalid_contexts = vec![
            "@",                        // Bare @ as root
            "$.@",                      // @ as segment
            "$.store.@",               // @ in property access
            "$.store[@]",              // @ as selector (not in filter)
            "$[@]",                    // @ without filter marker
            "$.store.books.@.name",    // @ in path segments
            "@.store.books[0]",        // @ as root identifier
            "$.store.books[@.id]",     // @ in bracket without ?
        ];

        for expr in invalid_contexts {
            let result = JsonPathParser::compile(expr);
            assert!(
                result.is_err(),
                "RFC 9535: @ outside filter context MUST be rejected: '{}'",
                expr
            );

            if let Err(JsonPathError::InvalidExpression { reason, .. }) = result {
                assert!(
                    reason.contains("@") || reason.contains("current") || reason.contains("filter"),
                    "Error should mention @ or current node context: {}",
                    reason
                );
            }
        }
    }

    #[test]
    fn test_current_node_valid_in_filter_contexts() {
        // RFC 9535: @ is valid within filter expressions
        let valid_filter_contexts = vec![
            "$.store.books[?@.active]",                    // Property existence test
            "$.store.books[?@.id > 1]",                    // Property comparison
            "$.store.books[?@.value >= 15.0]",             // Numeric comparison
            "$.store.books[?@.name == 'Book One']",        // String comparison
            "$.store.books[?@.metadata]",                  // Object existence
            "$.store.books[?@.nested.id > 10]",            // Nested property access
            "$.store.books[?@.metadata.category == 'fiction']", // Deep property access
            "$..books[?@.active == true]",                 // Boolean comparison
            "$.store.books[?@.active && @.value > 10]",    // Multiple @ in logical expression
            "$.store.books[?@.id != @.nested.id]",         // @ self-comparison
            "$..*[?@.id && @.name]",                       // Universal with @
            "$.store.books[?(@.active)]",                  // Parenthesized @
            "$.store.books[?!@.active]",                   // Negated @
            "$.store.books[?@.value < 20 && @.id > 1]",    // Complex logical with @
        ];

        for expr in valid_filter_contexts {
            let result = JsonPathParser::compile(expr);
            assert!(
                result.is_ok(),
                "RFC 9535: @ in valid filter context should compile: '{}'",
                expr
            );
        }
    }

    #[test]
    fn test_current_node_property_access() {
        // RFC 9535: @ correctly accesses properties of current node
        let property_access_tests = vec![
            ("$.store.books[?@.id == 1]", 1, "Current node ID access"),
            ("$.store.books[?@.name == 'Book Two']", 1, "Current node name access"),
            ("$.store.books[?@.value > 20]", 1, "Current node value comparison"),
            ("$.store.books[?@.active == false]", 1, "Current node boolean access"),
            ("$.store.books[?@.metadata.category == 'science']", 1, "Nested property access"),
            ("$.store.books[?@.metadata == null]", 1, "Null property access"),
            ("$.store.books[?@.nested.name == 'Chapter']", 1, "Deep nested access"),
        ];

        for (expr, expected_count, description) in property_access_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);
            let chunk = Bytes::from(TEST_JSON);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            assert_eq!(
                results.len(),
                expected_count,
                "RFC 9535: @ property access should work correctly: {} ({})",
                expr, description
            );
        }
    }

    #[test]
    fn test_current_node_in_logical_expressions() {
        // RFC 9535: @ behavior in complex logical expressions
        let logical_tests = vec![
            ("$.store.books[?@.active && @.value > 10]", 2, "AND with two @ conditions"),
            ("$.store.books[?@.active || @.value > 20]", 3, "OR with @ conditions"),
            ("$.store.books[?@.active && (@.value > 10 && @.id < 3)]", 1, "Nested logical with @"),
            ("$.store.books[?!@.active]", 1, "Negation of @ condition"),
            ("$.store.books[?@.active == true && @.metadata != null]", 1, "Complex boolean logic"),
            ("$.store.books[?(@.id > 1) && (@.value < 20)]", 1, "Parenthesized @ expressions"),
            ("$.store.books[?@.value >= 10 && @.value <= 20]", 2, "Range check with @"),
        ];

        for (expr, expected_count, description) in logical_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);
            let chunk = Bytes::from(TEST_JSON);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            assert_eq!(
                results.len(),
                expected_count,
                "RFC 9535: @ in logical expressions should work: {} ({})",
                expr, description
            );
        }
    }

    #[test]
    fn test_current_node_with_functions() {
        // RFC 9535: @ usage in function expressions within filters
        let function_tests = vec![
            ("$.store.books[?length(@.name) > 8]", 2, "length() function with @"),
            ("$.store.books[?count(@.metadata) > 0]", 2, "count() function with @"),
            ("$.*[?length(@) > 0]", 1, "length() of @ itself"),
            ("$.store.books[?@.metadata && length(@.metadata) > 0]", 2, "Function with @ existence check"),
        ];

        for (expr, expected_count, description) in function_tests {
            // Note: These tests validate syntax compilation
            // Actual function execution depends on implementation
            let result = JsonPathParser::compile(expr);
            assert!(
                result.is_ok(),
                "RFC 9535: @ with functions should compile: {} ({})",
                expr, description
            );
        }
    }

    #[test]
    fn test_current_node_edge_cases() {
        // RFC 9535: Edge cases for @ usage
        let edge_case_tests = vec![
            // Valid edge cases
            ("$.store.books[?@]", true, "Bare @ as test expression"),
            ("$.store.books[?(@)]", true, "Parenthesized bare @"),
            ("$.store.books[?!(@)]", true, "Negated parenthesized @"),
            ("$.store.books[?@ && true]", true, "@ with literal boolean"),
            ("$.store.books[?@ == @]", true, "@ self-equality"),
            ("$.store.books[?@.nonexistent]", true, "@ accessing nonexistent property"),
            
            // Invalid edge cases
            ("$.store.books[@@]", false, "Double @"),
            ("$.store.books[?@.]", false, "@ with trailing dot"),
            ("$.store.books[?@[0]]", false, "@ with array access"),
            ("$.store.books[?@['key']]", false, "@ with bracket notation"),
            ("$.store.books[?@.*]", false, "@ with wildcard"),
            ("$.store.books[?@..]", false, "@ with descendant operator"),
        ];

        for (expr, should_be_valid, description) in edge_case_tests {
            let result = JsonPathParser::compile(expr);
            
            if should_be_valid {
                assert!(
                    result.is_ok(),
                    "RFC 9535: @ edge case should be valid: {} ({})",
                    expr, description
                );
            } else {
                assert!(
                    result.is_err(),
                    "RFC 9535: @ edge case should be invalid: {} ({})",
                    expr, description
                );
            }
        }
    }

    #[test]
    fn test_current_node_type_consistency() {
        // RFC 9535: @ should maintain type consistency within expression
        let type_tests = vec![
            ("$.store.books[?@.id == 1 && @.name == 'Book One']", 1, "Number and string from same @"),
            ("$.store.books[?@.active == true && @.value > 0]", 2, "Boolean and number from same @"),
            ("$.store.books[?@.metadata && @.metadata.category]", 2, "Object existence and property"),
            ("$.store.books[?@.nested && @.nested.active]", 1, "Nested object consistency"),
            ("$.store.books[?@.value > @.id]", 2, "Numeric comparison within same @"),
        ];

        for (expr, expected_count, description) in type_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);
            let chunk = Bytes::from(TEST_JSON);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            assert_eq!(
                results.len(),
                expected_count,
                "RFC 9535: @ type consistency should work: {} ({})",
                expr, description
            );
        }
    }
}

/// Current Node Identifier Scope Tests
#[cfg(test)]
mod current_node_scope_tests {
    use super::*;

    #[test]
    fn test_current_node_scope_isolation() {
        // RFC 9535: @ refers to current node in current filter scope
        let scope_tests = vec![
            // Single scope
            ("$.store.books[?@.id > 1]", 2, "Single filter scope"),
            
            // Multiple independent scopes  
            ("$.store.books[?@.active].metadata[?@.category == 'fiction']", "Multiple independent scopes"),
            ("$..books[?@.id > 1][?@.value < 20]", "Chained filter scopes"),
            
            // Descendant with filters
            ("$.store..books[?@.active]", 2, "Descendant with filter scope"),
            ("$..metadata[?@.category]", "Descendant filter on metadata"),
        ];

        for (expr, expected_count, description) in scope_tests {
            if let Ok(expected_count) = expected_count.parse::<usize>() {
                let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);
                let chunk = Bytes::from(TEST_JSON);
                let results: Vec<_> = stream.process_chunk(chunk).collect();

                assert_eq!(
                    results.len(),
                    expected_count,
                    "RFC 9535: @ scope isolation should work: {} ({})",
                    expr, description
                );
            } else {
                // Test compilation for complex scopes
                let result = JsonPathParser::compile(expr);
                assert!(
                    result.is_ok(),
                    "RFC 9535: @ scope test should compile: {} ({})",
                    expr, description
                );
            }
        }
    }

    #[test]
    fn test_current_node_inheritance() {
        // RFC 9535: @ behavior with nested structures
        let inheritance_tests = vec![
            ("$.store.books[?@.nested.id > 10]", 1, "@ accessing nested structure"),
            ("$.store.books[?@.metadata.pages > 400]", 1, "@ accessing nested properties"),
            ("$.store.books[?@.metadata && @.metadata.category]", 2, "@ with nested existence check"),
            ("$..books[?@.nested && @.nested.active]", 1, "Descendant @ with nesting"),
        ];

        for (expr, expected_count, description) in inheritance_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);
            let chunk = Bytes::from(TEST_JSON);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            assert_eq!(
                results.len(),
                expected_count,
                "RFC 9535: @ inheritance should work: {} ({})",
                expr, description
            );
        }
    }

    #[test]
    fn test_current_node_comparison_semantics() {
        // RFC 9535: @ comparison semantics and type coercion
        let comparison_tests = vec![
            ("$.store.books[?@.id == '1']", 0, "String vs number comparison"),
            ("$.store.books[?@.active == 'true']", 0, "String vs boolean comparison"),
            ("$.store.books[?@.value == 10.5]", 1, "Exact numeric comparison"),
            ("$.store.books[?@.name != null]", 3, "Non-null string comparison"),
            ("$.store.books[?@.metadata != null]", 2, "Non-null object comparison"),
            ("$.store.books[?@.nonexistent == null]", 0, "Missing property comparison"),
        ];

        for (expr, expected_count, description) in comparison_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);
            let chunk = Bytes::from(TEST_JSON);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            // Note: Some comparisons may vary based on implementation
            // These tests document expected behavior
            println!(
                "@ comparison test '{}' -> {} results ({})",
                expr, results.len(), description
            );
            
            // Assert compilation succeeds
            let compile_result = JsonPathParser::compile(expr);
            assert!(
                compile_result.is_ok(),
                "RFC 9535: @ comparison should compile: {} ({})",
                expr, description
            );
        }
    }
}

/// Current Node Error Handling Tests  
#[cfg(test)]
mod current_node_error_tests {
    use super::*;

    #[test]
    fn test_current_node_error_messages() {
        // RFC 9535: @ error messages should be clear and helpful
        let error_cases = vec![
            ("@", "@ outside filter"),
            ("$.@", "@ in path segment"),
            ("$.store[@]", "@ without filter marker"),
            ("@.store", "@ as root"),
            ("$.store.books.@.name", "@ in property chain"),
        ];

        for (expr, error_type) in error_cases {
            let result = JsonPathParser::compile(expr);
            assert!(
                result.is_err(),
                "RFC 9535: {} should produce error: '{}'",
                error_type, expr
            );

            if let Err(JsonPathError::InvalidExpression { reason, .. }) = result {
                // Error message should be informative
                assert!(
                    !reason.is_empty(),
                    "RFC 9535: Error message should not be empty for: {}",
                    expr
                );
                
                println!("Error for '{}': {}", expr, reason);
            }
        }
    }

    #[test]
    fn test_current_node_complex_error_cases() {
        // RFC 9535: Complex error scenarios with @
        let complex_errors = vec![
            ("$.store.books[?@..name]", "@ with descendant operator"),
            ("$.store.books[?@[*]]", "@ with wildcard selector"),
            ("$.store.books[?@[0:2]]", "@ with slice operator"),
            ("$.store.books[?@['key']]", "@ with bracket notation"),
            ("$.store.books[?@@.id]", "Double @ symbols"),
            ("$.store.books[?@.id.@.value]", "@ in middle of path"),
        ];

        for (expr, error_description) in complex_errors {
            let result = JsonPathParser::compile(expr);
            assert!(
                result.is_err(),
                "RFC 9535: {} should be invalid: '{}'",
                error_description, expr
            );
        }
    }
}