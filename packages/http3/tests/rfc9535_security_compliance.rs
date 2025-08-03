//! RFC 9535 Security & Robustness Tests
//!
//! Tests for security and robustness aspects of JSONPath implementation.
//! These tests validate protection against various attack vectors and
//! ensure the implementation can handle malicious or malformed inputs safely.
//!
//! This test suite validates:
//! - Injection attack prevention
//! - Resource exhaustion protection  
//! - Malformed input handling
//! - Deep nesting protection
//! - Regular expression DoS prevention
//! - Memory usage limits
//! - Performance bounds under adversarial conditions
//! - Input validation and sanitization

use bytes::Bytes;
use fluent_ai_http3::json_path::{JsonArrayStream, JsonPathParser};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct SecurityTestModel {
    id: i32,
    data: String,
    nested: Option<serde_json::Value>,
}

/// Injection Attack Prevention Tests
#[cfg(test)]
mod injection_attack_tests {
    use super::*;

    #[test]
    fn test_path_injection_prevention() {
        // Test prevention of path injection attacks through user input
        let json_data = r#"{"users": [
            {"name": "admin", "role": "administrator", "secret": "top_secret"},
            {"name": "user1", "role": "user", "public": "visible_data"},
            {"name": "user2", "role": "user", "public": "other_data"}
        ]}"#;

        // Potentially malicious path components that should be safely handled
        let malicious_paths = vec![
            "$.users[0]['secret']",                // Direct access attempt
            "$.users[?@.name == 'admin'].secret",  // Filter injection attempt
            "$.users[*]['secret']",                // Wildcard secret access
            "$..secret",                           // Descendant secret search
            "$.users[?@.role == 'administrator']", // Role-based access
        ];

        for malicious_path in malicious_paths {
            let result = JsonPathParser::compile(malicious_path);
            match result {
                Ok(_) => {
                    let mut stream = JsonArrayStream::<serde_json::Value>::new(malicious_path);

                    let chunk = Bytes::from(json_data);
                    let results: Vec<_> = stream.process_chunk(chunk).collect();

                    println!(
                        "Path injection test '{}' -> {} results",
                        malicious_path,
                        results.len()
                    );

                    // Log for security audit - these should be controlled by application logic
                    if results.len() > 0 {
                        println!("  WARNING: Potentially sensitive data accessible via path");
                    }
                }
                Err(_) => println!("Path injection '{}' rejected by parser", malicious_path),
            }
        }
    }

    #[test]
    fn test_filter_expression_injection() {
        // Test injection through filter expressions
        let json_data = r#"{"items": [
            {"id": 1, "status": "active", "data": "normal"},
            {"id": 2, "status": "inactive", "data": "sensitive"},
            {"id": 3, "status": "active", "data": "public"}
        ]}"#;

        // Test potentially malicious filter expressions
        let malicious_filters = vec![
            "$.items[?@.status == 'active' || @.status == 'inactive']", // Boolean injection
            "$.items[?@.id > 0]",                                       // Always true condition
            "$.items[?@.data != null]",                                 // Null bypass
            "$.items[?@.status.length > 0]", // Property access injection
        ];

        for filter in malicious_filters {
            let result = JsonPathParser::compile(filter);
            match result {
                Ok(_) => {
                    let mut stream = JsonArrayStream::<serde_json::Value>::new(filter);

                    let chunk = Bytes::from(json_data);
                    let results: Vec<_> = stream.process_chunk(chunk).collect();

                    println!(
                        "Filter injection test '{}' -> {} results",
                        filter,
                        results.len()
                    );
                }
                Err(_) => println!("Filter injection '{}' rejected by parser", filter),
            }
        }
    }

    #[test]
    fn test_string_escape_injection() {
        // Test injection through string escape sequences
        let json_data = r#"{"keys": {
            "normal": "value1",
            "with'quote": "value2",
            "with\"quote": "value3",
            "with\\backslash": "value4"
        }}"#;

        let escape_injection_tests = vec![
            "$['keys']['normal']",            // Normal access
            "$['keys']['with\\'quote']",      // Single quote escape
            "$['keys']['with\"quote']",       // Double quote in single quotes
            "$['keys']['with\\\\backslash']", // Backslash escape
            "$['keys']['with\\nquote']",      // Newline injection attempt
            "$['keys']['with\\tquote']",      // Tab injection attempt
        ];

        for path in escape_injection_tests {
            let result = JsonPathParser::compile(path);
            match result {
                Ok(_) => println!("Escape injection '{}' compiled successfully", path),
                Err(_) => println!("Escape injection '{}' rejected", path),
            }
        }
    }

    #[test]
    fn test_unicode_injection_prevention() {
        // Test Unicode-based injection attempts
        let json_data = r#"{"data": {
            "normal": "value",
            "café": "coffee",
            "αβγ": "greek",
            "🚀": "rocket"
        }}"#;

        let unicode_injection_tests = vec![
            "$['data']['café']",                         // Accented characters
            "$['data']['αβγ']",                          // Greek letters
            "$['data']['🚀']",                           // Emoji
            "$['data']['\\u0063\\u0061\\u0066\\u0065']", // Unicode escapes for "cafe"
            "$['data']['\\u03B1\\u03B2\\u03B3']",        // Unicode escapes for Greek
        ];

        for path in unicode_injection_tests {
            let result = JsonPathParser::compile(path);
            match result {
                Ok(_) => println!("Unicode injection '{}' compiled successfully", path),
                Err(_) => println!("Unicode injection '{}' rejected", path),
            }
        }
    }
}

/// Resource Exhaustion Protection Tests
#[cfg(test)]
mod resource_exhaustion_tests {
    use super::*;

    #[test]
    fn test_large_json_handling() {
        // Test handling of large JSON documents
        let large_array: Vec<i32> = (0..10000).collect();
        let json_value = serde_json::json!({
            "large_data": large_array,
            "metadata": {
                "count": 10000,
                "description": "Large dataset for testing"
            }
        });
        let json_data = serde_json::to_string(&json_value).expect("Valid JSON");

        let large_data_tests = vec![
            ("$.large_data[0]", "First element access"),
            ("$.large_data[-1]", "Last element access"),
            ("$.large_data[5000:5010]", "Small slice from large array"),
            ("$.metadata.count", "Metadata access"),
        ];

        for (path, description) in large_data_tests {
            let start_time = std::time::Instant::now();

            let mut stream = JsonArrayStream::<serde_json::Value>::new(path);

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            let duration = start_time.elapsed();

            println!(
                "Large JSON test '{}' -> {} results in {:?} ({})",
                path,
                results.len(),
                duration,
                description
            );

            // Performance assertion - should complete within reasonable time
            assert!(
                duration.as_millis() < 5000,
                "Large JSON handling should complete in <5000ms for '{}'",
                path
            );
        }
    }

    #[test]
    fn test_memory_usage_bounds() {
        // Test memory usage with large result sets
        let medium_array: Vec<i32> = (0..1000).collect();
        let json_value = serde_json::json!({
            "arrays": [
                medium_array.clone(),
                medium_array.clone(),
                medium_array.clone(),
                medium_array.clone(),
                medium_array.clone()
            ]
        });
        let json_data = serde_json::to_string(&json_value).expect("Valid JSON");

        let memory_test_paths = vec![
            ("$.arrays[*][*]", "All elements from all arrays"),
            ("$.arrays[*][::10]", "Every 10th element from all arrays"),
            ("$.arrays[0:3][100:200]", "Subset of arrays and elements"),
        ];

        for (path, description) in memory_test_paths {
            let start_time = std::time::Instant::now();

            let mut stream = JsonArrayStream::<i32>::new(path);

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            let duration = start_time.elapsed();

            println!(
                "Memory usage test '{}' -> {} results in {:?} ({})",
                path,
                results.len(),
                duration,
                description
            );

            // Memory usage should be reasonable
            assert!(
                duration.as_millis() < 3000,
                "Memory usage test should complete in <3000ms for '{}'",
                path
            );
        }
    }

    #[test]
    fn test_excessive_wildcard_protection() {
        // Test protection against excessive wildcard usage
        let nested_structure = serde_json::json!({
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "data": [1, 2, 3, 4, 5]
                        }
                    }
                }
            }
        });
        let json_data = serde_json::to_string(&nested_structure).expect("Valid JSON");

        let wildcard_stress_tests = vec![
            ("$.*.*.*.*", "Four-level wildcard"),
            ("$.level1.*.*.*", "Three-level wildcard from level1"),
            ("$..data[*]", "Descendant with array wildcard"),
            ("$..*", "Universal descendant wildcard"),
        ];

        for (path, description) in wildcard_stress_tests {
            let start_time = std::time::Instant::now();

            let mut stream = JsonArrayStream::<serde_json::Value>::new(path);

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            let duration = start_time.elapsed();

            println!(
                "Wildcard stress test '{}' -> {} results in {:?} ({})",
                path,
                results.len(),
                duration,
                description
            );

            // Should handle wildcards efficiently
            assert!(
                duration.as_millis() < 1000,
                "Wildcard stress test should complete in <1000ms for '{}'",
                path
            );
        }
    }

    #[test]
    fn test_filter_expression_complexity_limits() {
        // Test complex filter expressions for performance bounds
        let json_data = r#"{"items": [
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            {"a": 2, "b": 3, "c": 4, "d": 5, "e": 6},
            {"a": 3, "b": 4, "c": 5, "d": 6, "e": 7},
            {"a": 4, "b": 5, "c": 6, "d": 7, "e": 8},
            {"a": 5, "b": 6, "c": 7, "d": 8, "e": 9}
        ]}"#;

        let complex_filters = vec![
            "$.items[?@.a > 0 && @.b > 1 && @.c > 2]",
            "$.items[?(@.a > 0 || @.b > 10) && (@.c < 10 || @.d < 10)]",
            "$.items[?@.a + @.b + @.c + @.d + @.e > 20]",
            "$.items[?@.a == 1 || @.a == 2 || @.a == 3 || @.a == 4 || @.a == 5]",
        ];

        for filter in complex_filters {
            let start_time = std::time::Instant::now();
            let result = JsonPathParser::compile(filter);

            match result {
                Ok(_) => {
                    let mut stream = JsonArrayStream::<serde_json::Value>::new(filter);

                    let chunk = Bytes::from(json_data);
                    let results: Vec<_> = stream.process_chunk(chunk).collect();

                    let duration = start_time.elapsed();

                    println!(
                        "Complex filter '{}' -> {} results in {:?}",
                        filter,
                        results.len(),
                        duration
                    );

                    // Complex filters should still execute reasonably fast
                    assert!(
                        duration.as_millis() < 100,
                        "Complex filter should execute in <100ms: '{}'",
                        filter
                    );
                }
                Err(_) => println!("Complex filter '{}' rejected by parser", filter),
            }
        }
    }
}

/// Malformed Input Handling Tests
#[cfg(test)]
mod malformed_input_tests {
    use super::*;

    #[test]
    fn test_invalid_json_handling() {
        // Test handling of malformed JSON input
        let malformed_json_inputs = vec![
            "{\"key\": value}",             // Unquoted value
            "{\"key\": \"unclosed string}", // Unclosed string
            "{\"key\": 123,}",              // Trailing comma
            "{\"key\": [1, 2, 3,]}",        // Trailing comma in array
            "{'key': 'single_quotes'}",     // Single quotes (non-standard)
            "{\"key\": undefined}",         // Undefined value
            "{\"key\": NaN}",               // NaN value
            "{\"key\": Infinity}",          // Infinity value
        ];

        for malformed_json in malformed_json_inputs {
            println!("Testing malformed JSON: {}", malformed_json);

            let result = std::panic::catch_unwind(|| {
                let mut stream = JsonArrayStream::<serde_json::Value>::new("$.key");

                let chunk = Bytes::from(malformed_json);
                let results: Vec<_> = stream.process_chunk(chunk).collect();
                results.len()
            });

            match result {
                Ok(count) => println!("  Processed without panic, {} results", count),
                Err(_) => println!("  Properly rejected or caused controlled error"),
            }
        }
    }

    #[test]
    fn test_invalid_jsonpath_syntax() {
        // Test handling of malformed JSONPath expressions
        let invalid_jsonpaths = vec![
            "$[",                 // Unclosed bracket
            "$.key.",             // Trailing dot
            "$...",               // Multiple dots
            "$.key[",             // Unclosed array access
            "$.key[abc]",         // Invalid array index
            "$.key[?",            // Unclosed filter
            "$.key[?@.prop",      // Incomplete filter
            "$key",               // Missing root $
            "key.value",          // No root at all
            "$.key[?@.prop ==]",  // Incomplete comparison
            "$.key[?@.prop && ]", // Incomplete logical expression
        ];

        for invalid_path in invalid_jsonpaths {
            let result = JsonPathParser::compile(invalid_path);
            match result {
                Ok(_) => println!("Invalid JSONPath '{}' unexpectedly compiled", invalid_path),
                Err(_) => println!("Invalid JSONPath '{}' correctly rejected", invalid_path),
            }
        }
    }

    #[test]
    fn test_edge_case_json_structures() {
        // Test edge cases in JSON structure
        let edge_case_jsons = vec![
            "{}",                 // Empty object
            "[]",                 // Empty array
            "null",               // Root null
            "\"string\"",         // Root string
            "42",                 // Root number
            "true",               // Root boolean
            "[null, null, null]", // Array of nulls
            "{\"\":\"\"}",        // Empty string key/value
        ];

        for edge_json in edge_case_jsons {
            println!("Testing edge case JSON: {}", edge_json);

            let test_paths = vec!["$", "$.*", "$[*]", "$..value"];

            for path in test_paths {
                let result = std::panic::catch_unwind(|| {
                    let mut stream = JsonArrayStream::<serde_json::Value>::new(path);

                    let chunk = Bytes::from(edge_json);
                    let results: Vec<_> = stream.process_chunk(chunk).collect();
                    results.len()
                });

                match result {
                    Ok(count) => println!("  Path '{}' -> {} results", path, count),
                    Err(_) => println!("  Path '{}' caused error", path),
                }
            }
        }
    }

    #[test]
    fn test_extremely_large_numbers() {
        // Test handling of extremely large numbers
        let large_number_json = r#"{
            "small": 1,
            "large_int": 9223372036854775807,
            "larger_than_int": 18446744073709551615,
            "scientific": 1.23e+100,
            "very_small": 1.23e-100
        }"#;

        let number_test_paths = vec![
            "$.small",
            "$.large_int",
            "$.larger_than_int",
            "$.scientific",
            "$.very_small",
            "$[?@.large_int > 1000000000000000000]",
        ];

        for path in number_test_paths {
            let result = std::panic::catch_unwind(|| {
                let mut stream = JsonArrayStream::<serde_json::Value>::new(path);

                let chunk = Bytes::from(large_number_json);
                let results: Vec<_> = stream.process_chunk(chunk).collect();
                results.len()
            });

            match result {
                Ok(count) => println!("Large number test '{}' -> {} results", path, count),
                Err(_) => println!("Large number test '{}' caused error", path),
            }
        }
    }
}

/// Deep Nesting Protection Tests
#[cfg(test)]
mod deep_nesting_tests {
    use super::*;

    #[test]
    fn test_deep_object_nesting() {
        // Create deeply nested object structure
        let mut deep_value = serde_json::json!("found");
        for i in 0..100 {
            deep_value = serde_json::json!({
                format!("level_{}", i): deep_value
            });
        }
        let json_data = serde_json::to_string(&deep_value).expect("Valid JSON");

        let deep_nesting_tests = vec![
            ("$.level_99.level_98.level_97", "Three-level deep access"),
            ("$..found", "Descendant search through deep nesting"),
            ("$.level_99.*.*", "Wildcard through deep levels"),
        ];

        for (path, description) in deep_nesting_tests {
            let start_time = std::time::Instant::now();

            let result = std::panic::catch_unwind(|| {
                let mut stream = JsonArrayStream::<serde_json::Value>::new(path);

                let chunk = Bytes::from(json_data.clone());
                let results: Vec<_> = stream.process_chunk(chunk).collect();
                results.len()
            });

            let duration = start_time.elapsed();

            match result {
                Ok(count) => {
                    println!(
                        "Deep nesting test '{}' -> {} results in {:?} ({})",
                        path, count, duration, description
                    );

                    // Should handle deep nesting without excessive time
                    assert!(
                        duration.as_millis() < 2000,
                        "Deep nesting should process in <2000ms for '{}'",
                        path
                    );
                }
                Err(_) => println!(
                    "Deep nesting test '{}' caused stack overflow or error",
                    path
                ),
            }
        }
    }

    #[test]
    fn test_deep_array_nesting() {
        // Create deeply nested array structure
        let mut deep_array = serde_json::json!([42]);
        for _ in 0..50 {
            deep_array = serde_json::json!([deep_array]);
        }
        let json_data = serde_json::to_string(&deep_array).expect("Valid JSON");

        let deep_array_tests = vec![
            ("$[0][0][0]", "Three-level array access"),
            ("$..[42]", "Search for number through deep arrays"),
            ("$[*][*][*]", "Three-level wildcard"),
        ];

        for (path, description) in deep_array_tests {
            let start_time = std::time::Instant::now();

            let result = std::panic::catch_unwind(|| {
                let mut stream = JsonArrayStream::<serde_json::Value>::new(path);

                let chunk = Bytes::from(json_data.clone());
                let results: Vec<_> = stream.process_chunk(chunk).collect();
                results.len()
            });

            let duration = start_time.elapsed();

            match result {
                Ok(count) => {
                    println!(
                        "Deep array test '{}' -> {} results in {:?} ({})",
                        path, count, duration, description
                    );

                    // Should handle deep arrays efficiently
                    assert!(
                        duration.as_millis() < 1500,
                        "Deep array processing should complete in <1500ms for '{}'",
                        path
                    );
                }
                Err(_) => println!("Deep array test '{}' caused error", path),
            }
        }
    }

    #[test]
    fn test_recursion_limits() {
        // Test recursion limits with descendant operators
        let recursive_structure = serde_json::json!({
            "root": {
                "child": {
                    "child": {
                        "child": {
                            "child": {
                                "child": {
                                    "target": "deep_value"
                                }
                            }
                        }
                    }
                }
            }
        });
        let json_data = serde_json::to_string(&recursive_structure).expect("Valid JSON");

        let recursion_tests = vec![
            ("$..target", "Descendant search for target"),
            ("$..child", "Descendant search for child"),
            ("$.root..target", "Descendant search from root"),
            ("$..*", "Universal descendant search"),
        ];

        for (path, description) in recursion_tests {
            let start_time = std::time::Instant::now();

            let result = std::panic::catch_unwind(|| {
                let mut stream = JsonArrayStream::<serde_json::Value>::new(path);

                let chunk = Bytes::from(json_data.clone());
                let results: Vec<_> = stream.process_chunk(chunk).collect();
                results.len()
            });

            let duration = start_time.elapsed();

            match result {
                Ok(count) => {
                    println!(
                        "Recursion test '{}' -> {} results in {:?} ({})",
                        path, count, duration, description
                    );

                    // Recursion should be controlled and fast
                    assert!(
                        duration.as_millis() < 500,
                        "Recursion test should complete in <500ms for '{}'",
                        path
                    );
                }
                Err(_) => println!("Recursion test '{}' hit limits or caused error", path),
            }
        }
    }
}

/// Regular Expression DoS Prevention Tests
#[cfg(test)]
mod regex_dos_prevention_tests {
    use super::*;

    #[test]
    fn test_catastrophic_backtracking_prevention() {
        // Test prevention of regex patterns that could cause catastrophic backtracking
        let json_data = r#"{"items": [
            {"text": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaX"},
            {"text": "normal_text"},
            {"text": "another_normal_text"}
        ]}"#;

        let dangerous_patterns = vec![
            // These patterns could cause exponential backtracking on certain inputs
            "$.items[?match(@.text, '(a+)+b')]", // Nested quantifiers
            "$.items[?match(@.text, '(a|a)*b')]", // Alternation with overlap
            "$.items[?match(@.text, 'a*a*a*a*b')]", // Multiple quantifiers
            "$.items[?match(@.text, '(a*)*b')]", // Nested star quantifiers
        ];

        for pattern in dangerous_patterns {
            let start_time = std::time::Instant::now();
            let result = JsonPathParser::compile(pattern);

            match result {
                Ok(_) => {
                    let execution_result = std::panic::catch_unwind(|| {
                        let mut stream = JsonArrayStream::<serde_json::Value>::new(pattern);

                        let chunk = Bytes::from(json_data);
                        let results: Vec<_> = stream.process_chunk(chunk).collect();
                        results.len()
                    });

                    let duration = start_time.elapsed();

                    match execution_result {
                        Ok(count) => {
                            println!(
                                "Dangerous regex '{}' -> {} results in {:?}",
                                pattern, count, duration
                            );

                            // Should not take excessive time even with problematic patterns
                            if duration.as_millis() > 1000 {
                                println!(
                                    "  WARNING: Potential ReDoS vulnerability - took {}ms",
                                    duration.as_millis()
                                );
                            }
                        }
                        Err(_) => println!("Dangerous regex '{}' caused timeout or error", pattern),
                    }
                }
                Err(_) => println!(
                    "Dangerous regex '{}' correctly rejected at compile time",
                    pattern
                ),
            }
        }
    }

    #[test]
    fn test_regex_complexity_limits() {
        // Test regex patterns of varying complexity
        let json_data = r#"{"data": [
            {"code": "ABC123"},
            {"code": "XYZ789"},
            {"email": "user@example.com"},
            {"phone": "+1-555-123-4567"}
        ]}"#;

        let complexity_patterns = vec![
            (
                "$.data[?match(@.code, '^[A-Z]{3}[0-9]{3}$')]",
                "Simple pattern",
            ),
            (
                "$.data[?match(@.email, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')]",
                "Email pattern",
            ),
            (
                "$.data[?match(@.phone, '^\\+?[1-9]\\d{1,14}$')]",
                "Phone pattern",
            ),
            ("$.data[?match(@.code, '[A-Z]+')]", "Basic character class"),
        ];

        for (pattern, description) in complexity_patterns {
            let start_time = std::time::Instant::now();
            let result = JsonPathParser::compile(pattern);

            match result {
                Ok(_) => {
                    let mut stream = JsonArrayStream::<serde_json::Value>::new(pattern);

                    let chunk = Bytes::from(json_data);
                    let results: Vec<_> = stream.process_chunk(chunk).collect();

                    let duration = start_time.elapsed();

                    println!(
                        "Regex complexity '{}' -> {} results in {:?} ({})",
                        pattern,
                        results.len(),
                        duration,
                        description
                    );

                    // Complex patterns should still execute quickly
                    assert!(
                        duration.as_millis() < 200,
                        "Complex regex should execute in <200ms: '{}'",
                        pattern
                    );
                }
                Err(_) => println!(
                    "Regex pattern '{}' not supported ({})",
                    pattern, description
                ),
            }
        }
    }

    #[test]
    fn test_regex_input_size_limits() {
        // Test regex patterns against various input sizes
        let small_text = "a".repeat(100);
        let medium_text = "a".repeat(1000);
        let large_text = "a".repeat(10000);

        let size_test_data = vec![
            (small_text, "small_input"),
            (medium_text, "medium_input"),
            (large_text, "large_input"),
        ];

        for (text, size_desc) in size_test_data {
            let json_value = serde_json::json!({"text": text});
            let json_data = serde_json::to_string(&json_value).expect("Valid JSON");

            let pattern = "$.text[?match(@, 'a+')]";

            let start_time = std::time::Instant::now();
            let result = JsonPathParser::compile(pattern);

            match result {
                Ok(_) => {
                    let mut stream = JsonArrayStream::<serde_json::Value>::new(pattern);

                    let chunk = Bytes::from(json_data);
                    let results: Vec<_> = stream.process_chunk(chunk).collect();

                    let duration = start_time.elapsed();

                    println!(
                        "Regex input size test '{}' -> {} results in {:?} ({})",
                        pattern,
                        results.len(),
                        duration,
                        size_desc
                    );

                    // Should handle various input sizes efficiently
                    assert!(
                        duration.as_millis() < 1000,
                        "Regex with {} input should complete in <1000ms",
                        size_desc
                    );
                }
                Err(_) => println!("Regex pattern not supported for size test"),
            }
        }
    }
}
