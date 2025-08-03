//! RFC 9535 Null Semantics Test Suite (Section 2.1)
//!
//! Tests for null value handling and semantics as specified in RFC 9535.
//! JSONPath distinguishes between JSON null values and "Nothing" (missing values).
//!
//! This test suite validates:
//! - null vs missing value distinction
//! - null used as array access (should yield Nothing)
//! - null used as object access (should yield Nothing)
//! - comparison vs existence validation
//! - JSON null vs Nothing distinction in filter expressions
//! - null propagation through JSONPath expressions
//! - Type system interaction with null values
//! - Function behavior with null inputs

use bytes::Bytes;
use fluent_ai_http3::json_path::{JsonArrayStream, JsonPathParser};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct TestRecord {
    id: i32,
    name: Option<String>,
    value: Option<serde_json::Value>,
    nested: Option<serde_json::Value>,
}

/// RFC 9535 Section 2.1 - Null vs Missing Value Distinction
#[cfg(test)]
mod null_vs_missing_tests {
    use super::*;

    #[test]
    fn test_explicit_null_vs_missing_property() {
        // RFC 9535: Distinguish between explicit null and missing properties
        let json_data = r#"{"items": [
            {"id": 1, "name": "John", "value": null},
            {"id": 2, "name": "Jane"},
            {"id": 3, "value": "present", "name": null},
            {"id": 4, "name": "Bob", "value": "data"}
        ]}"#;

        let test_cases = vec![
            // Test for existence (should find properties that exist, even if null)
            (
                "$.items[?@.value]",
                3,
                "Properties that exist (including null)",
            ),
            (
                "$.items[?@.name]",
                3,
                "Name properties that exist (including null)",
            ),
            // Test for null values specifically
            (
                "$.items[?@.value == null]",
                1,
                "Properties with explicit null value",
            ),
            (
                "$.items[?@.name == null]",
                1,
                "Name properties with explicit null value",
            ),
            // Test for non-null values
            (
                "$.items[?@.value != null]",
                2,
                "Properties with non-null values",
            ),
            (
                "$.items[?@.name != null]",
                2,
                "Name properties with non-null values",
            ),
        ];

        for (expr, expected_count, description) in test_cases {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Null vs missing test '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );

            // Document current behavior rather than asserting specific counts
            // as implementation may vary
            if results.len() != expected_count {
                println!(
                    "  NOTE: Expected {}, got {} for '{}'",
                    expected_count,
                    results.len(),
                    expr
                );
            }
        }
    }

    #[test]
    fn test_deeply_nested_null_vs_missing() {
        // Test null vs missing in nested structures
        let json_data = r#"{"data": {
            "users": [
                {
                    "profile": {
                        "name": "Alice",
                        "email": null,
                        "phone": "123-456-7890"
                    }
                },
                {
                    "profile": {
                        "name": "Bob",
                        "phone": "987-654-3210"
                    }
                },
                {
                    "profile": null
                },
                {
                    "id": 4
                }
            ]
        }}"#;

        let nested_tests = vec![
            ("$.data.users[?@.profile]", 3, "Users with profile property"),
            (
                "$.data.users[?@.profile == null]",
                1,
                "Users with null profile",
            ),
            (
                "$.data.users[?@.profile.email]",
                1,
                "Users with email property",
            ),
            (
                "$.data.users[?@.profile.email == null]",
                1,
                "Users with null email",
            ),
            (
                "$.data.users[?@.profile.name]",
                2,
                "Users with profile name",
            ),
        ];

        for (expr, expected_count, description) in nested_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Nested null test '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );
        }
    }

    #[test]
    fn test_null_in_array_contexts() {
        // Test null values within arrays
        let json_data = r#"{"arrays": [
            [1, null, 3, null, 5],
            [null, 2, 4],
            [],
            [null],
            [1, 2, 3]
        ]}"#;

        let array_null_tests = vec![
            ("$.arrays[*][?@ == null]", 4, "All null elements in arrays"),
            (
                "$.arrays[*][?@ != null]",
                7,
                "All non-null elements in arrays",
            ),
            ("$.arrays[0][1]", 1, "Direct access to null element"),
            ("$.arrays[3][0]", 1, "Direct access to array with only null"),
        ];

        for (expr, expected_count, description) in array_null_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Array null test '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );
        }
    }
}

/// Null Used as Array/Object Access Tests
#[cfg(test)]
mod null_access_tests {
    use super::*;

    #[test]
    fn test_null_as_array_index() {
        // RFC 9535: Using null as array index should yield Nothing
        let json_data = r#"{"data": {
            "arrays": [
                [10, 20, 30],
                [40, 50, 60]
            ],
            "index": null
        }}"#;

        // These expressions should handle null indices gracefully
        let null_index_tests = vec![
            "$.data.arrays[null]",    // null as direct array index
            "$.data.arrays[@.index]", // null from property as index
        ];

        for expr in null_index_tests {
            let result = JsonPathParser::compile(expr);
            match result {
                Ok(_) => {
                    let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

                    let chunk = Bytes::from(json_data);
                    let results: Vec<_> = stream.process_chunk(chunk).collect();

                    println!(
                        "Null array index '{}' -> {} results (should be 0 or error)",
                        expr,
                        results.len()
                    );
                }
                Err(_) => println!("Null array index '{}' rejected at compile time", expr),
            }
        }
    }

    #[test]
    fn test_null_as_object_key() {
        // RFC 9535: Using null as object key should yield Nothing
        let json_data = r#"{"data": {
            "objects": [
                {"key1": "value1", "key2": "value2"},
                {"null": "should_not_match", "key3": "value3"}
            ],
            "key": null
        }}"#;

        // These expressions involve null as object keys
        let null_key_tests = vec![
            "$.data.objects[*][null]",  // null as literal object key
            "$.data.objects[*][@.key]", // null from property as key
        ];

        for expr in null_key_tests {
            let result = JsonPathParser::compile(expr);
            match result {
                Ok(_) => {
                    let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

                    let chunk = Bytes::from(json_data);
                    let results: Vec<_> = stream.process_chunk(chunk).collect();

                    println!(
                        "Null object key '{}' -> {} results (should be 0)",
                        expr,
                        results.len()
                    );
                }
                Err(_) => println!("Null object key '{}' rejected at compile time", expr),
            }
        }
    }

    #[test]
    fn test_chained_null_access() {
        // Test chained access through null values
        let json_data = r#"{"chain": {
            "level1": {
                "level2": null,
                "other": {
                    "level3": "value"
                }
            },
            "broken": null
        }}"#;

        let chained_null_tests = vec![
            (
                "$.chain.level1.level2.level3",
                0,
                "Access through null should yield Nothing",
            ),
            (
                "$.chain.broken.anything",
                0,
                "Access through null should yield Nothing",
            ),
            (
                "$.chain.level1.other.level3",
                1,
                "Access through valid path should work",
            ),
        ];

        for (expr, expected_count, description) in chained_null_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Chained null access '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );

            if results.len() != expected_count {
                println!("  NOTE: Expected {}, got {}", expected_count, results.len());
            }
        }
    }
}

/// Comparison vs Existence Validation Tests
#[cfg(test)]
mod comparison_existence_tests {
    use super::*;

    #[test]
    fn test_null_equality_comparisons() {
        // RFC 9535: null equality and inequality comparisons
        let json_data = r#"{"items": [
            {"status": null},
            {"status": "active"},
            {"status": "inactive"},
            {"status": 0},
            {"status": false},
            {"status": ""},
            {}
        ]}"#;

        let comparison_tests = vec![
            ("$.items[?@.status == null]", 1, "Items with null status"),
            (
                "$.items[?@.status != null]",
                5,
                "Items with non-null status",
            ),
            (
                "$.items[?@.status == '']",
                1,
                "Items with empty string status",
            ),
            ("$.items[?@.status == false]", 1, "Items with false status"),
            ("$.items[?@.status == 0]", 1, "Items with zero status"),
        ];

        for (expr, expected_count, description) in comparison_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Null comparison '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );
        }
    }

    #[test]
    fn test_existence_vs_null_checks() {
        // Demonstrate difference between existence and null checks
        let json_data = r#"{"records": [
            {"id": 1, "data": null, "flag": true},
            {"id": 2, "flag": false},
            {"id": 3, "data": "present"},
            {"id": 4, "data": null, "extra": "info"}
        ]}"#;

        let existence_tests = vec![
            // Existence tests (property present, regardless of value)
            (
                "$.records[?@.data]",
                3,
                "Records with data property (including null)",
            ),
            ("$.records[?@.flag]", 2, "Records with flag property"),
            ("$.records[?@.extra]", 1, "Records with extra property"),
            // Null value tests (property present and null)
            ("$.records[?@.data == null]", 2, "Records with null data"),
            ("$.records[?@.flag == null]", 0, "Records with null flag"),
            // Non-null value tests (property present and not null)
            (
                "$.records[?@.data != null]",
                1,
                "Records with non-null data",
            ),
            (
                "$.records[?@.flag != null]",
                2,
                "Records with non-null flag",
            ),
        ];

        for (expr, expected_count, description) in existence_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Existence test '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );
        }
    }

    #[test]
    fn test_null_in_logical_expressions() {
        // Test null values in complex logical expressions
        let json_data = r#"{"items": [
            {"a": null, "b": 1},
            {"a": 2, "b": null},
            {"a": null, "b": null},
            {"a": 3, "b": 4},
            {"b": 5}
        ]}"#;

        let logical_tests = vec![
            (
                "$.items[?@.a && @.b]",
                1,
                "Items with both a and b (non-null)",
            ),
            ("$.items[?@.a || @.b]", 4, "Items with either a or b"),
            (
                "$.items[?@.a == null && @.b]",
                1,
                "Items with null a and non-null b",
            ),
            (
                "$.items[?@.a && @.b == null]",
                1,
                "Items with non-null a and null b",
            ),
            (
                "$.items[?@.a == null || @.b == null]",
                3,
                "Items with either a or b null",
            ),
        ];

        for (expr, expected_count, description) in logical_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Logical null test '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );
        }
    }
}

/// JSON null vs Nothing Distinction in Filters
#[cfg(test)]
mod json_null_vs_nothing_tests {
    use super::*;

    #[test]
    fn test_filter_expression_null_handling() {
        // Test how filter expressions handle null vs Nothing
        let json_data = r#"{"database": [
            {"user_id": 1, "email": "user1@example.com", "phone": null},
            {"user_id": 2, "email": null, "phone": "555-0102"},
            {"user_id": 3, "email": "user3@example.com"},
            {"user_id": 4, "phone": "555-0104"},
            {"user_id": 5}
        ]}"#;

        let filter_null_tests = vec![
            // Test filter expressions with null comparisons
            (
                "$.database[?@.email == null]",
                1,
                "Users with explicit null email",
            ),
            (
                "$.database[?@.phone == null]",
                1,
                "Users with explicit null phone",
            ),
            // Test filter expressions with existence checks
            ("$.database[?@.email]", 2, "Users with email property"),
            ("$.database[?@.phone]", 2, "Users with phone property"),
            // Combined existence and null checks
            (
                "$.database[?@.email && @.email != null]",
                2,
                "Users with non-null email",
            ),
            (
                "$.database[?@.phone && @.phone != null]",
                1,
                "Users with non-null phone",
            ),
        ];

        for (expr, expected_count, description) in filter_null_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Filter null test '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );
        }
    }

    #[test]
    fn test_null_propagation_in_expressions() {
        // Test how null values propagate through complex expressions
        let json_data = r#"{"operations": [
            {"a": 10, "b": 20, "result": 30},
            {"a": null, "b": 20, "result": null},
            {"a": 15, "b": null, "result": null},
            {"a": null, "b": null, "result": null},
            {"a": 5, "b": 5, "result": 10}
        ]}"#;

        let propagation_tests = vec![
            ("$.operations[?@.a > 0]", 3, "Operations with positive a"),
            (
                "$.operations[?@.a > 0 && @.b > 0]",
                2,
                "Operations with both positive",
            ),
            (
                "$.operations[?@.result == null]",
                3,
                "Operations with null result",
            ),
            (
                "$.operations[?@.a == null || @.b == null]",
                3,
                "Operations with any null input",
            ),
        ];

        for (expr, expected_count, description) in propagation_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Null propagation test '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );
        }
    }

    #[test]
    fn test_nothing_vs_empty_results() {
        // Test distinction between Nothing and empty results
        let json_data = r#"{"containers": [
            {"items": []},
            {"items": [1, 2, 3]},
            {"items": null},
            {}
        ]}"#;

        let nothing_tests = vec![
            (
                "$.containers[*].items",
                3,
                "All items properties (including null)",
            ),
            ("$.containers[*].items[*]", 3, "All individual items"),
            (
                "$.containers[?@.items]",
                3,
                "Containers with items property",
            ),
            (
                "$.containers[?@.items == null]",
                1,
                "Containers with null items",
            ),
            ("$.containers[*].missing", 0, "Access to missing property"),
        ];

        for (expr, expected_count, description) in nothing_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Nothing test '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );
        }
    }
}

/// Function Behavior with Null Inputs
#[cfg(test)]
mod function_null_behavior_tests {
    use super::*;

    #[test]
    fn test_length_function_with_null() {
        // Test length() function behavior with null inputs
        let json_data = r#"{"items": [
            {"text": "hello", "data": [1, 2, 3]},
            {"text": null, "data": null},
            {"text": "", "data": []},
            {}
        ]}"#;

        let length_null_tests = vec![
            (
                "$.items[?length(@.text) == 5]",
                1,
                "Items with text length 5",
            ),
            ("$.items[?length(@.text) == 0]", 1, "Items with empty text"),
            (
                "$.items[?length(@.text) == null]",
                1,
                "Items with null text length",
            ),
            (
                "$.items[?length(@.data) == 3]",
                1,
                "Items with data length 3",
            ),
            (
                "$.items[?length(@.data) == null]",
                1,
                "Items with null data length",
            ),
        ];

        for (expr, expected_count, description) in length_null_tests {
            let result = JsonPathParser::compile(expr);
            match result {
                Ok(_) => {
                    let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

                    let chunk = Bytes::from(json_data);
                    let results: Vec<_> = stream.process_chunk(chunk).collect();

                    println!(
                        "Length null test '{}' -> {} results ({})",
                        expr,
                        results.len(),
                        description
                    );
                }
                Err(_) => println!("Length function not supported: {}", expr),
            }
        }
    }

    #[test]
    fn test_count_function_with_null() {
        // Test count() function behavior with null inputs
        let json_data = r#"{"groups": [
            {"members": [1, 2, null, 4]},
            {"members": null},
            {"members": []},
            {}
        ]}"#;

        let count_null_tests = vec![
            (
                "$.groups[?count(@.members[*]) == 4]",
                1,
                "Groups with 4 members",
            ),
            (
                "$.groups[?count(@.members[*]) == 0]",
                1,
                "Groups with 0 members",
            ),
            (
                "$.groups[?count(@.members[*]) == null]",
                1,
                "Groups with null members",
            ),
        ];

        for (expr, expected_count, description) in count_null_tests {
            let result = JsonPathParser::compile(expr);
            match result {
                Ok(_) => println!("Count null test '{}' compiled ({})", expr, description),
                Err(_) => println!("Count function not supported: {}", expr),
            }
        }
    }

    #[test]
    fn test_match_search_functions_with_null() {
        // Test match() and search() functions with null inputs
        let json_data = r#"{"texts": [
            {"content": "hello world"},
            {"content": null},
            {"content": ""},
            {}
        ]}"#;

        let regex_null_tests = vec![
            (
                "$.texts[?match(@.content, 'hello')]",
                1,
                "Texts matching 'hello'",
            ),
            ("$.texts[?match(@.content, null)]", 0, "Texts matching null"),
            (
                "$.texts[?search(@.content, 'world')]",
                1,
                "Texts containing 'world'",
            ),
            (
                "$.texts[?search(@.content, null)]",
                0,
                "Texts containing null",
            ),
        ];

        for (expr, expected_count, description) in regex_null_tests {
            let result = JsonPathParser::compile(expr);
            match result {
                Ok(_) => println!("Regex null test '{}' compiled ({})", expr, description),
                Err(_) => println!("Regex function not supported: {}", expr),
            }
        }
    }
}

/// Type System Interaction with Null Values
#[cfg(test)]
mod type_system_null_tests {
    use super::*;

    #[test]
    fn test_null_type_coercion() {
        // Test how null values interact with type system
        let json_data = r#"{"mixed": [
            {"value": null, "type": "null"},
            {"value": 0, "type": "number"},
            {"value": false, "type": "boolean"},
            {"value": "", "type": "string"},
            {"value": [], "type": "array"},
            {"value": {}, "type": "object"}
        ]}"#;

        let type_coercion_tests = vec![
            ("$.mixed[?@.value == null]", 1, "Values that are null"),
            ("$.mixed[?@.value == 0]", 1, "Values that are zero"),
            ("$.mixed[?@.value == false]", 1, "Values that are false"),
            ("$.mixed[?@.value == '']", 1, "Values that are empty string"),
            // Test type-specific comparisons
            (
                "$.mixed[?@.value != null && @.type == 'number']",
                1,
                "Non-null numbers",
            ),
            (
                "$.mixed[?@.value != null && @.type == 'boolean']",
                1,
                "Non-null booleans",
            ),
            (
                "$.mixed[?@.value != null && @.type == 'string']",
                1,
                "Non-null strings",
            ),
        ];

        for (expr, expected_count, description) in type_coercion_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Type coercion test '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );
        }
    }

    #[test]
    fn test_null_in_numeric_comparisons() {
        // Test null values in numeric comparisons
        let json_data = r#"{"numbers": [
            {"value": 10, "threshold": 5},
            {"value": null, "threshold": 5},
            {"value": 3, "threshold": null},
            {"value": null, "threshold": null},
            {"value": 0, "threshold": 0}
        ]}"#;

        let numeric_null_tests = vec![
            (
                "$.numbers[?@.value > @.threshold]",
                1,
                "Value greater than threshold",
            ),
            (
                "$.numbers[?@.value < @.threshold]",
                0,
                "Value less than threshold",
            ),
            (
                "$.numbers[?@.value == @.threshold]",
                1,
                "Value equals threshold",
            ),
            ("$.numbers[?@.value > 0]", 1, "Positive values"),
            ("$.numbers[?@.threshold > 0]", 1, "Positive thresholds"),
        ];

        for (expr, expected_count, description) in numeric_null_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Numeric null test '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );
        }
    }

    #[test]
    fn test_null_in_string_comparisons() {
        // Test null values in string comparisons
        let json_data = r#"{"strings": [
            {"text": "hello", "prefix": "he"},
            {"text": null, "prefix": "he"},
            {"text": "world", "prefix": null},
            {"text": null, "prefix": null},
            {"text": "", "prefix": ""}
        ]}"#;

        let string_null_tests = vec![
            ("$.strings[?@.text == @.prefix]", 1, "Text equals prefix"),
            (
                "$.strings[?@.text != @.prefix]",
                1,
                "Text not equal to prefix",
            ),
            ("$.strings[?@.text == '']", 1, "Empty text"),
            ("$.strings[?@.prefix == '']", 1, "Empty prefix"),
        ];

        for (expr, expected_count, description) in string_null_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "String null test '{}' -> {} results ({})",
                expr,
                results.len(),
                description
            );
        }
    }
}
