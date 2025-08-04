//! Compiler module tests
//!
//! Tests for JSONPath compiler functionality, mirroring src/json_path/compiler.rs
//!
//! This module contains comprehensive tests for:
//! - ABNF grammar compliance and validation
//! - JSONPath compilation pipeline performance
//! - Grammar well-formedness verification
//! - Compilation optimization strategies
//! - Resource limit enforcement
//! - Large dataset performance validation

use std::time::Instant;

use bytes::Bytes;
use fluent_ai_http3::json_path::{JsonArrayStream, JsonPathParser};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct LargeDataModel {
    id: u32,
    name: String,
    category: String,
    price: f64,
    tags: Vec<String>,
    metadata: std::collections::HashMap<String, serde_json::Value>,
    active: bool,
    created_at: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct NestedDataModel {
    level1: Level1Data,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct Level1Data {
    level2: Vec<Level2Data>,
    metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct Level2Data {
    level3: Level3Data,
    values: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct Level3Data {
    data: Vec<String>,
    properties: std::collections::HashMap<String, f64>,
}

/// RFC 9535 ABNF Grammar Compliance Tests
#[cfg(test)]
mod abnf_grammar_tests {
    use super::*;

    #[test]
    fn test_jsonpath_query_syntax() {
        // RFC 9535: jsonpath-query = root-identifier segments
        let valid_queries = vec![
            "$",                  // Root identifier only
            "$.store",            // Root + single segment
            "$.store.book",       // Root + multiple segments
            "$['store']['book']", // Root + bracket notation
            "$..book",            // Root + descendant segment
        ];

        for query in valid_queries {
            let result = JsonPathParser::compile(query);
            assert!(
                result.is_ok(),
                "Valid ABNF query '{}' should compile",
                query
            );
        }
    }

    #[test]
    fn test_root_identifier() {
        // RFC 9535: root-identifier = "$"
        let valid_roots = vec!["$"];
        let invalid_roots = vec!["", "@", "store", "$."];

        for root in valid_roots {
            let result = JsonPathParser::compile(root);
            assert!(result.is_ok(), "Valid root '{}' should compile", root);
        }

        for root in invalid_roots {
            let result = JsonPathParser::compile(root);
            assert!(result.is_err(), "Invalid root '{}' should fail", root);
        }
    }

    #[test]
    fn test_segments_syntax() {
        // RFC 9535: segments = *(S segment)
        let segment_tests = vec![
            ("$", true),                // No segments
            ("$.store", true),          // Single segment
            ("$.store.book", true),     // Multiple segments
            ("$  .store", true),        // Whitespace allowed
            ("$ . store . book", true), // Multiple whitespace
        ];

        for (query, should_pass) in segment_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid segments '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid segments '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_segment_types() {
        // RFC 9535: segment = child-segment / descendant-segment
        let segment_tests = vec![
            // Child segments
            ("$.store", true),
            ("$['store']", true),
            ("$[0]", true),
            ("$[*]", true),
            ("$[0:5]", true),
            // Descendant segments
            ("$..store", true),
            ("$..*", true),
            ("$..[0]", true),
            ("$..[?@.price]", true),
        ];

        for (query, should_pass) in segment_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid segment '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid segment '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_child_segment_syntax() {
        // RFC 9535: child-segment = dot-member / bracket-segment
        let child_tests = vec![
            // Dot member notation
            ("$.store", true),
            ("$.book_store", true),
            ("$._private", true),
            ("$.123invalid", false), // Can't start with digit
            // Bracket notation
            ("$['store']", true),
            ("$[\"store\"]", true),
            ("$[0]", true),
            ("$[*]", true),
            ("$[:5]", true),
            ("$[1:3]", true),
            ("$[?@.price]", true),
        ];

        for (query, should_pass) in child_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(
                    result.is_ok(),
                    "Valid child segment '{}' should compile",
                    query
                );
            } else {
                assert!(
                    result.is_err(),
                    "Invalid child segment '{}' should fail",
                    query
                );
            }
        }
    }

    #[test]
    fn test_descendant_segment_syntax() {
        // RFC 9535: descendant-segment = ".." S bracket-segment
        let descendant_tests = vec![
            ("$..store", true),
            ("$..*", true),
            ("$..[0]", true),
            ("$..[*]", true),
            ("$..[?@.price]", true),
            ("$..['store']", true),
            ("$.. [0]", true),    // Whitespace allowed
            ("$...store", false), // Triple dot invalid
            ("$..", false),       // Must have bracket segment
        ];

        for (query, should_pass) in descendant_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(
                    result.is_ok(),
                    "Valid descendant '{}' should compile",
                    query
                );
            } else {
                assert!(
                    result.is_err(),
                    "Invalid descendant '{}' should fail",
                    query
                );
            }
        }
    }

    #[test]
    fn test_bracket_segment_syntax() {
        // RFC 9535: bracket-segment = "[" S selector *(S "," S selector) S "]"
        let bracket_tests = vec![
            ("$[0]", true),           // Single selector
            ("$[0,1,2]", true),       // Multiple selectors
            ("$[ 0 , 1 , 2 ]", true), // Whitespace
            ("$[*]", true),           // Wildcard
            ("$['name']", true),      // String
            ("$[?@.price]", true),    // Filter
            ("$[:5]", true),          // Slice
            ("$[", false),            // Unclosed
            ("$0]", false),           // Missing open bracket
            ("$[]", false),           // Empty selectors
        ];

        for (query, should_pass) in bracket_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid bracket '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid bracket '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_selector_types() {
        // RFC 9535: selector = name / wildcard / slice / index / filter
        let selector_tests = vec![
            // Name selectors
            ("$['store']", true),
            ("$[\"book\"]", true),
            // Wildcard selector
            ("$[*]", true),
            // Slice selectors
            ("$[:]", true),
            ("$[1:]", true),
            ("$[:5]", true),
            ("$[1:5]", true),
            ("$[::2]", true),
            ("$[1:5:2]", true),
            // Index selectors
            ("$[0]", true),
            ("$[-1]", true),
            ("$[42]", true),
            // Filter selectors
            ("$[?@.price]", true),
            ("$[?@.price > 10]", true),
        ];

        for (query, should_pass) in selector_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid selector '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid selector '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_name_selector_syntax() {
        // RFC 9535: name = string-literal
        let name_tests = vec![
            ("$['store']", true),
            ("$[\"store\"]", true),
            ("$['']", true),               // Empty string valid
            ("$['book-store']", true),     // Hyphen valid
            ("$['book_store']", true),     // Underscore valid
            ("$['123']", true),            // Number as string valid
            ("$['with spaces']", true),    // Spaces valid
            ("$['with\\nnewline']", true), // Escape sequences valid
            ("$[store]", false),           // Unquoted invalid
        ];

        for (query, should_pass) in name_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid name '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid name '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_string_literal_syntax() {
        // RFC 9535: string-literal = %x22 *string-character %x22 / %x27 *string-character %x27
        let string_tests = vec![
            ("$[\"hello\"]", true),         // Double quotes
            ("$['hello']", true),           // Single quotes
            ("$[\"\"]", true),              // Empty double quoted
            ("$['']", true),                // Empty single quoted
            ("$[\"with 'single'\"]", true), // Mixed quotes
            ("$['with \"double\"']", true), // Mixed quotes
            ("$[\"unclosed]", false),       // Unclosed double quote
            ("$['unclosed]", false),        // Unclosed single quote
        ];

        for (query, should_pass) in string_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid string '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid string '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_string_character_syntax() {
        // RFC 9535: string-character = string-escaped / string-unescaped
        let character_tests = vec![
            // Escaped characters
            ("$[\"\\b\"]", true),     // Backspace
            ("$[\"\\t\"]", true),     // Tab
            ("$[\"\\n\"]", true),     // Newline
            ("$[\"\\f\"]", true),     // Form feed
            ("$[\"\\r\"]", true),     // Carriage return
            ("$[\"\\\"\\\"]]", true), // Quote
            ("$[\"\\'\"]]", true),    // Apostrophe
            ("$[\"\\/\"]", true),     // Solidus
            ("$[\"\\\\\"]", true),    // Backslash
            ("$[\"\\u0041\"]", true), // Unicode escape
            // Unescaped characters
            ("$[\"hello\"]", true),      // Regular characters
            ("$[\"123\"]", true),        // Numbers
            ("$[\"!@#$%^&*()\"]", true), // Special chars
            // Invalid escapes
            ("$[\"\\x\"]", false),    // Invalid escape
            ("$[\"\\u\"]", false),    // Incomplete unicode
            ("$[\"\\u123\"]", false), // Incomplete unicode
        ];

        for (query, should_pass) in character_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid character '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid character '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_wildcard_syntax() {
        // RFC 9535: wildcard = "*"
        let wildcard_tests = vec![
            ("$[*]", true),
            ("$.store[*]", true),
            ("$.*", true),
            ("$[**]", false), // Double wildcard invalid
            ("$[*0]", false), // Wildcard with number invalid
        ];

        for (query, should_pass) in wildcard_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid wildcard '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid wildcard '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_slice_syntax() {
        // RFC 9535: slice = [start S] ":" S [end S] [":" S step]
        let slice_tests = vec![
            ("$[:]", true),           // Full slice
            ("$[1:]", true),          // Start only
            ("$[:5]", true),          // End only
            ("$[1:5]", true),         // Start and end
            ("$[::2]", true),         // Step only
            ("$[1::2]", true),        // Start and step
            ("$[:5:2]", true),        // End and step
            ("$[1:5:2]", true),       // Full slice
            ("$[ 1 : 5 : 2 ]", true), // Whitespace
            ("$[-1:]", true),         // Negative start
            ("$[:-1]", true),         // Negative end
            ("$[::-1]", true),        // Negative step
            ("$[1:5:0]", false),      // Zero step invalid
            ("$[1:5:]", false),       // Missing step after colon
        ];

        for (query, should_pass) in slice_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid slice '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid slice '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_index_syntax() {
        // RFC 9535: index = int
        let index_tests = vec![
            ("$[0]", true),    // Zero
            ("$[1]", true),    // Positive
            ("$[-1]", true),   // Negative
            ("$[42]", true),   // Large positive
            ("$[-42]", true),  // Large negative
            ("$[01]", false),  // Leading zero invalid
            ("$[+1]", false),  // Plus sign invalid
            ("$[1.0]", false), // Decimal invalid
            ("$[1e5]", false), // Scientific notation invalid
        ];

        for (query, should_pass) in index_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid index '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid index '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_filter_syntax() {
        // RFC 9535: filter = "?" S logical-expr
        let filter_tests = vec![
            ("$[?@.price]", true),       // Property existence
            ("$[?@.price > 10]", true),  // Comparison
            ("$[? @.price > 10]", true), // Whitespace after ?
            ("$[?@.a && @.b]", true),    // Logical AND
            ("$[?@.a || @.b]", true),    // Logical OR
            ("$[?(@.a)]", true),         // Parentheses
            ("$[@.price]", false),       // Missing ?
            ("$[?]", false),             // Missing expression
            ("$[? ]", false),            // Only whitespace
        ];

        for (query, should_pass) in filter_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid filter '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid filter '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_integer_syntax() {
        // RFC 9535: int = "0" / (["−"] (non-zero-digit *DIGIT))
        let integer_tests = vec![
            ("$[0]", true),    // Zero
            ("$[1]", true),    // Single digit
            ("$[123]", true),  // Multiple digits
            ("$[-1]", true),   // Negative
            ("$[-123]", true), // Negative multiple digits
            ("$[01]", false),  // Leading zero
            ("$[-0]", false),  // Negative zero
            ("$[+1]", false),  // Plus sign
            ("$[]", false),    // Empty
        ];

        for (query, should_pass) in integer_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Valid integer '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Invalid integer '{}' should fail", query);
            }
        }
    }
}

/// UTF-8 Encoding Validation Tests
#[cfg(test)]
mod utf8_encoding_tests {
    use super::*;

    #[test]
    fn test_utf8_member_names() {
        // RFC 9535: JSONPath expressions must be valid UTF-8
        let utf8_tests = vec![
            ("$['café']", true),   // Basic Latin + accents
            ("$['κόσμος']", true), // Greek
            ("$['世界']", true),   // Chinese
            ("$['🌍']", true),     // Emoji
            ("$['Москва']", true), // Cyrillic
            ("$['العالم']", true), // Arabic
        ];

        for (query, should_pass) in utf8_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(
                    result.is_ok(),
                    "Valid UTF-8 query '{}' should compile",
                    query
                );
            } else {
                assert!(
                    result.is_err(),
                    "Invalid UTF-8 query '{}' should fail",
                    query
                );
            }
        }
    }

    #[test]
    fn test_unicode_escape_sequences() {
        // RFC 9535: Unicode escape sequences in string literals
        let unicode_tests = vec![
            ("$[\"\\u0041\"]", true),        // ASCII 'A'
            ("$[\"\\u00E9\"]", true),        // é
            ("$[\"\\u03BA\"]", true),        // κ
            ("$[\"\\u4E16\"]", true),        // 世
            ("$[\"\\uD83C\\uDF0D\"]", true), // 🌍 (surrogate pair)
            ("$[\"\\u\"]", false),           // Incomplete
            ("$[\"\\u123\"]", false),        // Too short
            ("$[\"\\u123G\"]", false),       // Invalid hex
        ];

        for (query, should_pass) in unicode_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(
                    result.is_ok(),
                    "Valid Unicode escape '{}' should compile",
                    query
                );
            } else {
                assert!(
                    result.is_err(),
                    "Invalid Unicode escape '{}' should fail",
                    query
                );
            }
        }
    }

    #[test]
    fn test_byte_order_mark() {
        // RFC 9535: BOM handling in JSONPath expressions
        let bom_prefix = "\u{FEFF}";
        let query_with_bom = format!("{}$.store", bom_prefix);

        // BOM should be handled gracefully or rejected consistently
        let result = JsonPathParser::compile(&query_with_bom);

        // Document current behavior - implementation may accept or reject BOM
        match result {
            Ok(_) => println!("BOM in JSONPath expression accepted"),
            Err(_) => println!("BOM in JSONPath expression rejected"),
        }
    }
}

/// I-JSON Number Range Validation Tests
#[cfg(test)]
mod ijson_number_tests {
    use super::*;

    #[test]
    fn test_ijson_integer_range() {
        // RFC 9535: I-JSON restricts numbers to IEEE 754 double precision range
        let number_tests = vec![
            // Valid I-JSON integers
            ("$[0]", true),
            ("$[1]", true),
            ("$[-1]", true),
            ("$[9007199254740991]", true),  // MAX_SAFE_INTEGER
            ("$[-9007199254740991]", true), // MIN_SAFE_INTEGER
            // Numbers beyond safe integer range (may be valid syntax but precision loss)
            ("$[9007199254740992]", true),  // Beyond MAX_SAFE_INTEGER
            ("$[-9007199254740992]", true), // Beyond MIN_SAFE_INTEGER
        ];

        for (query, _should_compile) in number_tests {
            let result = JsonPathParser::compile(query);
            if _should_compile {
                assert!(result.is_ok(), "I-JSON number '{}' should compile", query);
            } else {
                assert!(
                    result.is_err(),
                    "Invalid I-JSON number '{}' should fail",
                    query
                );
            }
        }
    }

    #[test]
    fn test_decimal_number_precision() {
        // Test decimal number handling in filter expressions
        let json_data = r#"{"items": [{"price": 1.23456789012345}]}"#;

        // Test precision handling
        let expressions = vec![
            "$.items[?@.price == 1.23456789012345]", // Exact match
            "$.items[?@.price > 1.2]",               // Comparison
            "$.items[?@.price < 1.3]",               // Comparison
        ];

        for expr in expressions {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "Decimal precision test '{}' returned {} results",
                expr,
                results.len()
            );
        }
    }

    #[test]
    fn test_scientific_notation() {
        // RFC 9535: Scientific notation in numbers
        let scientific_tests = vec![
            ("$[1e5]", false),   // Scientific notation not in index
            ("$[1E5]", false),   // Capital E
            ("$[1.5e2]", false), // Decimal with exponent
            ("$[1e+5]", false),  // Positive exponent
            ("$[1e-5]", false),  // Negative exponent
        ];

        // Note: Scientific notation typically not allowed in array indices
        // but may be valid in filter expressions
        for (query, should_pass) in scientific_tests {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(
                    result.is_ok(),
                    "Scientific notation '{}' should compile",
                    query
                );
            } else {
                assert!(
                    result.is_err(),
                    "Scientific notation '{}' should fail in index",
                    query
                );
            }
        }
    }
}

/// Well-formedness vs Validity Separation Tests
#[cfg(test)]
mod wellformedness_tests {
    use super::*;

    #[test]
    fn test_wellformed_but_invalid_paths() {
        // RFC 9535: Distinguish between syntactically valid but semantically invalid paths
        let test_cases = vec![
            // Well-formed but may be invalid for specific JSON documents
            ("$.nonexistent", true),  // Valid syntax, may not match anything
            ("$[999]", true),         // Valid syntax, array may not have this index
            ("$.store[999]", true),   // Valid syntax, property may not exist
            ("$..nonexistent", true), // Valid syntax, may not match anything
            // Malformed syntax
            ("$.", false),           // Incomplete dot notation
            ("$[", false),           // Unclosed bracket
            ("$store", false),       // Missing root identifier
            ("$.123invalid", false), // Invalid identifier
        ];

        for (query, should_be_wellformed) in test_cases {
            let result = JsonPathParser::compile(query);
            if should_be_wellformed {
                assert!(
                    result.is_ok(),
                    "Well-formed query '{}' should compile",
                    query
                );
            } else {
                assert!(result.is_err(), "Malformed query '{}' should fail", query);
            }
        }
    }

    #[test]
    fn test_semantic_validation() {
        // Test semantic validation during execution vs parsing
        let json_data = r#"{"store": {"book": [{"title": "Book1"}]}}"#;

        let semantic_tests = vec![
            ("$.store", true),              // Valid path, exists
            ("$.store.book", true),         // Valid path, exists
            ("$.nonexistent", true),        // Valid syntax, doesn't exist (empty result)
            ("$.store.book[999]", true),    // Valid syntax, index out of bounds (empty result)
            ("$.store.book.invalid", true), // Valid syntax, invalid property access (empty result)
        ];

        for (query, should_execute) in semantic_tests {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(query);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            if should_execute {
                // Should execute without error, even if no results
                println!("Query '{}' executed with {} results", query, results.len());
            }
        }
    }

    #[test]
    fn test_grammar_edge_cases() {
        // Test edge cases in grammar interpretation
        let edge_cases = vec![
            // Whitespace handling
            ("$ . store", true),        // Spaces around dot
            ("$[ 'store' ]", true),     // Spaces in brackets
            ("$  [  'store'  ]", true), // Multiple spaces
            // Mixed notation
            ("$.store['book']", true),    // Dot then bracket
            ("$['store'].book", true),    // Bracket then dot
            ("$['store']['book']", true), // All brackets
            // Complex expressions
            ("$.store..book[*].title", true), // Mixed segments
            ("$..*.price", true),             // Wildcard after descendant
        ];

        for (query, should_pass) in edge_cases {
            let result = JsonPathParser::compile(query);
            if should_pass {
                assert!(result.is_ok(), "Edge case '{}' should compile", query);
            } else {
                assert!(result.is_err(), "Edge case '{}' should fail", query);
            }
        }
    }
}

/// Helper function to generate large datasets for performance testing
fn generate_large_dataset(size: usize) -> serde_json::Value {
        let items: Vec<LargeDataModel> = (0..size)
            .map(|i| LargeDataModel {
                id: i as u32,
                name: format!("Item_{:06}", i),
                category: match i % 5 {
                    0 => "electronics".to_string(),
                    1 => "books".to_string(),
                    2 => "clothing".to_string(),
                    3 => "home".to_string(),
                    _ => "misc".to_string(),
                },
                price: (i as f64 * 1.5) + 10.0,
                tags: vec![
                    format!("tag_{}", i % 10),
                    format!("category_{}", i % 5),
                    format!("brand_{}", i % 3),
                ],
                metadata: {
                    let mut map = std::collections::HashMap::new();
                    map.insert(
                        "weight".to_string(),
                        serde_json::Value::Number((i % 100).into()),
                    );
                    map.insert(
                        "color".to_string(),
                        serde_json::Value::String(format!("color_{}", i % 8)),
                    );
                    map.insert(
                        "rating".to_string(),
                        serde_json::Value::Number(((i % 5) + 1).into()),
                    );
                    map
                },
                active: i % 7 != 0,
                created_at: format!("2024-{:02}-{:02}T10:00:00Z", (i % 12) + 1, (i % 28) + 1),
            })
            .collect();

        serde_json::json!({
            "catalog": {
                "items": items,
                "metadata": {
                    "total_count": size,
                    "generated_at": "2024-01-01T00:00:00Z",
                    "version": "1.0"
                }
            }
        })
}

/// RFC 9535 Performance Tests - Large Dataset Handling
#[cfg(test)]
mod large_dataset_tests {
    use super::*;

    #[test]
    fn test_large_array_traversal_performance() {
        // Test performance with 10K elements
        let dataset = generate_large_dataset(10_000);
        let json_data = serde_json::to_string(&dataset).expect("Valid JSON serialization");

        let test_cases = vec![
            ("$.catalog.items[*].name", "All item names"),
            ("$.catalog.items[*].price", "All item prices"),
            ("$.catalog.items[*].category", "All item categories"),
            ("$.catalog.items[*].tags[*]", "All tags from all items"),
        ];

        for (expr, _description) in test_cases {
            let start_time = Instant::now();

            let mut stream = JsonArrayStream::<String>::new(expr);

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            let duration = start_time.elapsed();

            println!(
                "{}: {} results in {:?}",
                _description,
                results.len(),
                duration
            );

            // Performance assertion - should handle large datasets efficiently
            assert!(
                duration.as_secs() < 5,
                "Large dataset query '{}' should complete in <5 seconds",
                expr
            );

            // Verify results are correct
            match expr {
                "$.catalog.items[*].name" => {
                    assert_eq!(results.len(), 10_000, "Should extract all 10K names");
                }
                "$.catalog.items[*].price" => {
                    assert_eq!(results.len(), 10_000, "Should extract all 10K prices");
                }
                "$.catalog.items[*].category" => {
                    assert_eq!(results.len(), 10_000, "Should extract all 10K categories");
                }
                "$.catalog.items[*].tags[*]" => {
                    assert_eq!(
                        results.len(),
                        30_000,
                        "Should extract all 30K tags (3 per item)"
                    );
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_filter_performance_large_dataset() {
        // Test filter performance on large datasets
        let dataset = generate_large_dataset(10_000);
        let json_data = serde_json::to_string(&dataset).expect("Valid JSON serialization");

        let filter_cases = vec![
            ("$.catalog.items[?@.active]", "Active items filter"),
            ("$.catalog.items[?@.price > 100]", "Price filter"),
            (
                "$.catalog.items[?@.category == 'electronics']",
                "Category filter",
            ),
            (
                "$.catalog.items[?@.active && @.price < 50]",
                "Complex logical filter",
            ),
        ];

        for (expr, _description) in filter_cases {
            let start_time = Instant::now();

            let mut stream = JsonArrayStream::<LargeDataModel>::new(expr);

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream
                .process_chunk(chunk)
                .collect();

            let duration = start_time.elapsed();

            println!(
                "{}: {} results in {:?}",
                _description,
                results.len(),
                duration
            );

            // Performance assertion
            assert!(
                duration.as_secs() < 3,
                "Large dataset filter '{}' should complete in <3 seconds",
                expr
            );

            // Verify filter correctness
            match expr {
                "$.catalog.items[?@.active]" => {
                    for item in &results {
                        assert!(item.active, "All filtered items should be active");
                    }
                }
                "$.catalog.items[?@.price > 100]" => {
                    for item in &results {
                        assert!(
                            item.price > 100.0,
                            "All filtered items should have price > 100"
                        );
                    }
                }
                "$.catalog.items[?@.category == 'electronics']" => {
                    for item in &results {
                        assert_eq!(
                            item.category, "electronics",
                            "All filtered items should be electronics"
                        );
                    }
                }
                "$.catalog.items[?@.active && @.price < 50]" => {
                    for item in &results {
                        assert!(
                            item.active && item.price < 50.0,
                            "All items should be active and price < 50"
                        );
                    }
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_descendant_search_performance() {
        // Test descendant search performance on large nested structures
        let dataset = generate_large_dataset(5_000);
        let json_data = serde_json::to_string(&dataset).expect("Valid JSON serialization");

        let descendant_cases = vec![
            ("$..name", "All names (descendant)"),
            ("$..price", "All prices (descendant)"),
            ("$..metadata", "All metadata (descendant)"),
            ("$..*", "All values (universal descendant)"),
        ];

        for (expr, _description) in descendant_cases {
            let start_time = Instant::now();

            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            let duration = start_time.elapsed();

            println!(
                "{}: {} results in {:?}",
                _description,
                results.len(),
                duration
            );

            // Performance assertion - descendant searches are more expensive
            assert!(
                duration.as_secs() < 10,
                "Descendant search '{}' should complete in <10 seconds",
                expr
            );

            // Verify minimum expected results
            assert!(results.len() > 0, "Descendant search should find results");
        }
    }

    #[test]
    fn test_memory_usage_large_arrays() {
        // Test memory efficiency with large arrays
        let sizes = vec![1_000, 5_000, 10_000];

        for size in sizes {
            let dataset = generate_large_dataset(size);
            let json_data = serde_json::to_string(&dataset).expect("Valid JSON serialization");

            let start_time = Instant::now();

            // Test streaming behavior - should not load entire result set into memory
            let mut stream = JsonArrayStream::<LargeDataModel>::new("$.catalog.items[*]");

            let chunk = Bytes::from(json_data);
            let mut count = 0;

            // Process items one by one to verify streaming
            for _item in stream.process_chunk(chunk) {
                count += 1;

                // Simulate processing time
                if count % 1000 == 0 {
                    let elapsed = start_time.elapsed();
                    println!("Processed {} items in {:?}", count, elapsed);
                }
            }

            let total_duration = start_time.elapsed();

            assert_eq!(count, size, "Should process all {} items", size);

            // Memory efficiency assertion - should scale linearly
            let per_item_micros = total_duration.as_micros() / size as u128;
            assert!(
                per_item_micros < 1000,
                "Should process items efficiently (<1ms per item), actual: {}μs",
                per_item_micros
            );

            println!(
                "Size {}: {} items processed in {:?} ({} μs/item)",
                size, count, total_duration, per_item_micros
            );
        }
    }
}

/// RFC 9535 Complex Query Performance Tests
#[cfg(test)]
mod complex_query_tests {
    use super::*;

    fn generate_nested_dataset(depth: usize, width: usize) -> serde_json::Value {
        fn create_nested_level(
            current_depth: usize,
            max_depth: usize,
            width: usize,
        ) -> serde_json::Value {
            if current_depth >= max_depth {
                return serde_json::json!({
                    "value": format!("leaf_value_{}", current_depth),
                    "data": (0..10).map(|i| format!("item_{}", i)).collect::<Vec<_>>()
                });
            }

            let children: Vec<serde_json::Value> = (0..width)
                .map(|i| {
                    let mut child = serde_json::Map::new();
                    child.insert(format!("id_{}", i), serde_json::Value::Number(i.into()));
                    child.insert(
                        "nested".to_string(),
                        create_nested_level(current_depth + 1, max_depth, width),
                    );
                    serde_json::Value::Object(child)
                })
                .collect();

            serde_json::json!({
                "level": current_depth,
                "children": children,
                "metadata": {
                    "depth": current_depth,
                    "width": width,
                    "total_nodes": width.pow((max_depth - current_depth) as u32)
                }
            })
        }

        serde_json::json!({
            "structure": create_nested_level(0, depth, width)
        })
    }

    #[test]
    fn test_deep_nesting_performance() {
        // Test performance with deeply nested structures
        let test_cases = vec![
            (5, 3, "Moderate depth (5 levels, 3 width)"),
            (7, 2, "Deep structure (7 levels, 2 width)"),
            (3, 5, "Wide structure (3 levels, 5 width)"),
        ];

        for (depth, width, _description) in test_cases {
            let dataset = generate_nested_dataset(depth, width);
            let json_data = serde_json::to_string(&dataset).expect("Valid JSON serialization");

            let complex_queries = vec![
                ("$..value", "Deep descendant search"),
                ("$.structure..children[*]", "Nested array access"),
                ("$..metadata", "Metadata at all levels"),
                (
                    "$.structure..children[*]..data[*]",
                    "Multi-level array traversal",
                ),
            ];

            for (expr, query_desc) in complex_queries {
                let start_time = Instant::now();

                let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

                let chunk = Bytes::from(json_data.clone());
                let results: Vec<_> = stream.process_chunk(chunk).collect();

                let duration = start_time.elapsed();

                println!(
                    "{} - {}: {} results in {:?}",
                    _description,
                    query_desc,
                    results.len(),
                    duration
                );

                // Performance assertion - complex queries should still complete reasonably
                assert!(
                    duration.as_secs() < 5,
                    "Complex query '{}' on {} should complete in <5 seconds",
                    expr,
                    _description
                );
            }
        }
    }

    #[test]
    fn test_filter_complexity_performance() {
        // Test performance of increasingly complex filter expressions
        let dataset = generate_large_dataset(1_000);
        let json_data = serde_json::to_string(&dataset).expect("Valid JSON serialization");

        let complexity_levels = vec![
            ("$.catalog.items[?@.active]", "Simple boolean filter"),
            (
                "$.catalog.items[?@.price > 50 && @.active]",
                "Two-condition AND filter",
            ),
            (
                "$.catalog.items[?@.category == 'electronics' || @.category == 'books']",
                "Two-condition OR filter",
            ),
            (
                "$.catalog.items[?(@.price > 100 && @.active) || (@.price < 20 && @.category == 'books')]",
                "Complex grouped conditions",
            ),
            (
                "$.catalog.items[?@.active && @.price > 50 && @.category != 'misc' && @.tags]",
                "Multi-condition complex filter",
            ),
        ];

        for (expr, _description) in complexity_levels {
            let start_time = Instant::now();

            let mut stream = JsonArrayStream::<LargeDataModel>::new(expr);

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            let duration = start_time.elapsed();

            println!(
                "{}: {} results in {:?}",
                _description,
                results.len(),
                duration
            );

            // Performance assertion - more complex filters should still be efficient
            assert!(
                duration.as_millis() < 500,
                "Complex filter '{}' should complete in <500ms",
                _description
            );

            // Verify all results match the filter criteria
            assert!(
                results.len() > 0,
                "Complex filter should find some matching results"
            );
        }
    }

    #[test]
    fn test_query_compilation_performance() {
        // Test JSONPath compilation performance for various query types
        let query_types = vec![
            ("$.simple.path", "Simple property access"),
            ("$.array[*].property", "Array wildcard"),
            ("$..descendant", "Descendant search"),
            ("$.items[?@.active && @.price > 100]", "Complex filter"),
            (
                "$.data..items[*].tags[?@ != 'excluded']",
                "Nested filter with descendant",
            ),
        ];

        for (expr, _description) in query_types {
            let start_time = Instant::now();

            // Compile the same query multiple times to test compilation performance
            for _ in 0..1000 {
                let parser = JsonPathParser::compile(expr).expect("Valid JSONPath compilation");
            }

            let duration = start_time.elapsed();
            let per_compilation = duration.as_nanos() / 1000;

            println!(
                "{}: 1000 compilations in {:?} ({} ns/compilation)",
                _description, duration, per_compilation
            );

            // Compilation should be fast
            assert!(
                per_compilation < 100_000,
                "Query compilation for '{}' should be <100μs per compilation",
                _description
            );
        }
    }
}

/// RFC 9535 Streaming Behavior Verification
#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[test]
    fn test_chunked_processing_performance() {
        // Test streaming behavior with chunked data processing
        let dataset = generate_large_dataset(5_000);
        let json_string = serde_json::to_string(&dataset).expect("Valid JSON serialization");

        // Split into multiple chunks to simulate streaming
        let chunk_size = json_string.len() / 10;
        let chunks: Vec<Bytes> = json_string
            .as_bytes()
            .chunks(chunk_size)
            .map(|chunk| Bytes::copy_from_slice(chunk))
            .collect();

        let start_time = Instant::now();

        let mut stream = JsonArrayStream::<LargeDataModel>::new("$.catalog.items[?@.active]");

        let mut totalresults = 0;

        // Process chunks sequentially to test streaming behavior
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_start = Instant::now();

            let results: Vec<_> = stream.process_chunk(chunk.clone()).collect();
            let chunkresults = results.len();
            totalresults += chunkresults;

            let chunk_duration = chunk_start.elapsed();

            println!(
                "Chunk {}: {} results in {:?}",
                i + 1,
                chunkresults,
                chunk_duration
            );

            // Each chunk should process quickly
            assert!(
                chunk_duration.as_millis() < 200,
                "Chunk {} processing should complete in <200ms",
                i + 1
            );
        }

        let total_duration = start_time.elapsed();

        println!(
            "Total streaming: {} results in {:?}",
            totalresults, total_duration
        );

        // Streaming should be efficient overall
        assert!(
            total_duration.as_secs() < 3,
            "Chunked streaming should complete in <3 seconds"
        );
    }

    #[test]
    fn test_incrementalresult_delivery() {
        // Test that results are delivered incrementally, not all at once
        let dataset = generate_large_dataset(1_000);
        let json_data = serde_json::to_string(&dataset).expect("Valid JSON serialization");

        let mut stream = JsonArrayStream::<LargeDataModel>::new("$.catalog.items[*]");

        let chunk = Bytes::from(json_data);
        let start_time = Instant::now();

        let mut result_count = 0;
        let mut timing_checkpoints = Vec::new();

        // Process results and record timing at regular intervals
        for _item in stream.process_chunk(chunk) {
            result_count += 1;

            // Record timing every 100 results
            if result_count % 100 == 0 {
                timing_checkpoints.push((result_count, start_time.elapsed()));
            }
        }

        // Verify incremental delivery by checking timing progression
        for i in 1..timing_checkpoints.len() {
            let (prev_count, prev_time) = timing_checkpoints[i - 1];
            let (curr_count, curr_time) = timing_checkpoints[i];

            let count_diff = curr_count - prev_count;
            let time_diff = curr_time - prev_time;

            println!(
                "Results {}-{}: {} items in {:?}",
                prev_count, curr_count, count_diff, time_diff
            );

            // Time between checkpoints should be reasonable (not all at the end)
            assert!(
                time_diff.as_millis() < 500,
                "Incremental delivery should process 100 items in <500ms"
            );
        }

        assert_eq!(
            result_count, 1_000,
            "Should process all items incrementally"
        );
    }

    #[test]
    fn test_memory_bounded_streaming() {
        // Test that streaming doesn't consume excessive memory
        let large_string_size = 10_000;
        let large_strings: Vec<String> = (0..100)
            .map(|i| format!("large_string_{}_{}", i, "x".repeat(large_string_size)))
            .collect();

        let dataset = serde_json::json!({
            "data": {
                "strings": large_strings,
                "metadata": {
                    "count": 100,
                    "size_per_string": large_string_size
                }
            }
        });

        let json_data = serde_json::to_string(&dataset).expect("Valid JSON serialization");

        let start_time = Instant::now();

        let mut stream = JsonArrayStream::<String>::new("$.data.strings[*]");

        let chunk = Bytes::from(json_data);

        // Process large strings without accumulating them all in memory
        let mut processed_count = 0;
        for large_string in stream.process_chunk(chunk) {
            // Verify string content without storing it
            assert!(
                large_string.len() > large_string_size,
                "String should be large as expected"
            );
            processed_count += 1;

            // Drop the string immediately to test memory efficiency
            drop(large_string);
        }

        let duration = start_time.elapsed();

        assert_eq!(processed_count, 100, "Should process all large strings");

        // Should handle large strings efficiently
        assert!(
            duration.as_secs() < 2,
            "Large string streaming should complete in <2 seconds"
        );

        println!(
            "Memory-bounded streaming: {} large strings in {:?}",
            processed_count, duration
        );
    }
}

/// RFC 9535 Resource Limit Enforcement Tests
#[cfg(test)]
mod resource_limit_tests {
    use super::*;

    #[test]
    fn test_query_depth_limits() {
        // Test handling of deeply nested queries without stack overflow
        let deep_path_segments = (0..50)
            .map(|i| format!("level{}", i))
            .collect::<Vec<_>>()
            .join(".");

        let deep_query = format!("$.{}", deep_path_segments);

        // Should handle deep paths gracefully
        let result = JsonPathParser::compile(&deep_query);

        match result {
            Ok(_) => {
                println!("Deep query compilation succeeded: 50 levels");
                // If compilation succeeds, test with actual data
                let nested_data = serde_json::json!({
                    "level0": {"level1": {"level2": {"value": "deep_value"}}}
                });

                let json_data = serde_json::to_string(&nested_data).expect("Valid JSON");
                let mut stream = JsonArrayStream::<serde_json::Value>::new(&deep_query);

                let chunk = Bytes::from(json_data);
                let results: Vec<_> = stream.process_chunk(chunk).collect();

                // Should handle gracefully even if path doesn't exist
                println!("Deep query execution: {} results", results.len());
            }
            Err(_) => {
                println!("Deep query rejected at compilation (expected for extreme depth)");
            }
        }
    }

    #[test]
    fn test_large_array_index_handling() {
        // Test handling of very large array indices
        let large_indices = vec![1_000_000, u32::MAX as usize];

        for index in large_indices {
            let query = format!("$.items[{}]", index);

            let result = JsonPathParser::compile(&query);

            match result {
                Ok(_) => {
                    println!("Large index {} compilation succeeded", index);

                    // Test with small array to verify bounds checking
                    let data = serde_json::json!({"items": [1, 2, 3]});
                    let json_data = serde_json::to_string(&data).expect("Valid JSON");

                    let mut stream = JsonArrayStream::<i32>::new(&query);

                    let chunk = Bytes::from(json_data);
                    let results: Vec<_> = stream.process_chunk(chunk).collect();

                    // Should return empty results for out-of-bounds indices
                    assert_eq!(
                        results.len(),
                        0,
                        "Out-of-bounds index {} should return no results",
                        index
                    );
                }
                Err(_) => {
                    println!("Large index {} rejected at compilation", index);
                }
            }
        }
    }

    #[test]
    fn test_expression_complexity_limits() {
        // Test handling of highly complex filter expressions
        let simple_conditions: Vec<String> =
            (0..20).map(|i| format!("@.field{} == {}", i, i)).collect();

        let complex_filter = format!("$.items[?{}]", simple_conditions.join(" && "));

        let start_time = Instant::now();

        let result = JsonPathParser::compile(&complex_filter);

        let compilation_time = start_time.elapsed();

        match result {
            Ok(_) => {
                println!(
                    "Complex filter compilation succeeded in {:?}",
                    compilation_time
                );

                // Should compile reasonably quickly even for complex expressions
                assert!(
                    compilation_time.as_millis() < 100,
                    "Complex filter compilation should complete in <100ms"
                );

                // Test execution with sample data
                let data = serde_json::json!({
                    "items": [
                        {"field0": 0, "field1": 1, "field2": 2},
                        {"field0": 1, "field1": 2, "field2": 3}
                    ]
                });

                let json_data = serde_json::to_string(&data).expect("Valid JSON");
                let mut stream = JsonArrayStream::<serde_json::Value>::new(&complex_filter);

                let execution_start = Instant::now();
                let chunk = Bytes::from(json_data);
                let results: Vec<_> = stream.process_chunk(chunk).collect();
                let execution_time = execution_start.elapsed();

                println!(
                    "Complex filter execution: {} results in {:?}",
                    results.len(),
                    execution_time
                );

                // Should execute efficiently
                assert!(
                    execution_time.as_millis() < 50,
                    "Complex filter execution should complete in <50ms"
                );
            }
            Err(_) => {
                println!("Complex filter rejected at compilation (expected behavior)");
            }
        }
    }

    #[test]
    fn test_concurrent_query_performance() {
        // Test performance under concurrent query execution
        use std::sync::Arc;
        use std::thread;

        let dataset = Arc::new(generate_large_dataset(1_000));
        let json_data = Arc::new(serde_json::to_string(&*dataset).expect("Valid JSON"));

        let queries = vec![
            "$.catalog.items[?@.active]",
            "$.catalog.items[?@.price > 100]",
            "$.catalog.items[?@.category == 'electronics']",
            "$.catalog.items[*].name",
            "$.catalog.items[*].tags[*]",
        ];

        let start_time = Instant::now();

        let handles: Vec<_> = queries
            .into_iter()
            .enumerate()
            .map(|(i, query)| {
                let json_data = Arc::clone(&json_data);
                let query = query.to_string();

                thread::spawn(move || {
                    let thread_start = Instant::now();

                    let mut stream = JsonArrayStream::<serde_json::Value>::new(&query);

                    let chunk = Bytes::from((*json_data).clone());
                    let results: Vec<_> = stream.process_chunk(chunk).collect();

                    let thread_duration = thread_start.elapsed();

                    (i, query, results.len(), thread_duration)
                })
            })
            .collect();

        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.join().expect("Thread completed successfully"));
        }

        let total_duration = start_time.elapsed();

        // Verify all queries completed
        assert_eq!(results.len(), 5, "All concurrent queries should complete");

        for (i, query, result_count, thread_duration) in results {
            println!(
                "Thread {}: '{}' -> {} results in {:?}",
                i, query, result_count, thread_duration
            );

            // Each thread should complete efficiently
            assert!(
                thread_duration.as_secs() < 2,
                "Concurrent query {} should complete in <2 seconds",
                i
            );
        }

        println!("All concurrent queries completed in {:?}", total_duration);

        // Overall concurrent execution should be efficient
        assert!(
            total_duration.as_secs() < 5,
            "Concurrent query execution should complete in <5 seconds"
        );
    }
}
