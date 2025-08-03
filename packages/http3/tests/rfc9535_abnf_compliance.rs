//! RFC 9535 ABNF Grammar Compliance Tests
//!
//! Tests the complete ABNF grammar specification for JSONPath expressions
//! Validates UTF-8 encoding, I-JSON number ranges, and grammar well-formedness

use bytes::Bytes;
use fluent_ai_http3::json_path::{JsonArrayStream, JsonPathParser};

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
            ("$[\"\\\"\"]", true),    // Quote
            ("$[\"\\'\"]", true),     // Apostrophe
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
        // RFC 9535: int = "0" / (["-"] (non-zero-digit *DIGIT))
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

        // RFC 9535: BOM should be rejected in JSONPath expressions (not valid JSON)
        assert!(
            result.is_err(),
            "RFC 9535: BOM in JSONPath expression should be rejected"
        );
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

        for (query, should_compile) in number_tests {
            let result = JsonPathParser::compile(query);
            if should_compile {
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

        // RFC 9535: Test high precision decimal handling
        let test_cases = vec![
            ("$.items[?@.price == 1.23456789012345]", 1), // Exact match should work
            ("$.items[?@.price > 1.2]", 1),               // Greater than comparison
            ("$.items[?@.price < 1.3]", 1),               // Less than comparison
        ];

        for (expr, expected_count) in test_cases {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data);
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            assert_eq!(
                results.len(),
                expected_count,
                "RFC 9535: Decimal precision test '{}' should return {} results",
                expr,
                expected_count
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
                // Valid queries should execute without panicking - test passes if we get here
                // (The fact that we got results without error is the test)
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
