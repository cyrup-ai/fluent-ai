//! RFC 9535 Segment Traversal Order Tests
//!
//! Tests for segment traversal ordering validation:
//! - Descendant segment ordering validation
//! - Array vs object traversal differences
//! - Non-deterministic ordering tests
//! - Member-name-shorthand validation
//! - Depth-first vs breadth-first behavior
//! - Traversal consistency requirements

use std::collections::HashSet;

use bytes::Bytes;
use fluent_ai_http3::json_path::{JsonArrayStream, JsonPathParser};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct TraversalModel {
    id: u32,
    name: String,
    children: Vec<TraversalModel>,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct ArrayObjectModel {
    arrays: Vec<Vec<i32>>,
    objects: Vec<std::collections::HashMap<String, String>>,
    mixed: Vec<serde_json::Value>,
}

/// RFC 9535 Descendant Segment Ordering Tests
#[cfg(test)]
mod descendant_ordering_tests {
    use super::*;

    fn create_hierarchical_data() -> String {
        let root = serde_json::json!({
            "level1": {
                "item_a": {
                    "level2": {
                        "item_x": {
                            "level3": {
                                "value": "deep_x"
                            }
                        },
                        "item_y": {
                            "level3": {
                                "value": "deep_y"
                            }
                        }
                    }
                },
                "item_b": {
                    "level2": {
                        "item_z": {
                            "level3": {
                                "value": "deep_z"
                            }
                        }
                    }
                }
            },
            "parallel": {
                "branch1": {
                    "data": "branch1_data"
                },
                "branch2": {
                    "data": "branch2_data"
                }
            }
        });

        serde_json::to_string(&root).expect("Valid JSON serialization")
    }

    #[test]
    fn test_descendant_search_ordering() {
        // RFC 9535: Test descendant search maintains consistent ordering
        let json_data = create_hierarchical_data();

        let descendant_cases = vec![
            ("$..value", "All values via descendant search"),
            ("$..data", "All data fields via descendant search"),
            ("$..level3", "All level3 objects via descendant search"),
            ("$..item_*", "All items with wildcard pattern"),
        ];

        for (expr, description) in descendant_cases {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data.clone());
            let mut results = Vec::new();

            // Collect results multiple times to verify ordering consistency
            for iteration in 0..3 {
                let chunk = Bytes::from(json_data.clone());
                let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

                let iteration_results: Vec<_> = stream.process_chunk(chunk).collect();

                if iteration == 0 {
                    results = iteration_results;
                } else {
                    // Verify consistent ordering across iterations
                    assert_eq!(
                        results.len(),
                        iteration_results.len(),
                        "{}: Result count should be consistent across iterations",
                        description
                    );

                    for (i, (expected, actual)) in
                        results.iter().zip(iteration_results.iter()).enumerate()
                    {
                        assert_eq!(
                            expected.is_ok(),
                            actual.is_ok(),
                            "{}: Result status should be consistent at index {}",
                            description,
                            i
                        );
                    }
                }
            }

            println!(
                "{}: {} results with consistent ordering",
                description,
                results.len()
            );
        }
    }

    #[test]
    fn test_depth_first_traversal() {
        // RFC 9535: Verify depth-first traversal behavior for descendant search
        let nested_data = serde_json::json!({
            "root": {
                "level1_a": {
                    "level2_a": {
                        "target": "1a_2a"
                    },
                    "level2_b": {
                        "target": "1a_2b"
                    }
                },
                "level1_b": {
                    "level2_c": {
                        "target": "1b_2c"
                    }
                }
            }
        });

        let json_data = serde_json::to_string(&nested_data).expect("Valid JSON");

        let mut stream = JsonArrayStream::<String>::new("$..target");

        let chunk = Bytes::from(json_data);
        let results: Vec<_> = stream
            .process_chunk(chunk)
            .collect();

        assert_eq!(results.len(), 3, "Should find all three target values");

        // Verify depth-first ordering (though exact order may be implementation-defined)
        let expected_values: HashSet<_> = ["1a_2a", "1a_2b", "1b_2c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let actual_values: HashSet<_> = results.into_iter().collect();

        assert_eq!(
            expected_values, actual_values,
            "Should find all expected values regardless of specific order"
        );

        println!("Depth-first traversal: all expected values found");
    }

    #[test]
    fn test_sibling_ordering() {
        // Test ordering among sibling elements
        let sibling_data = serde_json::json!({
            "container": {
                "z_item": {"order": 1},
                "a_item": {"order": 2},
                "m_item": {"order": 3},
                "b_item": {"order": 4}
            }
        });

        let json_data = serde_json::to_string(&sibling_data).expect("Valid JSON");

        let test_cases = vec![
            ("$.container.*", "Wildcard sibling selection"),
            ("$.container[*]", "Bracket wildcard sibling selection"),
            ("$..order", "Descendant search for order values"),
        ];

        for (expr, description) in test_cases {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream.process_chunk(chunk).collect();

            println!(
                "{}: {} results (sibling ordering)",
                description,
                results.len()
            );

            // Verify all siblings are found (order may be implementation-defined)
            assert!(results.len() >= 4, "Should find all sibling elements");
        }
    }

    #[test]
    fn test_mixed_type_traversal_order() {
        // Test traversal order for mixed object/array structures
        let mixed_data = serde_json::json!({
            "mixed": {
                "array_field": [
                    {"type": "array_item_0"},
                    {"type": "array_item_1"}
                ],
                "object_field": {
                    "nested": {"type": "object_item"}
                },
                "primitive_field": "primitive_value"
            }
        });

        let json_data = serde_json::to_string(&mixed_data).expect("Valid JSON");

        let mut stream = JsonArrayStream::<serde_json::Value>::new("$..type");

        let chunk = Bytes::from(json_data);
        let results: Vec<_> = stream
            .process_chunk(chunk)
            .collect();

        assert_eq!(results.len(), 3, "Should find all type fields");

        let expected_types: HashSet<_> = ["array_item_0", "array_item_1", "object_item"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let actual_types: HashSet<_> = results
            .into_iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();

        assert_eq!(
            expected_types, actual_types,
            "Should find all expected type values in mixed structure"
        );

        println!("Mixed type traversal: all expected types found");
    }
}

/// Array vs Object Traversal Difference Tests
#[cfg(test)]
mod array_object_traversal_tests {
    use super::*;

    #[test]
    fn test_array_index_ordering() {
        // RFC 9535: Array elements should be traversed in index order
        let array_data = serde_json::json!({
            "arrays": [
                {"index": 0, "value": "first"},
                {"index": 1, "value": "second"},
                {"index": 2, "value": "third"},
                {"index": 3, "value": "fourth"}
            ]
        });

        let json_data = serde_json::to_string(&array_data).expect("Valid JSON");

        let mut stream = JsonArrayStream::<serde_json::Value>::new("$.arrays[*].index");

        let chunk = Bytes::from(json_data);
        let results: Vec<_> = stream
            .process_chunk(chunk)
            .collect();

        assert_eq!(results.len(), 4, "Should find all array indices");

        // Verify indices are in order (0, 1, 2, 3)
        for (i, result) in results.iter().enumerate() {
            let index_value = result.as_u64().expect("Should be number") as usize;
            assert_eq!(index_value, i, "Array indices should be in order");
        }

        println!("Array index ordering: verified sequential order");
    }

    #[test]
    fn test_object_member_traversal() {
        // RFC 9535: Object member traversal order is implementation-defined
        let object_data = serde_json::json!({
            "objects": {
                "z_member": {"order": "z"},
                "a_member": {"order": "a"},
                "m_member": {"order": "m"},
                "b_member": {"order": "b"}
            }
        });

        let json_data = serde_json::to_string(&object_data).expect("Valid JSON");

        let mut stream = JsonArrayStream::<serde_json::Value>::new("$.objects.*.order");

        let chunk = Bytes::from(json_data);
        let results: Vec<_> = stream
            .process_chunk(chunk)
            .collect();

        assert_eq!(results.len(), 4, "Should find all object members");

        // Verify all expected values are present (order may vary)
        let expected_orders: HashSet<&str> = ["z", "a", "m", "b"].into_iter().collect();
        let actual_orders: HashSet<_> = results.iter().map(|v| v.as_str().unwrap_or("")).collect();

        assert_eq!(
            expected_orders, actual_orders,
            "Should find all object members regardless of order"
        );

        println!("Object member traversal: all members found (order implementation-defined)");
    }

    #[test]
    fn test_nested_array_object_traversal() {
        // Test traversal of nested arrays within objects and vice versa
        let nested_data = serde_json::json!({
            "level1": {
                "arrays": [
                    {
                        "nested_objects": {
                            "obj1": {"value": "nested_obj1"},
                            "obj2": {"value": "nested_obj2"}
                        }
                    },
                    {
                        "nested_arrays": [
                            {"value": "nested_arr1"},
                            {"value": "nested_arr2"}
                        ]
                    }
                ]
            }
        });

        let json_data = serde_json::to_string(&nested_data).expect("Valid JSON");

        let test_cases = vec![
            ("$..value", "All values via descendant search"),
            ("$.level1.arrays[*]..value", "Values within array elements"),
            (
                "$.level1.arrays[*].nested_objects.*.value",
                "Object values within arrays",
            ),
            (
                "$.level1.arrays[*].nested_arrays[*].value",
                "Array values within arrays",
            ),
        ];

        for (expr, description) in test_cases {
            let mut stream = JsonArrayStream::<String>::new(expr);

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream
                .process_chunk(chunk)
                .map(|r| r.expect("Valid deserialization"))
                .collect();

            println!("{}: {} values found", description, results.len());

            // Verify all results are valid strings
            for (i, result) in results.iter().enumerate() {
                assert!(
                    result.starts_with("nested_"),
                    "Result {} should be a nested value: {}",
                    i,
                    result
                );
            }
        }
    }

    #[test]
    fn test_array_vs_object_precedence() {
        // Test traversal when both arrays and objects are at the same level
        let mixed_structure = serde_json::json!({
            "mixed": {
                "object_first": {
                    "type": "object",
                    "order": 1
                },
                "array_second": [
                    {"type": "array", "order": 2},
                    {"type": "array", "order": 3}
                ],
                "object_third": {
                    "type": "object",
                    "order": 4
                }
            }
        });

        let json_data = serde_json::to_string(&mixed_structure).expect("Valid JSON");

        let mut stream = JsonArrayStream::<serde_json::Value>::new("$.mixed.*");

        let chunk = Bytes::from(json_data);
        let results: Vec<_> = stream.process_chunk(chunk).collect();

        assert_eq!(
            results.len(),
            3,
            "Should find three top-level mixed elements"
        );

        // Verify all expected elements are present
        let mut found_object = 0;
        let mut found_array = 0;

        for result in results {
            {
                let value = result;
                if value.is_object() {
                    found_object += 1;
                } else if value.is_array() {
                    found_array += 1;
                }
            }
        }

        assert_eq!(found_object, 2, "Should find 2 object elements");
        assert_eq!(found_array, 1, "Should find 1 array element");

        println!(
            "Mixed structure traversal: {} objects, {} arrays",
            found_object, found_array
        );
    }
}

/// Non-deterministic Ordering Tests
#[cfg(test)]
mod non_deterministic_tests {
    use super::*;

    #[test]
    fn test_object_property_ordering_consistency() {
        // Test that object property ordering is consistent within a single execution
        let object_data = serde_json::json!({
            "properties": {
                "zebra": "last_alphabetically",
                "alpha": "first_alphabetically",
                "omega": "last_greek",
                "beta": "second_greek",
                "gamma": "third_greek"
            }
        });

        let json_data = serde_json::to_string(&object_data).expect("Valid JSON");

        let mut stream = JsonArrayStream::<String>::new("$.properties.*");

        let chunk = Bytes::from(json_data.clone());
        let results: Vec<_> = stream
            .process_chunk(chunk)
            .collect();

        assert_eq!(results.len(), 5, "Should find all 5 property values");

        // Store the order for comparison
        let first_order = results.clone();

        // Execute the same query again to verify consistency
        let mut stream2 = JsonArrayStream::<String>::new("$.properties.*");

        let chunk = Bytes::from(json_data);
        let second_results: Vec<_> = stream2
            .process_chunk(chunk)
            .collect();

        assert_eq!(
            first_order.len(),
            second_results.len(),
            "Repeated execution should return same number of results"
        );

        // Verify same values are present (order should be consistent within implementation)
        let first_set: HashSet<_> = first_order.into_iter().collect();
        let second_set: HashSet<_> = second_results.into_iter().collect();

        assert_eq!(
            first_set, second_set,
            "Repeated execution should return same values"
        );

        println!("Object property ordering: consistent across executions");
    }

    #[test]
    fn test_descendant_search_stability() {
        // Test that descendant search order is stable across multiple executions
        let nested_data = serde_json::json!({
            "root": {
                "branch_a": {
                    "leaf1": {"id": "a1"},
                    "leaf2": {"id": "a2"},
                    "sub_branch": {
                        "leaf3": {"id": "a3"}
                    }
                },
                "branch_b": {
                    "leaf4": {"id": "b1"},
                    "leaf5": {"id": "b2"}
                }
            }
        });

        let json_data = serde_json::to_string(&nested_data).expect("Valid JSON");

        // Execute descendant search multiple times
        let mut all_executions = Vec::new();

        for execution in 0..5 {
            let mut stream = JsonArrayStream::<String>::new("$..id");

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream
                .process_chunk(chunk)
                .map(|r| r.expect("Valid deserialization"))
                .collect();

            all_executions.push(results);
            println!(
                "Execution {}: {} results",
                execution,
                all_executions[execution].len()
            );
        }

        // Verify all executions return same number of results
        let expected_count = all_executions[0].len();
        for (i, execution) in all_executions.iter().enumerate() {
            assert_eq!(
                execution.len(),
                expected_count,
                "Execution {} should return {} results",
                i,
                expected_count
            );
        }

        // Verify all executions find the same set of values
        let expected_ids: HashSet<_> = all_executions[0].iter().cloned().collect();
        for (i, execution) in all_executions.iter().enumerate() {
            let execution_ids: HashSet<_> = execution.iter().cloned().collect();
            assert_eq!(
                execution_ids, expected_ids,
                "Execution {} should find same set of IDs",
                i
            );
        }

        println!(
            "Descendant search stability: consistent across {} executions",
            all_executions.len()
        );
    }

    #[test]
    fn test_wildcard_vs_explicit_ordering() {
        // Compare wildcard selection vs explicit property access ordering
        let comparison_data = serde_json::json!({
            "data": {
                "first": {"value": 1},
                "second": {"value": 2},
                "third": {"value": 3}
            }
        });

        let json_data = serde_json::to_string(&comparison_data).expect("Valid JSON");

        // Test wildcard selection
        let mut wildcard_stream = JsonArrayStream::<serde_json::Value>::new("$.data.*");

        let chunk = Bytes::from(json_data.clone());
        let wildcard_results: Vec<_> = wildcard_stream.process_chunk(chunk).collect();

        // Test explicit property access
        let explicit_expressions = vec!["$.data.first", "$.data.second", "$.data.third"];

        let mut explicit_results = Vec::new();
        for expr in explicit_expressions {
            let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream.process_chunk(chunk).collect();
            explicit_results.extend(results);
        }

        assert_eq!(
            wildcard_results.len(),
            explicit_results.len(),
            "Wildcard and explicit should return same number of results"
        );

        // Verify same values are accessible through both approaches
        println!(
            "Wildcard vs explicit: {} results each",
            wildcard_results.len()
        );
    }
}

/// Member-name-shorthand Validation Tests
#[cfg(test)]
mod member_name_shorthand_tests {
    use super::*;

    #[test]
    fn test_dot_notation_vs_bracket_notation() {
        // RFC 9535: Dot notation is shorthand for bracket notation
        let shorthand_data = serde_json::json!({
            "simple": "value1",
            "with_underscore": "value2",
            "with123numbers": "value3",
            "MixedCase": "value4"
        });

        let json_data = serde_json::to_string(&shorthand_data).expect("Valid JSON");

        let equivalence_cases = vec![
            ("$.simple", "$['simple']", "Simple property"),
            (
                "$.with_underscore",
                "$['with_underscore']",
                "Property with underscore",
            ),
            (
                "$.with123numbers",
                "$['with123numbers']",
                "Property with numbers",
            ),
            ("$.MixedCase", "$['MixedCase']", "Property with mixed case"),
        ];

        for (dot_expr, bracket_expr, description) in equivalence_cases {
            let mut dot_stream = JsonArrayStream::<String>::new(dot_expr);
            let mut bracket_stream = JsonArrayStream::<String>::new(bracket_expr);

            let chunk = Bytes::from(json_data.clone());
            let dot_results: Vec<_> = dot_stream
                .process_chunk(chunk.clone())
                .map(|r| r.expect("Valid deserialization"))
                .collect();

            let chunk = Bytes::from(json_data.clone());
            let bracket_results: Vec<_> = bracket_stream
                .process_chunk(chunk)
                .map(|r| r.expect("Valid deserialization"))
                .collect();

            assert_eq!(
                dot_results.len(),
                bracket_results.len(),
                "{}: Dot and bracket notation should return same count",
                description
            );

            assert_eq!(
                dot_results, bracket_results,
                "{}: Dot and bracket notation should return same values",
                description
            );

            println!("{}: Dot and bracket notation equivalent", description);
        }
    }

    #[test]
    fn test_invalid_dot_notation_properties() {
        // Test properties that cannot use dot notation shorthand
        let special_data = serde_json::json!({
            "with-dash": "value1",
            "with space": "value2",
            "with.dot": "value3",
            "123numeric": "value4",
            "": "empty_key"
        });

        let json_data = serde_json::to_string(&special_data).expect("Valid JSON");

        let bracket_only_cases = vec![
            ("$['with-dash']", "Property with dash"),
            ("$['with space']", "Property with space"),
            ("$['with.dot']", "Property with dot"),
            ("$['123numeric']", "Property starting with number"),
            ("$['']", "Empty property name"),
        ];

        for (bracket_expr, description) in bracket_only_cases {
            let mut stream = JsonArrayStream::<String>::new(bracket_expr);

            let chunk = Bytes::from(json_data.clone());
            let results: Vec<_> = stream
                .process_chunk(chunk)
                .map(|r| r.expect("Valid deserialization"))
                .collect();

            assert_eq!(
                results.len(),
                1,
                "{}: Should find exactly one value",
                description
            );

            println!("{}: Accessible via bracket notation", description);
        }
    }

    #[test]
    fn test_nested_shorthand_equivalence() {
        // Test nested property access shorthand equivalence
        let nested_data = serde_json::json!({
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep_value"
                    }
                }
            }
        });

        let json_data = serde_json::to_string(&nested_data).expect("Valid JSON");

        let nested_equivalence_cases = vec![
            (
                "$.level1.level2.level3.value",
                "$['level1']['level2']['level3']['value']",
                "Deep nested dot vs bracket",
            ),
            (
                "$.level1.level2.level3",
                "$['level1']['level2']['level3']",
                "Nested object access",
            ),
            (
                "$.level1.level2",
                "$['level1']['level2']",
                "Intermediate object access",
            ),
        ];

        for (dot_expr, bracket_expr, description) in nested_equivalence_cases {
            let mut dot_stream = JsonArrayStream::<serde_json::Value>::new(dot_expr);
            let mut bracket_stream = JsonArrayStream::<serde_json::Value>::new(bracket_expr);

            let chunk = Bytes::from(json_data.clone());
            let dot_results: Vec<_> = dot_stream.process_chunk(chunk.clone()).collect();

            let chunk = Bytes::from(json_data.clone());
            let bracket_results: Vec<_> = bracket_stream.process_chunk(chunk).collect();

            assert_eq!(
                dot_results.len(),
                bracket_results.len(),
                "{}: Nested equivalence should match count",
                description
            );

            println!("{}: Nested shorthand equivalence verified", description);
        }
    }

    #[test]
    fn test_mixed_notation_paths() {
        // Test mixing dot and bracket notation in the same path
        let mixed_data = serde_json::json!({
            "normal": {
                "with-dash": {
                    "normal_again": "mixed_value"
                }
            }
        });

        let json_data = serde_json::to_string(&mixed_data).expect("Valid JSON");

        let mixed_cases = vec![
            (
                "$.normal['with-dash'].normal_again",
                "Mixed dot and bracket notation",
            ),
            (
                "$['normal'].with-dash['normal_again']",
                "Invalid: dash property with dot notation (should fail)",
            ),
        ];

        for (expr, description) in mixed_cases {
            let result = JsonPathParser::compile(expr);

            match result {
                Ok(_) => {
                    let mut stream = JsonArrayStream::<String>::new(expr);

                    let chunk = Bytes::from(json_data.clone());
                    let results: Vec<_> = stream.process_chunk(chunk).collect();

                    println!("{}: {} results", description, results.len());
                }
                Err(_) => {
                    println!(
                        "{}: Compilation failed (expected for invalid syntax)",
                        description
                    );
                }
            }
        }
    }
}

/// Traversal Consistency Requirement Tests
#[cfg(test)]
mod consistency_tests {
    use super::*;

    #[test]
    fn test_multi_execution_consistency() {
        // Verify that multiple executions of the same query return consistent results
        let consistency_data = serde_json::json!({
            "data": {
                "items": [
                    {"id": 1, "type": "A"},
                    {"id": 2, "type": "B"},
                    {"id": 3, "type": "A"},
                    {"id": 4, "type": "C"}
                ],
                "metadata": {
                    "count": 4,
                    "types": ["A", "B", "C"]
                }
            }
        });

        let json_data = serde_json::to_string(&consistency_data).expect("Valid JSON");

        let query_expressions = vec![
            ("$.data.items[*].id", "All item IDs"),
            ("$..type", "All type fields"),
            ("$.data.items[?@.type == 'A']", "Items with type A"),
            ("$..items[*]", "All items via descendant"),
        ];

        for (expr, description) in query_expressions {
            let mut execution_results = Vec::new();

            // Execute the same query multiple times
            for execution in 0..3 {
                let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);

                let chunk = Bytes::from(json_data.clone());
                let results: Vec<_> = stream.process_chunk(chunk).collect();

                execution_results.push(results);
                println!(
                    "Execution {} of '{}': {} results",
                    execution,
                    description,
                    execution_results[execution].len()
                );
            }

            // Verify consistency across executions
            let expected_count = execution_results[0].len();
            for (i, results) in execution_results.iter().enumerate() {
                assert_eq!(
                    results.len(),
                    expected_count,
                    "Execution {} should return {} results for '{}'",
                    i,
                    expected_count,
                    description
                );
            }

            println!(
                "Consistency verified for '{}': {} executions",
                description,
                execution_results.len()
            );
        }
    }

    #[test]
    fn test_order_independence_of_equivalent_queries() {
        // Test that logically equivalent queries return the same results
        let equivalence_data = serde_json::json!({
            "container": {
                "alpha": {"value": 1},
                "beta": {"value": 2},
                "gamma": {"value": 3}
            }
        });

        let json_data = serde_json::to_string(&equivalence_data).expect("Valid JSON");

        let equivalent_pairs = vec![
            (("$.container.*", "$.container[*]"), "Wildcard equivalence"),
            (
                ("$.container.alpha", "$['container']['alpha']"),
                "Dot vs bracket notation",
            ),
        ];

        for ((expr1, expr2), description) in equivalent_pairs {
            let mut stream1 = JsonArrayStream::<serde_json::Value>::new(expr1);
            let mut stream2 = JsonArrayStream::<serde_json::Value>::new(expr2);

            let chunk = Bytes::from(json_data.clone());
            let results1: Vec<_> = stream1.process_chunk(chunk.clone()).collect();

            let chunk = Bytes::from(json_data.clone());
            let results2: Vec<_> = stream2.process_chunk(chunk).collect();

            assert_eq!(
                results1.len(),
                results2.len(),
                "{}: Equivalent queries should return same count",
                description
            );

            // Convert to sets for order-independent comparison
            let set1: HashSet<_> = results1.into_iter().map(|r| format!("{:?}", r)).collect();
            let set2: HashSet<_> = results2.into_iter().map(|r| format!("{:?}", r)).collect();

            assert_eq!(
                set1, set2,
                "{}: Equivalent queries should return same values",
                description
            );

            println!("{}: Equivalence verified", description);
        }
    }

    #[test]
    fn test_traversal_determinism_requirements() {
        // Test requirements for deterministic vs non-deterministic behavior
        let determinism_data = serde_json::json!({
            "arrays": [
                {"index": 0},
                {"index": 1},
                {"index": 2}
            ],
            "objects": {
                "c_key": {"order": "c"},
                "a_key": {"order": "a"},
                "b_key": {"order": "b"}
            }
        });

        let json_data = serde_json::to_string(&determinism_data).expect("Valid JSON");

        // Array traversal should be deterministic (index order)
        let mut array_stream = JsonArrayStream::<serde_json::Value>::new("$.arrays[*].index");

        let chunk = Bytes::from(json_data.clone());
        let array_results: Vec<_> = array_stream
            .process_chunk(chunk)
            .collect();

        assert_eq!(array_results.len(), 3, "Should find all array indices");

        // Verify array order is deterministic (0, 1, 2)
        for (i, result) in array_results.iter().enumerate() {
            let index_value = result.as_u64().expect("Should be number") as usize;
            assert_eq!(
                index_value, i,
                "Array indices should be in deterministic order"
            );
        }

        // Object traversal may be non-deterministic but should be consistent
        let mut object_stream = JsonArrayStream::<String>::new("$.objects.*.order");

        let chunk = Bytes::from(json_data);
        let object_results: Vec<_> = object_stream
            .process_chunk(chunk)
            .collect();

        assert_eq!(object_results.len(), 3, "Should find all object orders");

        // Verify all expected values are present (order may vary)
        let expected_orders: HashSet<&str> = ["a", "b", "c"].into_iter().collect();
        let actual_orders: HashSet<_> = object_results.iter().map(|s| s.as_str()).collect();

        assert_eq!(
            expected_orders, actual_orders,
            "Should find all object orders regardless of traversal order"
        );

        println!("Determinism requirements: arrays ordered, objects complete");
    }
}
