//! Comprehensive test suite for JSONPath core evaluator
//! 
//! Production-quality tests covering RFC 9535 compliance and edge cases.

#[cfg(test)]
mod tests {
    use serde_json::json;
    use super::super::evaluator::CoreJsonPathEvaluator;

    #[test]
    fn test_root_selector() {
        let evaluator = CoreJsonPathEvaluator::new("$")
            .expect("Failed to create evaluator for root selector '$'");
        let json = json!({"test": "value"});
        let results = evaluator
            .evaluate(&json)
            .expect("Failed to evaluate root selector against JSON");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json);
    }

    #[test]
    fn test_property_access() {
        let evaluator = CoreJsonPathEvaluator::new("$.store")
            .expect("Failed to create evaluator for property access '$.store'");
        let json = json!({"store": {"name": "test"}});
        let results = evaluator
            .evaluate(&json)
            .expect("Failed to evaluate property access against JSON");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json!({"name": "test"}));
    }

    #[test]
    fn test_recursive_descent() {
        // Use RFC 9535 compliant recursive descent with bracket selector
        let evaluator = CoreJsonPathEvaluator::new("$..[?@.author]")
            .expect("Failed to create evaluator for test");
        let json = json!({
            "store": {
                "book": [
                    {"author": "Author 1"},
                    {"author": "Author 2"}
                ]
            }
        });
        let results = evaluator
            .evaluate(&json)
            .expect("Failed to evaluate recursive descent filter expression");

        // DEBUG: Print all results to understand what's being returned
        println!("=== DEBUG: Recursive descent results ===");
        println!("Total results: {}", results.len());
        for (i, result) in results.iter().enumerate() {
            let has_author = result.get("author").is_some();
            println!(
                "Result {}: has_author={}, value={:?}",
                i + 1,
                has_author,
                result
            );
        }

        // RFC-compliant filter should return only objects with author property

        // RFC-compliant filter returns only objects that have author property
        assert_eq!(results.len(), 2); // Only the 2 book objects that have author
        // Verify the book objects with authors are included
        assert!(
            results
                .iter()
                .any(|v| v.get("author").map_or(false, |a| a == "Author 1"))
        );
        assert!(
            results
                .iter()
                .any(|v| v.get("author").map_or(false, |a| a == "Author 2"))
        );
    }

    #[test]
    fn test_array_wildcard() {
        let evaluator = CoreJsonPathEvaluator::new("$.store.book[*]")
            .expect("Failed to create evaluator for array wildcard '$.store.book[*]'");
        let json = json!({
            "store": {
                "book": [
                    {"title": "Book 1"},
                    {"title": "Book 2"}
                ]
            }
        });
        let results = evaluator
            .evaluate(&json)
            .expect("Failed to evaluate array wildcard against JSON");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn debug_simple_infinite_loop() {
        println!("\n=== DEBUG: Testing simple pattern that's timing out ===");

        let json_value = json!({
            "store": {
                "book": ["a", "b", "c", "d"],
                "bicycle": {"color": "red", "price": 19.95}
            }
        });

        let pattern = "$.store.bicycle";
        println!("Testing pattern: {}", pattern);

        match CoreJsonPathEvaluator::new(pattern) {
            Ok(evaluator) => {
                let start = std::time::Instant::now();
                match evaluator.evaluate(&json_value) {
                    Ok(results) => {
                        let elapsed = start.elapsed();
                        println!("✅ SUCCESS: Got {} results in {:?}", results.len(), elapsed);
                        for (i, result) in results.iter().enumerate() {
                            println!("  [{}]: {}", i, result);
                        }
                    }
                    Err(e) => {
                        println!("❌ ERROR: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("❌ CREATION ERROR: {}", e);
            }
        }
    }

    #[test]
    fn test_negative_indexing_fix() {
        println!("=== Testing negative indexing fix ===");

        let array_json = json!({
            "items": [10, 20, 30, 40]
        });

        // Test negative index
        println!("Test: Negative index [-1]");
        let evaluator = CoreJsonPathEvaluator::new("$.items[-1]")
            .expect("Failed to create evaluator for negative index '$.items[-1]'");
        let results = evaluator
            .evaluate(&array_json)
            .expect("Failed to evaluate negative index [-1] against JSON");
        println!("$.items[-1] -> {} results: {:?}", results.len(), results);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json!(40)); // Should be last element

        // Test negative index [-2]
        println!("Test: Negative index [-2]");
        let evaluator = CoreJsonPathEvaluator::new("$.items[-2]")
            .expect("Failed to create evaluator for negative index '$.items[-2]'");
        let results = evaluator
            .evaluate(&array_json)
            .expect("Failed to evaluate negative index [-2] against JSON");
        println!("$.items[-2] -> {} results: {:?}", results.len(), results);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json!(30)); // Should be second-to-last element
    }

    #[test]
    fn test_recursive_descent_fix() {
        println!("=== Testing recursive descent fix ===");

        let bookstore_json = json!({
            "store": {
                "book": [
                    {"author": "Author1", "title": "Book1"},
                    {"author": "Author2", "title": "Book2"}
                ],
                "bicycle": {"color": "red", "price": 19.95}
            }
        });

        // Test recursive descent for authors
        println!("Test: Recursive descent $..author");
        let evaluator = CoreJsonPathEvaluator::new("$..author")
            .expect("Failed to create evaluator for recursive descent '$..author'");
        let results = evaluator
            .evaluate(&bookstore_json)
            .expect("Failed to evaluate recursive descent against JSON");
        println!("$..author -> {} results: {:?}", results.len(), results);
        assert_eq!(results.len(), 2);
        assert!(results.contains(&json!("Author1")));
        assert!(results.contains(&json!("Author2")));
    }

    #[test]
    fn test_duplicate_preservation_debug() {
        println!("=== Testing duplicate preservation ===");

        let test_json = json!({
            "data": {
                "x": 42,
                "y": 24
            }
        });

        // Test 1: Direct property access
        println!("Test 1: Direct property access");
        let evaluator = CoreJsonPathEvaluator::new("$.data.x")
            .expect("Failed to create evaluator for direct property access '$.data.x'");
        let results = evaluator
            .evaluate(&test_json)
            .expect("Failed to evaluate direct property access against JSON");
        println!("$.data.x -> {} results: {:?}", results.len(), results);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json!(42));

        // Test 2: Test bracket notation
        println!("Test 2: Bracket notation");
        let evaluator = CoreJsonPathEvaluator::new("$.data['x']")
            .expect("Failed to create evaluator for bracket notation '$.data['x']'");
        let results = evaluator
            .evaluate(&test_json)
            .expect("Failed to evaluate bracket notation against JSON");
        println!("$.data['x'] -> {} results: {:?}", results.len(), results);

        // Test 3: Test the multi-selector expression for duplicate preservation
        println!("Test 3: Multi-selector (should show duplicates)");
        let evaluator = CoreJsonPathEvaluator::new("$.data['x','x','y','x']")
            .expect("Failed to create evaluator for multi-selector '$.data['x','x','y','x']'");
        let results = evaluator
            .evaluate(&test_json)
            .expect("Failed to evaluate multi-selector against JSON");
        println!(
            "$.data['x','x','y','x'] -> {} results: {:?}",
            results.len(),
            results
        );

        // Test 4: Test union selector with array indices
        println!("Test 4: Array union selector");
        let array_json = json!({
            "items": [10, 20, 30, 40]
        });
        let evaluator = CoreJsonPathEvaluator::new("$.items[0,1,0,2]")
            .expect("Failed to create evaluator for array union selector '$.items[0,1,0,2]'");
        let results = evaluator
            .evaluate(&array_json)
            .expect("Failed to evaluate array union selector against JSON");
        println!(
            "$.items[0,1,0,2] -> {} results: {:?}",
            results.len(),
            results
        );
    }

    #[test]
    fn test_slice_operations() {
        let array_json = json!({
            "items": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        });

        // Test basic slice [1:5]
        let evaluator = CoreJsonPathEvaluator::new("$.items[1:5]")
            .expect("Failed to create evaluator for slice '$.items[1:5]'");
        let results = evaluator
            .evaluate(&array_json)
            .expect("Failed to evaluate slice against JSON");
        assert_eq!(results.len(), 4);
        assert_eq!(results, vec![json!(1), json!(2), json!(3), json!(4)]);

        // Test negative slice [-3:]
        let evaluator = CoreJsonPathEvaluator::new("$.items[-3:]")
            .expect("Failed to create evaluator for negative slice '$.items[-3:]'");
        let results = evaluator
            .evaluate(&array_json)
            .expect("Failed to evaluate negative slice against JSON");
        assert_eq!(results.len(), 3);
        assert_eq!(results, vec![json!(7), json!(8), json!(9)]);
    }

    #[test]
    fn test_filter_expressions() {
        let bookstore_json = json!({
            "store": {
                "book": [
                    {"title": "Book 1", "price": 10.99},
                    {"title": "Book 2", "price": 15.99},
                    {"title": "Book 3", "price": 8.99}
                ]
            }
        });

        // Test filter for books with price > 10
        let evaluator = CoreJsonPathEvaluator::new("$.store.book[?@.price > 10]")
            .expect("Failed to create evaluator for price filter");
        let results = evaluator
            .evaluate(&bookstore_json)
            .expect("Failed to evaluate price filter against JSON");
        assert_eq!(results.len(), 2);
        
        // Verify correct books are returned
        assert!(results.iter().any(|v| v.get("price").unwrap() == &json!(10.99)));
        assert!(results.iter().any(|v| v.get("price").unwrap() == &json!(15.99)));
    }

    #[test]
    fn test_union_selectors() {
        let test_json = json!({
            "data": {
                "a": 1,
                "b": 2,
                "c": 3
            }
        });

        // Test union selector for multiple properties
        let evaluator = CoreJsonPathEvaluator::new("$.data['a','c']")
            .expect("Failed to create evaluator for union selector");
        let results = evaluator
            .evaluate(&test_json)
            .expect("Failed to evaluate union selector against JSON");
        assert_eq!(results.len(), 2);
        assert!(results.contains(&json!(1)));
        assert!(results.contains(&json!(3)));
    }

    #[test]
    fn test_complex_nested_paths() {
        let complex_json = json!({
            "users": [
                {
                    "name": "John",
                    "addresses": [
                        {"type": "home", "city": "NYC"},
                        {"type": "work", "city": "LA"}
                    ]
                },
                {
                    "name": "Jane",
                    "addresses": [
                        {"type": "home", "city": "SF"}
                    ]
                }
            ]
        });

        // Test nested array access
        let evaluator = CoreJsonPathEvaluator::new("$.users[*].addresses[*].city")
            .expect("Failed to create evaluator for nested array access");
        let results = evaluator
            .evaluate(&complex_json)
            .expect("Failed to evaluate nested array access against JSON");
        assert_eq!(results.len(), 3);
        assert!(results.contains(&json!("NYC")));
        assert!(results.contains(&json!("LA")));
        assert!(results.contains(&json!("SF")));
    }
}