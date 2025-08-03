use crate::json_path::{CoreJsonPathEvaluator, JsonPathParser};
use serde_json::{json, Value};

pub fn debug_infinite_loop_patterns() {
    let bookstore_json = json!({
        "store": {
            "book": [
                {
                    "category": "reference",
                    "author": "Nigel Rees",
                    "title": "Sayings of the Century",
                    "price": 8.95
                },
                {
                    "category": "fiction",
                    "author": "Evelyn Waugh", 
                    "title": "Sword of Honour",
                    "price": 12.99
                },
                {
                    "category": "fiction",
                    "author": "Herman Melville",
                    "title": "Moby Dick",
                    "isbn": "0-553-21311-3",
                    "price": 8.99
                },
                {
                    "category": "fiction",
                    "author": "J. R. R. Tolkien",
                    "title": "The Lord of the Rings",
                    "isbn": "0-395-19395-8", 
                    "price": 22.99
                }
            ],
            "bicycle": {
                "color": "red",
                "price": 19.95
            }
        }
    });

    println!("=== Debug infinite loop patterns ===");
    
    // Test the patterns that are timing out
    let failing_patterns = vec![
        "$..book[2]",
        "$.store.bicycle"
    ];
    
    for pattern in failing_patterns {
        println!("\n--- Testing pattern: {} ---", pattern);
        
        // First check if parser can compile the expression
        match JsonPathParser::compile(pattern) {
            Ok(parsed_expr) => {
                println!("✓ Parser successfully compiled: {}", pattern);
                let selectors = parsed_expr.selectors();
                println!("  Selectors: {:?}", selectors);
                
                // Now test the CoreJsonPathEvaluator with a timeout
                match CoreJsonPathEvaluator::new(pattern) {
                    Ok(evaluator) => {
                        println!("✓ CoreJsonPathEvaluator created");
                        
                        // Add manual timeout since we suspect infinite loop
                        let timeout = std::time::Duration::from_millis(100);
                        let start = std::time::Instant::now();
                        
                        match std::panic::catch_unwind(|| {
                            evaluator.evaluate(&bookstore_json)
                        }) {
                            Ok(result) => {
                                let elapsed = start.elapsed();
                                println!("  ✓ Evaluation completed in {:?}", elapsed);
                                
                                match result {
                                    Ok(values) => {
                                        println!("  Results: {} values found", values.len());
                                        for (i, value) in values.iter().enumerate() {
                                            println!("    [{}]: {}", i, value);
                                        }
                                    }
                                    Err(e) => {
                                        println!("  ✗ Evaluation error: {}", e);
                                    }
                                }
                            }
                            Err(_) => {
                                println!("  ✗ Evaluation panicked (likely infinite loop)");
                            }
                        }
                        
                        if start.elapsed() > timeout {
                            println!("  ⚠ Evaluation took longer than expected: {:?}", start.elapsed());
                        }
                    }
                    Err(e) => {
                        println!("✗ CoreJsonPathEvaluator creation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("✗ Parser failed: {}", e);
            }
        }
    }
    
    // Test some known working patterns for comparison
    println!("\n=== Testing known working patterns ===");
    let working_patterns = vec![
        "$",
        "$.store",
        "$.store.book",
        "$.store.book[0]"
    ];
    
    for pattern in working_patterns {
        println!("\n--- Testing working pattern: {} ---", pattern);
        
        match CoreJsonPathEvaluator::new(pattern) {
            Ok(evaluator) => {
                match evaluator.evaluate(&bookstore_json) {
                    Ok(values) => {
                        println!("  ✓ {} values found", values.len());
                    }
                    Err(e) => {
                        println!("  ✗ Error: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("  ✗ Creation failed: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_infinite_loop() {
        debug_infinite_loop_patterns();
    }
}