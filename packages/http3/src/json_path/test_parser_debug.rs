use crate::json_path::{JsonPathParser, CoreJsonPathEvaluator};
use serde_json::json;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_debug_for_failing_patterns() {
        println!("\n=== Parser Debug for Failing Patterns ===");
        
        let patterns = vec![
            "$..book[2]",
            "$.store.bicycle",
            "$.store.book[2]", // This should work
            "$" // Root should work
        ];
        
        for pattern in patterns {
            println!("\n--- Testing pattern: {} ---", pattern);
            
            match JsonPathParser::compile(pattern) {
                Ok(parsed_expr) => {
                    println!("✓ Parsed successfully");
                    let selectors = parsed_expr.selectors();
                    println!("  Selectors ({} total):", selectors.len());
                    for (i, selector) in selectors.iter().enumerate() {
                        println!("    [{}]: {:?}", i, selector);
                    }
                    
                    // Now test simple data structure to see if evaluation works
                    let simple_data = json!({
                        "store": {
                            "book": ["a", "b", "c", "d"],
                            "bicycle": {"color": "red"}
                        }
                    });
                    
                    match CoreJsonPathEvaluator::new(pattern) {
                        Ok(evaluator) => {
                            println!("  ✓ CoreJsonPathEvaluator created");
                            
                            // Try evaluation with timeout detection
                            use std::time::{Duration, Instant};
                            let start = Instant::now();
                            let timeout = Duration::from_millis(50);
                            
                            // We can't actually interrupt the evaluation, but we can at least
                            // see if it takes too long and mark it as problematic
                            let result = evaluator.evaluate(&simple_data);
                            let elapsed = start.elapsed();
                            
                            if elapsed > timeout {
                                println!("  ⚠ SLOW: Evaluation took {:?} (expected < {:?})", elapsed, timeout);
                                println!("    This indicates a performance issue or infinite loop");
                            } else {
                                println!("  ✓ Fast evaluation: {:?}", elapsed);
                            }
                            
                            match result {
                                Ok(values) => {
                                    println!("  ✓ Result: {} values", values.len());
                                    for (i, value) in values.iter().enumerate() {
                                        println!("    [{}]: {}", i, value);
                                    }
                                }
                                Err(e) => {
                                    println!("  ✗ Evaluation error: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            println!("  ✗ CoreJsonPathEvaluator creation failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Parser failed: {}", e);
                }
            }
        }
    }
}