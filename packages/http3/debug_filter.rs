use fluent_ai_http3::json_path::core_evaluator::CoreJsonPathEvaluator;
use serde_json::json;

fn main() {
    println!("=== DEBUGGING FILTER EVALUATION ===");
    
    let json = json!({
        "store": {
            "book": [
                {"author": "Author 1"},
                {"author": "Author 2"}
            ]
        }
    });
    
    println!("JSON: {}", serde_json::to_string_pretty(&json).unwrap());
    println!();
    
    // Test the failing pattern
    let expression = "$..[?@.author]";
    println!("Testing: {}", expression);
    println!("Expected: 2 results (the two book objects with author property)");
    println!("Current problem: Getting 3 results including store object");
    println!();
    
    match CoreJsonPathEvaluator::new(expression) {
        Ok(evaluator) => {
            match evaluator.evaluate(&json) {
                Ok(results) => {
                    println!("Results count: {}", results.len());
                    for (i, result) in results.iter().enumerate() {
                        println!("  [{}]: {}", i, result);
                    }
                    println!();
                    
                    if results.len() == 3 {
                        println!("❌ BUG CONFIRMED: Store object without author property is matching [?@.author] filter");
                        println!("Store object: {}", results[0]);
                        println!("This should NOT match because it has no 'author' property");
                    }
                }
                Err(e) => println!("❌ Evaluation error: {}", e),
            }
        }
        Err(e) => println!("❌ Parser error: {}", e),
    }
}