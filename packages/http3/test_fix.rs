// Quick test of the RFC 9535 compliant recursive descent with filter
use fluent_ai_http3::json_path::core_evaluator::CoreJsonPathEvaluator;
use serde_json::json;

fn main() {
    println!("Testing RFC 9535 compliant recursive descent with filter");
    
    let evaluator = CoreJsonPathEvaluator::new("$..[?@.author]")
        .expect("Failed to create evaluator");
    
    let json = json!({
        "store": {
            "book": [
                {"author": "Author 1"},
                {"author": "Author 2"}
            ]
        }
    });
    
    match evaluator.evaluate(&json) {
        Ok(results) => {
            println!("Results count: {}", results.len());
            for (i, result) in results.iter().enumerate() {
                let has_author = result.get("author").is_some();
                println!("Result {}: has_author={}, value={:?}", i + 1, has_author, result);
            }
            
            // Expected: exactly 2 results, both book objects with author
            if results.len() == 2 {
                println!("✅ PASS: Correct number of results");
            } else {
                println!("❌ FAIL: Expected 2 results, got {}", results.len());
            }
            
            let author_results = results.iter()
                .filter(|v| v.get("author").is_some())
                .count();
                
            if author_results == 2 {
                println!("✅ PASS: All results have author property");
            } else {
                println!("❌ FAIL: Expected 2 results with author, got {}", author_results);
            }
        }
        Err(e) => {
            println!("❌ ERROR: {:?}", e);
        }
    }
}