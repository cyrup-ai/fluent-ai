use fluent_ai_http3::json_path::core_evaluator::CoreJsonPathEvaluator;
use serde_json::json;

fn main() {
    println!("=== DEBUGGING RECURSIVE DESCENT DETAILED ===");
    
    let json = json!({
        "store": {
            "book": [
                {"author": "Author 1"},
                {"author": "Author 2"}
            ]
        }
    });
    
    println!("JSON structure:");
    println!("{}", serde_json::to_string_pretty(&json).unwrap());
    
    let evaluator = CoreJsonPathEvaluator::new("$..[?@.author]")
        .expect("Failed to create evaluator");
    
    match evaluator.evaluate(&json) {
        Ok(results) => {
            println!("\nResults found: {}", results.len());
            for (i, result) in results.iter().enumerate() {
                println!("Result {}: {}", i + 1, serde_json::to_string_pretty(result).unwrap());
                
                // Check if this result has an author property
                let has_author = result.get("author").is_some();
                println!("  -> Has 'author' property: {}", has_author);
            }
        }
        Err(e) => {
            println!("Evaluation failed: {:?}", e);
        }
    }
    
    // Test what objects should actually match [?@.author]
    println!("\n=== EXPECTED MATCHES ===");
    let root = &json;
    let store = &json["store"];
    let book1 = &json["store"]["book"][0];
    let book2 = &json["store"]["book"][1];
    
    println!("Root has author: {}", root.get("author").is_some());
    println!("Store has author: {}", store.get("author").is_some());
    println!("Book1 has author: {}", book1.get("author").is_some());
    println!("Book2 has author: {}", book2.get("author").is_some());
}