use serde_json::{json, Value};
use crate::json_path::{CoreJsonPathEvaluator, filter::FilterEvaluator};

fn main() {
    let json = json!({
        "store": {
            "book": [
                {"author": "Author 1"},
                {"author": "Author 2"}
            ]
        }
    });
    
    println!("=== DEBUG: Understanding recursive descent issue ===");
    
    // First, let's see what descendants are collected from root
    let evaluator = CoreJsonPathEvaluator::new("$..").expect("Create evaluator");
    let all_descendants = evaluator.evaluate(&json).expect("Get all descendants");
    
    println!("All descendants from $..: {} total", all_descendants.len());
    for (i, desc) in all_descendants.iter().enumerate() {
        let has_author = desc.get("author").is_some();
        println!("  [{}] has_author={}: {:?}", i, has_author, desc);
    }
    
    println!("\n=== Testing filter on each descendant ===");
    
    // Now test the filter on each descendant
    for (i, desc) in all_descendants.iter().enumerate() {
        let filter_result = FilterEvaluator::evaluate_predicate(desc, "@.author");
        match filter_result {
            Ok(matches) => {
                println!("  [{}] Filter result: {} for: {:?}", i, matches, desc);
            }
            Err(e) => {
                println!("  [{}] Filter error: {} for: {:?}", i, e, desc);
            }
        }
    }
    
    println!("\n=== Testing the full $..[?@.author] expression ===");
    let filter_evaluator = CoreJsonPathEvaluator::new("$..[?@.author]").expect("Create filter evaluator");
    let filter_results = filter_evaluator.evaluate(&json).expect("Evaluate filter");
    
    println!("Filter results: {} total", filter_results.len());
    for (i, result) in filter_results.iter().enumerate() {
        println!("  [{}]: {:?}", i, result);
    }
}