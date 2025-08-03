use serde_json::json;
use fluent_ai_http3::json_path::SimpleJsonPathEvaluator;

fn main() {
    let test_json = json!({
        "data": {
            "x": 42,
            "y": 24  
        }
    });

    // Test 1: Basic property access
    println!("=== Testing basic property access ===");
    let evaluator = SimpleJsonPathEvaluator::new("$.data.x").unwrap();
    let results = evaluator.evaluate(&test_json).unwrap();
    println!("$.data.x -> {} results: {:?}", results.len(), results);

    // Test 2: Test the failing expression
    println!("\n=== Testing failing expression ===");
    let evaluator = SimpleJsonPathEvaluator::new("$.data['x','x','y','x']").unwrap();
    let results = evaluator.evaluate(&test_json).unwrap();
    println!("$.data['x','x','y','x'] -> {} results: {:?}", results.len(), results);

    // Test 3: Test other bracket access
    println!("\n=== Testing bracket access ===");
    let evaluator = SimpleJsonPathEvaluator::new("$.data['x']").unwrap();
    let results = evaluator.evaluate(&test_json).unwrap();
    println!("$.data['x'] -> {} results: {:?}", results.len(), results);

    // Test 4: Test union selector
    println!("\n=== Testing union with indices ===");
    let array_json = json!({
        "items": [10, 20, 30, 40]
    });
    let evaluator = SimpleJsonPathEvaluator::new("$.items[0,1,0,2]").unwrap();
    let results = evaluator.evaluate(&array_json).unwrap();
    println!("$.items[0,1,0,2] -> {} results: {:?}", results.len(), results);
}