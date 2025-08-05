use std::time::Instant;

fn main() {
    println!("Testing JSONPath pattern: $.text[?match(@, 'a+')]");
    
    let json_data = r#"{"text": "aaaaaaaaaa"}"#;
    println!("Input JSON: {}", json_data);
    
    let json_value: serde_json::Value = serde_json::from_str(json_data).unwrap();
    println!("Parsed JSON: {:?}", json_value);
    
    // Test the specific pattern from the failing test
    let pattern = "$.text[?match(@, 'a+')]";
    
    let start = Instant::now();
    
    // Try to compile the pattern
    match fluent_ai_http3::json_path::JsonPathParser::compile(pattern) {
        Ok(parsed) => {
            println!("Pattern compiled successfully: {:?}", parsed);
            
            // Try to evaluate it
            match fluent_ai_http3::json_path::CoreJsonPathEvaluator::new(pattern) {
                Ok(evaluator) => {
                    println!("Evaluator created successfully");
                    
                    match evaluator.evaluate(&json_value) {
                        Ok(results) => {
                            let elapsed = start.elapsed();
                            println!("Evaluation completed in {:?}", elapsed);
                            println!("Results: {:?}", results);
                        }
                        Err(e) => {
                            let elapsed = start.elapsed();
                            println!("Evaluation failed in {:?}: {:?}", elapsed, e);
                        }
                    }
                }
                Err(e) => {
                    println!("Failed to create evaluator: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("Failed to compile pattern: {:?}", e);
        }
    }
    
    println!("Test completed");
}