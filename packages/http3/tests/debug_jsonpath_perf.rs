use std::time::Instant;
use bytes::Bytes;
use fluent_ai_http3::json_path::{JsonArrayStream, JsonPathParser, CoreJsonPathEvaluator};

#[test]
fn test_jsonpath_filter_on_string_performance() {
    println!("Testing JSONPath pattern: $.text[?match(@, 'a+')]");
    
    // This is the exact same pattern and data from the failing test
    let json_data = r#"{"text": "aaaaaaaaaa"}"#;
    println!("Input JSON: {}", json_data);
    
    let json_value: serde_json::Value = serde_json::from_str(json_data).unwrap();
    println!("Parsed JSON: {:?}", json_value);
    
    // Test the specific pattern from the failing test
    let pattern = "$.text[?match(@, 'a+')]";
    
    let start = Instant::now();
    
    // Test 1: Direct compilation
    println!("\n=== Test 1: JSONPath Compilation ===");
    let compilation_start = Instant::now();
    let parsed_result = JsonPathParser::compile(pattern);
    let compilation_time = compilation_start.elapsed();
    println!("Compilation time: {:?}", compilation_time);
    
    match parsed_result {
        Ok(parsed) => {
            println!("Pattern compiled successfully");
            println!("Expression: {:?}", parsed);
        }
        Err(e) => {
            println!("Failed to compile pattern: {:?}", e);
            return;
        }
    }
    
    // Test 2: Core evaluator
    println!("\n=== Test 2: Core Evaluator ===");
    let eval_start = Instant::now();
    let evaluator = CoreJsonPathEvaluator::new(pattern).unwrap();
    let eval_create_time = eval_start.elapsed();
    println!("Evaluator creation time: {:?}", eval_create_time);
    
    let eval_exec_start = Instant::now();
    let eval_result = evaluator.evaluate(&json_value);
    let eval_exec_time = eval_exec_start.elapsed();
    println!("Evaluator execution time: {:?}", eval_exec_time);
    
    match eval_result {
        Ok(results) => {
            println!("Evaluation results: {:?}", results);
            println!("Number of results: {}", results.len());
        }
        Err(e) => {
            println!("Evaluation failed: {:?}", e);
        }
    }
    
    // Test 3: JsonArrayStream (this is what the test actually uses)
    println!("\n=== Test 3: JsonArrayStream Processing ===");
    let stream_start = Instant::now();
    let mut stream = JsonArrayStream::<serde_json::Value>::new(pattern);
    let stream_create_time = stream_start.elapsed();
    println!("Stream creation time: {:?}", stream_create_time);
    
    let chunk = Bytes::from(json_data);
    let stream_process_start = Instant::now();
    
    // Set a timeout for this operation
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(5));
        println!("TIMEOUT: Stream processing took longer than 5 seconds!");
        std::process::exit(1);
    });
    
    let results: Vec<_> = stream.process_chunk_sync(chunk);
    let stream_process_time = stream_process_start.elapsed();
    
    println!("Stream processing time: {:?}", stream_process_time);
    println!("Stream results: {:?}", results);
    println!("Number of stream results: {}", results.len());
    
    let total_time = start.elapsed();
    println!("\n=== Total Test Time: {:?} ===", total_time);
    
    // Fail if any individual step took too long
    assert!(compilation_time.as_millis() < 100, "Compilation took too long: {:?}", compilation_time);
    assert!(eval_exec_time.as_millis() < 100, "Core evaluation took too long: {:?}", eval_exec_time);
    assert!(stream_process_time.as_millis() < 1000, "Stream processing took too long: {:?}", stream_process_time);
}

#[test] 
fn test_simplified_jsonpath_patterns() {
    println!("\n=== Testing Simplified Patterns ===");
    
    let json_data = r#"{"text": "aaaaaaaaaa"}"#;
    let json_value: serde_json::Value = serde_json::from_str(json_data).unwrap();
    
    // Test patterns in order of complexity
    let test_patterns = vec![
        "$.text",                          // Simple property access
        "$[?(@.text)]",                   // Filter on root object
        "$.text[0]",                      // Index access (should fail gracefully)
        "$.text[?true]",                  // Filter with constant
        "$.text[?match(@, 'a')]",         // The problematic filter  
    ];
    
    for pattern in test_patterns {
        println!("\nTesting pattern: {}", pattern);
        let start = Instant::now();
        
        match CoreJsonPathEvaluator::new(pattern) {
            Ok(evaluator) => {
                match evaluator.evaluate(&json_value) {
                    Ok(results) => {
                        let elapsed = start.elapsed();
                        println!("  ✅ Success in {:?}: {} results", elapsed, results.len());
                        if elapsed.as_millis() > 100 {
                            println!("  ⚠️  Warning: took longer than 100ms");
                        }
                    }
                    Err(e) => {
                        let elapsed = start.elapsed();
                        println!("  ❌ Error in {:?}: {:?}", elapsed, e);
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Failed to create evaluator: {:?}", e);
            }
        }
    }
}