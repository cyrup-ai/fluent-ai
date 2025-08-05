use std::time::Instant;
use bytes::Bytes;
use fluent_ai_http3::json_path::JsonArrayStream;

fn main() {
    println!("Testing JsonArrayStream hanging issue");
    
    let json_data = r#"{"text": "aaaaaaaaaa"}"#;
    let pattern = "$.text[?match(@, 'a+')]";
    
    println!("JSON: {}", json_data);
    println!("Pattern: {}", pattern);
    
    let mut stream = JsonArrayStream::<serde_json::Value>::new(pattern);
    let chunk = Bytes::from(json_data);
    
    println!("About to call process_chunk...");
    let start = Instant::now();
    
    // Set a timeout
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(10));
        println!("TIMEOUT: Process took longer than 10 seconds!");
        std::process::exit(1);
    });
    
    let results: Vec<_> = stream.process_chunk(chunk).collect();
    let elapsed = start.elapsed();
    
    println!("Completed in {:?}", elapsed);
    println!("Results: {:?}", results);
}