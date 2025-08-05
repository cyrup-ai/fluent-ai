use std::time::Instant;
use bytes::Bytes;
use fluent_ai_http3::json_path::JsonArrayStream;

fn main() {
    println!("Testing JsonArrayStream quick fix");
    
    let json_data = r#"{"text": "aaaaaaaaaa"}"#;
    let pattern = "$.text[?match(@, 'a+')]";
    
    let mut stream = JsonArrayStream::<serde_json::Value>::new(pattern);
    let chunk = Bytes::from(json_data);
    
    let start = Instant::now();
    
    // Test synchronous method
    let results = stream.process_chunk_sync(chunk);
    let elapsed = start.elapsed();
    
    println!("Results: {:?}", results);
    println!("Time: {:?}", elapsed);
    
    // Should be very fast now
    if elapsed.as_millis() < 100 {
        println!("✅ SUCCESS: Fixed the timeout issue!");
    } else {
        println!("❌ FAIL: Still taking too long");
    }
}