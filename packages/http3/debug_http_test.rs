use fluent_ai_http3::{Http3, HttpStreamExt};

#[tokio::main] 
async fn main() {
    env_logger::init();
    
    println!("🔍 Starting standalone HTTP debug test...");
    
    let stream = Http3::json()
        .debug()
        .get("https://httpbin.org/get");

    let responses: Vec<serde_json::Value> = stream.collect();
    
    println!("🔍 Collected {} responses", responses.len());
    
    if !responses.is_empty() {
        println!("🔍 First response: {:?}", responses[0]);
    } else {
        println!("🔍 No responses received!");
    }
}