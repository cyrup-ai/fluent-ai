use bytes::Bytes;
use fluent_ai_http3::json_path::{JsonArrayStream, JsonPathParser};
use serde_json::json;

fn main() {
    println!("🧪 Testing RFC 9535 root-only query compliance...");

    // Test 1: Verify "$" compiles successfully
    match JsonPathParser::compile("$") {
        Ok(_) => println!("✅ JsonPathParser::compile('$') returns Ok - RFC 9535 compliant!"),
        Err(e) => {
            println!("❌ JsonPathParser::compile('$') failed: {:?}", e);
            return;
        }
    }

    // Test 2: Verify "$" returns the root node itself
    let test_data = json!({
        "name": "test",
        "items": [1, 2, 3],
        "nested": {
            "value": 42
        }
    });

    let json_str = serde_json::to_string(&test_data).unwrap();
    let mut stream = JsonArrayStream::<serde_json::Value>::new("$");
    
    let chunk = Bytes::from(json_str);
    let results: Vec<_> = stream.process_chunk(chunk).collect();
    
    println!("📊 Results from '$' query:");
    println!("   Count: {}", results.len());
    
    if results.len() == 1 {
        println!("✅ Correct: '$' returns exactly 1 result (the root node)");
        if results[0] == test_data {
            println!("✅ Correct: '$' returns the entire query argument as expected per RFC 9535");
        } else {
            println!("❌ Wrong: '$' result doesn't match original data");
            println!("   Expected: {}", test_data);
            println!("   Got: {}", results[0]);
        }
    } else {
        println!("❌ Wrong: '$' should return exactly 1 result, got {}", results.len());
    }

    println!("\n🎯 RFC 9535 Section 2.2.2: 'The root identifier $ represents the root node of the query argument.'");
    println!("🎯 RFC 9535 ABNF: jsonpath-query = root-identifier segments, where segments = *(S segment)");
    println!("🎯 Therefore '$' with zero segments is valid and should return the root node itself.");
}