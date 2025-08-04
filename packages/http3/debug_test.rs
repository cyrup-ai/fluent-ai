use bytes::Bytes;
use fluent_ai_http3::json_path::{JsonArrayStream, JsonPathParser};

fn main() {
    let test_json = r#"{
      "store": {
        "books": [
          {
            "id": 1,
            "name": "Book One",
            "value": 10.5,
            "active": true,
            "metadata": {"category": "fiction", "pages": 300}
          },
          {
            "id": 2,
            "name": "Book Two", 
            "value": 25.0,
            "active": false,
            "metadata": {"category": "science", "pages": 450}
          },
          {
            "id": 3,
            "name": "Book Three",
            "value": 15.75,
            "active": true,
            "metadata": null
          }
        ]
      }
    }"#;

    println!("Testing JSONPath: $.store.books[?@.metadata == null]");
    
    let mut stream = JsonArrayStream::<serde_json::Value>::new("$.store.books[?@.metadata == null]");
    let chunk = Bytes::from(test_json);
    let results: Vec<_> = stream.process_chunk(chunk).collect();
    
    println!("Found {} results:", results.len());
    for (i, result) in results.iter().enumerate() {
        println!("  [{}]: {}", i, result);
    }
    
    // Also test compilation
    match JsonPathParser::compile("$.store.books[?@.metadata == null]") {
        Ok(expr) => println!("JSONPath compiled successfully: {}", expr.original()),
        Err(e) => println!("JSONPath compilation failed: {:?}", e),
    }
}