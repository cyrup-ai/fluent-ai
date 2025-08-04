use fluent_ai_http3::json_path::deserializer::core::JsonPathDeserializer;
use fluent_ai_http3::json_path::parser::JsonPathParser;
use fluent_ai_http3::json_path::buffer::StreamBuffer;
use fluent_ai_http3::json_path::state_machine::StreamStateMachine;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, PartialEq)]
struct TestModel {
    id: String,
    value: i32,
}

fn main() {
    let json_data = r#"{"data":[{"id":"nested1","value":100}],"meta":"info"}"#;
    let path_expr = JsonPathParser::compile("$.data[*]").expect("Valid JSONPath expression");
    let mut buffer = StreamBuffer::with_capacity(1024);
    let mut state = StreamStateMachine::new();

    buffer.append_chunk(Bytes::from(json_data));

    let mut deserializer =
        JsonPathDeserializer::<TestModel>::new(&path_expr, &mut buffer, &mut state);
    
    println!("JSON: {}", json_data);
    println!("JSONPath: $.data[*]");
    println!("Expected target property: {:?}", deserializer.target_property);
    
    let results: Vec<_> = deserializer.process_available().collect();
    
    println!("Results count: {}", results.len());
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(model) => println!("Result {}: {:?}", i, model),
            Err(e) => println!("Error {}: {:?}", i, e),
        }
    }
}