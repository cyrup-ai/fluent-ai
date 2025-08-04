//! JSON Path Deserializer Tests
//!
//! Tests for the JSONPath deserializer functionality, moved from src/json_path/deserializer_old.rs

use bytes::Bytes;
use fluent_ai_http3::json_path::{
    JsonPathParser, buffer::StreamBuffer, deserializer::JsonPathDeserializer,
    state_machine::StreamStateMachine,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct TestModel {
    id: String,
    value: i32,
}

#[cfg(test)]
mod deserializer_tests {
    use super::*;

    #[test]
    fn test_simple_array_deserialization() {
        let json_data = r#"[{"id":"test1","value":42},{"id":"test2","value":24}]"#;
        let path_expr = JsonPathParser::compile("$[*]").expect("Valid JSONPath expression");
        let mut buffer = StreamBuffer::with_capacity(1024);
        let mut state = StreamStateMachine::new();

        buffer.append_chunk(Bytes::from(json_data));

        let mut deserializer =
            JsonPathDeserializer::<TestModel>::new(&path_expr, &mut buffer, &mut state);
        let _results: Vec<_> = deserializer.process_available().collect();

        assert_eq!(_results.len(), 2);
        assert!(_results[0].is_ok());
        assert!(_results[1].is_ok());

        let first = _results[0].as_ref().unwrap();
        assert_eq!(first.id, "test1");
        assert_eq!(first.value, 42);
    }

    #[test]
    fn test_nested_object_deserialization() {
        let json_data = r#"{"data":[{"id":"nested1","value":100}],"meta":"info"}"#;
        let path_expr = JsonPathParser::compile("$.data[*]").expect("Valid JSONPath expression");
        let mut buffer = StreamBuffer::with_capacity(1024);
        let mut state = StreamStateMachine::new();

        buffer.append_chunk(Bytes::from(json_data));

        let mut deserializer =
            JsonPathDeserializer::<TestModel>::new(&path_expr, &mut buffer, &mut state);
        let _results: Vec<_> = deserializer.process_available().collect();

        assert_eq!(_results.len(), 1);
        assert!(_results[0].is_ok());

        let item = _results[0].as_ref().unwrap();
        assert_eq!(item.id, "nested1");
        assert_eq!(item.value, 100);
    }

    #[test]
    fn test_streaming_chunks() {
        let path_expr = JsonPathParser::compile("$.items[*]").expect("Valid JSONPath expression");
        let mut buffer = StreamBuffer::with_capacity(1024);
        let mut state = StreamStateMachine::new();

        // Add data in chunks to simulate streaming
        buffer.append_chunk(Bytes::from(r#"{"items":["#));
        buffer.append_chunk(Bytes::from(r#"{"id":"chunk1","value":1},"#));
        buffer.append_chunk(Bytes::from(r#"{"id":"chunk2","value":2}"#));
        buffer.append_chunk(Bytes::from(r#"]}"#));

        let mut deserializer =
            JsonPathDeserializer::<TestModel>::new(&path_expr, &mut buffer, &mut state);
        let _results: Vec<_> = deserializer.process_available().collect();

        assert_eq!(_results.len(), 2);
        assert!(_results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_malformed_json_error_handling() {
        let json_data = r#"{"data":[{"id":"test1","invalid":}]}"#; // Missing value
        let path_expr = JsonPathParser::compile("$.data[*]").expect("Valid JSONPath expression");
        let mut buffer = StreamBuffer::with_capacity(1024);
        let mut state = StreamStateMachine::new();

        buffer.append_chunk(Bytes::from(json_data));

        let mut deserializer =
            JsonPathDeserializer::<TestModel>::new(&path_expr, &mut buffer, &mut state);
        let _results: Vec<_> = deserializer.process_available().collect();

        assert!(!_results.is_empty());
        assert!(_results[0].is_err());
    }
}
