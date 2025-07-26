use fluent_ai_candle::streaming::formats::json::*;
use fluent_ai_candle::streaming::formats::*;

use crate::streaming::{TokenMetadata, TokenTiming};
    use crate::streaming::formats::types::OutputFormat;

    fn create_test_response() -> StreamingTokenResponse {
        StreamingTokenResponse {
            content: "test token".to_string(),
            token_id: Some(123),
            position: 5,
            is_complete_token: true,
            probability: Some(0.95),
            alternatives: None,
            timing: TokenTiming::default(),
            metadata: TokenMetadata::default()}
    }

    #[test]
    fn test_json_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Json);
        let response = create_test_response();

        let formatted = formatter.format_json(&response).unwrap();
        assert!(formatted.contains("test token"));
        assert!(formatted.contains("123"));
        assert!(formatted.contains("0.95"));
    }

    #[test]
    fn test_websocket_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::WebSocket);
        let response = create_test_response();

        let formatted = formatter.format_websocket(&response).unwrap();
        assert!(formatted.contains("\"type\":\"token\""));
        assert!(formatted.contains("\"sequence\":0"));
    }

    #[test]
    fn test_minimal_json_custom_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("minimal_json".to_string()));
        let response = create_test_response();

        let formatted = formatter.format_custom(&response, "minimal_json").unwrap();
        assert!(formatted.contains("test token"));
        assert!(formatted.contains("\"sequence_id\":5"));
        assert!(formatted.contains("\"is_final\":true"));
    }

    #[test]
    fn test_csv_custom_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("csv".to_string()));
        let response = create_test_response();

        let formatted = formatter.format_custom(&response, "csv").unwrap();
        assert_eq!(formatted, "5,test token,true,1.0");
    }

    #[test]
    fn test_csv_comma_escaping() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("csv".to_string()));
        let mut response = create_test_response();
        response.text = "test, token".to_string();

        let formatted = formatter.format_custom(&response, "csv").unwrap();
        assert!(formatted.contains("test\\, token"));
    }

    #[test]
    fn test_unknown_custom_format() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("unknown".to_string()));
        let response = create_test_response();

        let result = formatter.format_custom(&response, "unknown");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown custom format"));
    }

    #[test]
    fn test_json_end_marker() {
        let formatter = StreamingFormatter::new(OutputFormat::Json);
        let end_marker = formatter.format_json_end_marker();

        assert!(end_marker.contains("\"type\":\"end\""));
        assert!(end_marker.contains("\"sequence\":0"));
    }

    #[test]
    fn test_websocket_end_marker() {
        let formatter = StreamingFormatter::new(OutputFormat::WebSocket);
        let end_marker = formatter.format_websocket_end_marker().unwrap();

        assert!(end_marker.contains("\"type\":\"end\""));
        assert!(end_marker.contains("\"sequence\":0"));
    }

    #[test]
    fn test_json_error_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Json);
        let error = StreamingError::Utf8Error("test error".to_string());

        let formatted = formatter.format_json_error(&error);
        assert!(formatted.contains("\"type\":\"error\""));
        assert!(formatted.contains("test error"));
    }

    #[test]
    fn test_websocket_error_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::WebSocket);
        let error = StreamingError::Utf8Error("test error".to_string());

        let formatted = formatter.format_websocket_error(&error).unwrap();
        assert!(formatted.contains("\"type\":\"error\""));
        assert!(formatted.contains("test error"));
    }

    #[test]
    fn test_custom_end_markers() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("minimal_json".to_string()));
        
        let minimal_end = formatter.format_custom_end_marker("minimal_json").unwrap();
        assert_eq!(minimal_end, serde_json::json!({"end": true}).to_string());

        let csv_end = formatter.format_custom_end_marker("csv").unwrap();
        assert_eq!(csv_end, "END,[END],0.0");
    }

    #[test]
    fn test_custom_error_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("csv".to_string()));
        let error = StreamingError::Utf8Error("test, error".to_string());

        let formatted = formatter.format_custom_error(&error, "csv").unwrap();
        assert!(formatted.contains("ERROR,"));
        assert!(formatted.contains("test\\, error")); // Comma should be escaped
    }
