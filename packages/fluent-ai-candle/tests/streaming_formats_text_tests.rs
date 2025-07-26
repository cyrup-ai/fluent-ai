use fluent_ai_candle::streaming::formats::text::*;
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
    fn test_sse_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        let response = create_test_response();

        let formatted = formatter.format_sse(&response).unwrap();
        assert!(formatted.starts_with("id: token-0"));
        assert!(formatted.contains("event: token"));
        assert!(formatted.contains("data: "));
        assert!(formatted.ends_with("\n\n"));
    }

    #[test]
    fn test_plain_text_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let response = create_test_response();

        let formatted = formatter.format_plain_text(&response).unwrap();
        assert_eq!(formatted, "test token");
    }

    #[test]
    fn test_raw_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Raw);
        let response = create_test_response();

        let formatted = formatter.format_raw(&response).unwrap();
        assert_eq!(formatted, "5|0|test token");
    }

    #[test]
    fn test_unified_format_response() {
        let mut formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let response = create_test_response();

        let formatted = formatter.format_response(&response).unwrap();
        assert_eq!(formatted, "test token");
        assert_eq!(formatter.sequence_number(), 1);
    }

    #[test]
    fn test_end_markers() {
        let mut json_formatter = StreamingFormatter::new(OutputFormat::Json);
        let json_end = json_formatter.format_end_marker().unwrap();
        assert!(json_end.contains("\"type\":\"end\""));

        let mut sse_formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        let sse_end = sse_formatter.format_end_marker().unwrap();
        assert!(sse_end.contains("event: end"));

        let mut text_formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let text_end = text_formatter.format_end_marker().unwrap();
        assert_eq!(text_end, "\n[END]\n");

        let mut raw_formatter = StreamingFormatter::new(OutputFormat::Raw);
        let raw_end = raw_formatter.format_end_marker().unwrap();
        assert_eq!(raw_end, "END|0|");
    }

    #[test]
    fn test_error_formatting() {
        let mut formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let error = StreamingError::Utf8Error("test error".to_string());

        let formatted = formatter.format_error(&error).unwrap();
        assert!(formatted.contains("ERROR: test error"));
    }

    #[test]
    fn test_format_parsing() {
        assert_eq!(
            text_utils::parse_format("json").unwrap(),
            OutputFormat::Json
        );
        assert_eq!(
            text_utils::parse_format("sse").unwrap(),
            OutputFormat::ServerSentEvents
        );
        assert_eq!(
            text_utils::parse_format("websocket").unwrap(),
            OutputFormat::WebSocket
        );
        assert_eq!(
            text_utils::parse_format("text").unwrap(),
            OutputFormat::PlainText
        );
        assert_eq!(
            text_utils::parse_format("raw").unwrap(),
            OutputFormat::Raw
        );

        if let OutputFormat::Custom(name) = text_utils::parse_format("custom:test").unwrap() {
            assert_eq!(name, "test");
        } else {
            panic!("Expected custom format");
        }

        assert!(text_utils::parse_format("invalid").is_err());
    }

    #[test]
    fn test_sse_data_escaping() {
        let data = "line1\nline2\rline3";
        let escaped = text_utils::escape_sse_data(data);
        assert_eq!(escaped, "line1\\nline2\\rline3");
    }

    #[test]
    fn test_raw_data_escaping() {
        let data = "field1|field2\nfield3";
        let escaped = text_utils::escape_raw_data(data);
        assert_eq!(escaped, "field1\\|field2\\nfield3");
    }

    #[test]
    fn test_text_chunking() {
        let text = "This is a long text that needs to be chunked for optimal streaming performance";
        let chunks = text_utils::chunk_text_for_streaming(text, 20);
        
        assert!(chunks.len() > 1);
        for chunk in chunks {
            assert!(chunk.len() <= 25); // Allow for word boundary adjustments
        }
    }

    #[test]
    fn test_sse_format_validation() {
        let valid_data = "data: some content\n\n";
        assert!(text_utils::validate_sse_format(valid_data).is_ok());

        let invalid_data = "data: some\n\ncontent in middle";
        assert!(text_utils::validate_sse_format(invalid_data).is_err());
    }
