use fluent_ai_candle::streaming::formats::types::*;
use fluent_ai_candle::streaming::formats::*;

#[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::ServerSentEvents.to_string(), "sse");
        assert_eq!(OutputFormat::WebSocket.to_string(), "websocket");
        assert_eq!(OutputFormat::PlainText.to_string(), "text");
        assert_eq!(OutputFormat::Raw.to_string(), "raw");
        assert_eq!(
            OutputFormat::Custom("test".to_string()).to_string(),
            "custom:test"
        );
    }

    #[test]
    fn test_output_format_default() {
        assert_eq!(OutputFormat::default(), OutputFormat::Json);
    }

    #[test]
    fn test_streaming_formatter_creation() {
        let formatter = StreamingFormatter::new(OutputFormat::Json);
        assert_eq!(formatter.format(), &OutputFormat::Json);
        assert_eq!(formatter.sequence_number(), 0);
    }

    #[test]
    fn test_event_id_prefix() {
        let mut formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        formatter.set_event_id_prefix("custom".to_string());
        assert_eq!(formatter.event_id_prefix, "custom");
    }

    #[test]
    fn test_metadata_support() {
        let json_formatter = StreamingFormatter::new(OutputFormat::Json);
        assert!(json_formatter.supports_metadata());

        let text_formatter = StreamingFormatter::new(OutputFormat::PlainText);
        assert!(!text_formatter.supports_metadata());
    }

    #[test]
    fn test_web_compatibility() {
        let sse_formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        assert!(sse_formatter.is_web_compatible());

        let raw_formatter = StreamingFormatter::new(OutputFormat::Raw);
        assert!(!raw_formatter.is_web_compatible());
    }

    #[test]
    fn test_content_type() {
        let json_formatter = StreamingFormatter::new(OutputFormat::Json);
        assert_eq!(json_formatter.content_type(), "application/json");

        let sse_formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        assert_eq!(sse_formatter.content_type(), "text/event-stream");
    }

    #[test]
    fn test_sequence_reset() {
        let mut formatter = StreamingFormatter::new(OutputFormat::Json);
        formatter.increment_sequence();
        formatter.increment_sequence();
        assert_eq!(formatter.sequence_number(), 2);

        formatter.reset_sequence();
        assert_eq!(formatter.sequence_number(), 0);
    }

    #[test]
    fn test_buffer_size_optimization() {
        assert_eq!(format_constants::optimal_buffer_size(&OutputFormat::Raw), 256);
        assert_eq!(
            format_constants::optimal_buffer_size(&OutputFormat::PlainText),
            512
        );
        assert_eq!(format_constants::optimal_buffer_size(&OutputFormat::Json), 1024);
        assert_eq!(
            format_constants::optimal_buffer_size(&OutputFormat::ServerSentEvents),
            2048
        );
    }

    #[test]
    fn test_buffering_requirements() {
        assert!(format_constants::requires_buffering(
            &OutputFormat::ServerSentEvents
        ));
        assert!(format_constants::requires_buffering(&OutputFormat::WebSocket));
        assert!(!format_constants::requires_buffering(&OutputFormat::PlainText));
        assert!(!format_constants::requires_buffering(&OutputFormat::Raw));
    }
