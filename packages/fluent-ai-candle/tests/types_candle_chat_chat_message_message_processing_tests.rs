use fluent_ai_candle::types::candle_chat::chat::message::message_processing::*;
use fluent_ai_candle::types::candle_chat::chat::message::*;

#[test]
    fn test_process_message() {
        let message = Message {
            role: MessageRole::User,
            content: "  Hello, world!  ".to_string(),
            id: None,
            timestamp: None};

        // Test would use the async stream result in real implementation
        let _ = process_message(message);
    }

    #[test]
    fn test_validate_message_sync() {
        let valid_message = Message {
            role: MessageRole::User,
            content: "Hello, world!".to_string(),
            id: None,
            timestamp: None};

        let empty_message = Message {
            role: MessageRole::User,
            content: "   ".to_string(),
            id: None,
            timestamp: None};

        assert!(validate_message_sync(&valid_message).is_ok());
        // Empty content after trimming should fail validation
        assert!(validate_message_sync(&empty_message).is_err());
    }

    #[test]
    fn test_sanitize_content() {
        assert_eq!(sanitize_content("  Hello, world!  "), "Hello, world!");
        assert_eq!(sanitize_content(""), "");
        assert_eq!(sanitize_content("  "), "");
    }
