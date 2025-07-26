use fluent_ai_candle::types::candle_chat::message::message_processing::*;
use fluent_ai_candle::types::candle_chat::message::super::types::{Message, MessageRole};
use fluent_ai_candle::types::candle_chat::message::*;

#[test]
    fn test_process_message() {
        let mut message = Message {
            role: MessageRole::User,
            content: "  Hello, world!  ".to_string(),
            id: None,
            timestamp: None};

        process_message(&mut message).unwrap();
        assert_eq!(message.content, "Hello, world!");
    }

    #[test]
    fn test_validate_message() {
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

        assert!(validate_message(&valid_message).is_ok());
        assert!(validate_message(&empty_message).is_err());
    }

    #[test]
    fn test_sanitize_content() {
        assert_eq!(sanitize_content("  Hello, world!  "), "Hello, world!");
        assert_eq!(sanitize_content(""), "");
        assert_eq!(sanitize_content("  "), "");
    }
