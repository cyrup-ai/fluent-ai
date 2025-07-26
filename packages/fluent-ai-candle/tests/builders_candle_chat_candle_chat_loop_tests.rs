use fluent_ai_candle::builders::candle_chat::candle_chat_loop::*;
use fluent_ai_candle::builders::candle_chat::*;

#[test]
    fn test_chat_loop_reprompt() {
        let loop_val = ChatLoop::Reprompt("Hello".to_string());
        assert!(loop_val.should_continue());
        assert_eq!(loop_val.message(), Some("Hello"));
        assert_eq!(loop_val.user_prompt(), None);
        assert!(!loop_val.needs_user_input());
    }

    #[test]
    fn test_chat_loop_user_prompt_with_message() {
        let loop_val = ChatLoop::UserPrompt(Some("Enter your choice:".to_string()));
        assert!(loop_val.should_continue());
        assert_eq!(loop_val.message(), None);
        assert_eq!(loop_val.user_prompt(), Some("Enter your choice:"));
        assert!(loop_val.needs_user_input());
    }

    #[test]
    fn test_chat_loop_user_prompt_without_message() {
        let loop_val = ChatLoop::UserPrompt(None);
        assert!(loop_val.should_continue());
        assert_eq!(loop_val.message(), None);
        assert_eq!(loop_val.user_prompt(), None);
        assert!(loop_val.needs_user_input());
    }

    #[test]
    fn test_chat_loop_break() {
        let loop_val = ChatLoop::Break;
        assert!(!loop_val.should_continue());
        assert_eq!(loop_val.message(), None);
        assert_eq!(loop_val.user_prompt(), None);
        assert!(!loop_val.needs_user_input());
    }

    #[test]
    fn test_from_string() {
        let loop_val: ChatLoop = "test message".into();
        assert_eq!(loop_val, ChatLoop::Reprompt("test message".to_string()));
    }

    #[test]
    fn test_from_str() {
        let loop_val: ChatLoop = String::from("test message").into();
        assert_eq!(loop_val, ChatLoop::Reprompt("test message".to_string()));
    }
