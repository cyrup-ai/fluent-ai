use fluent_ai_candle::tokenizer::mod::*;
use fluent_ai_candle::tokenizer::*;

#[tokio::test]
    async fn test_tokenizer_config_builder() {
        let config = TokenizerConfigBuilder::new()
            .add_bos_token(true)
            .add_eos_token(false)
            .max_length(Some(1024))
            .build();

        assert!(config.add_bos_token);
        assert!(!config.add_eos_token);
        assert_eq!(config.max_length, Some(1024));
    }

    #[test]
    fn test_chat_message() {
        let msg = crate::types::CandleMessage::system("You are a helpful assistant");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are a helpful assistant");

        let user_msg = crate::types::CandleMessage::user("Hello world");
        assert_eq!(user_msg.role, "user");
        assert_eq!(user_msg.content, "Hello world");
    }

    #[test]
    fn test_utils() {
        let config = utils::config_for_model_type("llama");
        assert!(config.add_bos_token);
        assert!(!config.add_eos_token);

        let config = utils::config_for_model_type("phi");
        assert!(!config.add_bos_token);
        assert!(config.add_eos_token);
    }
