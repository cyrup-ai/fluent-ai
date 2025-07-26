use fluent_ai_candle::types::candle_chat::chat::config::mod::*;
use fluent_ai_candle::types::candle_chat::chat::config::*;

#[test]
    fn test_configuration_builder() {
        let config = ConfigurationBuilder::new()
            .name("Test Config")
            .description("A test configuration")
            .model(|m| m.provider("openai").model_name("gpt-4").temperature(0.7))
            .personality(|p| p.name("TestBot").personality_type("professional"))
            .build();

        assert_eq!(config.name.as_ref(), "Test Config");
        assert_eq!(config.description.as_ref().unwrap().as_ref(), "A test configuration");
        assert_eq!(config.model.provider.as_ref(), "openai");
        assert_eq!(config.model.model_name.as_ref(), "gpt-4");
        assert_eq!(config.model.temperature, 0.7);
        assert_eq!(config.personality.name.as_ref(), "TestBot");
    }

    #[test]
    fn test_configuration_manager() {
        let manager = ConfigurationManager::new();
        let config = manager.get_config();
        
        assert_eq!(config.name.as_ref(), "Default Chat Configuration");
        assert_eq!(manager.get_version(), 1);
    }

    #[test]
    fn test_validation() {
        let valid_config = professional();
        assert!(validate_configuration(&valid_config).is_ok());

        let mut invalid_config = ConfigurationBuilder::new().build();
        invalid_config.personality.formality_level = 2.0; // Invalid: should be 0.0-1.0
        
        assert!(validate_configuration(&invalid_config).is_err());
    }

    #[test]
    fn test_presets() {
        let professional = professional();
        assert_eq!(professional.personality.personality_type.as_ref(), "professional");

        let casual = casual();
        assert_eq!(casual.personality.personality_type.as_ref(), "casual");

        let creative = creative();
        assert_eq!(creative.personality.personality_type.as_ref(), "creative");

        let technical = technical();
        assert_eq!(technical.personality.personality_type.as_ref(), "technical");
    }
