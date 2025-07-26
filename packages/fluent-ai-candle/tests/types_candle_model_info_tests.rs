use fluent_ai_candle::types::candle_model::info::*;
use fluent_ai_candle::types::candle_model::*;

#[test]
    fn test_model_info_builder() {
        let model = ModelInfo::builder()
            .provider_name("test")
            .name("test-model")
            .max_input_tokens(4096)
            .max_output_tokens(2048)
            .pricing(1.5, 2.0)
            .with_vision(true)
            .with_function_calling(true)
            .with_streaming(true)
            .with_embeddings(false)
            .requires_max_tokens(true)
            .with_thinking(100)
            .build()
            .unwrap();

        assert_eq!(model.provider_name, "test");
        assert_eq!(model.name, "test-model");
        assert_eq!(model.max_input_tokens.unwrap().get(), 4096);
        assert_eq!(model.max_output_tokens.unwrap().get(), 2048);
        assert_eq!(model.input_price, Some(1.5));
        assert_eq!(model.output_price, Some(2.0));
        assert!(model.supports_vision);
        assert!(model.supports_function_calling);
        assert!(model.supports_streaming);
        assert!(!model.supports_embeddings);
        assert!(model.requires_max_tokens);
        assert!(model.supports_thinking);
        assert_eq!(model.optimal_thinking_budget, Some(100));
    }

    #[test]
    fn test_model_info_validation() {
        // Missing provider name
        let err = ModelInfo::builder().name("test").build().unwrap_err();
        assert!(matches!(
            err,
            ModelError::InvalidConfiguration("provider_name is required")
        ));

        // Missing model name
        let err = ModelInfo::builder()
            .provider_name("test")
            .build()
            .unwrap_err();
        assert!(matches!(
            err,
            ModelError::InvalidConfiguration("name is required")
        ));

        // Zero input tokens
        let err = ModelInfo::builder()
            .provider_name("test")
            .name("test")
            .max_input_tokens(0)
            .build()
            .unwrap_err();
        assert!(matches!(err, ModelError::InvalidConfiguration(_)));

        // Zero output tokens
        let err = ModelInfo::builder()
            .provider_name("test")
            .name("test")
            .max_output_tokens(0)
            .build()
            .unwrap_err();
        assert!(matches!(err, ModelError::InvalidConfiguration(_)));

        // Thinking without budget
        let err = ModelInfo::builder()
            .provider_name("test")
            .name("test")
            .supports_thinking = true;
        // Note: This is a compile-time check, so we don't need to test it here
    }

    #[test]
    fn test_provider_models() {
        let mut provider = ProviderModels::new("test");

        let model1 = ModelInfo::builder()
            .provider_name("test")
            .name("model1")
            .build()
            .unwrap();

        let model2 = ModelInfo::builder()
            .provider_name("test")
            .name("model2")
            .build()
            .unwrap();

        provider.add_model(model1.clone()).unwrap();
        provider.add_model(model2.clone()).unwrap();

        // Test duplicate model
        let err = provider.add_model(model1).unwrap_err();
        assert!(matches!(
            err,
            ModelError::ModelAlreadyExists {
                provider: "test",
                name: "model1"
            }
        ));

        // Test wrong provider
        let wrong_provider = ModelInfo::builder()
            .provider_name("wrong")
            .name("model3")
            .build()
            .unwrap();

        let err = provider.add_model(wrong_provider).unwrap_err();
        assert!(matches!(
            err,
            ModelError::InvalidConfiguration("model provider does not match collection provider")
        ));

        // Test get and all
        assert_eq!(provider.get("model1").unwrap().name, "model1");
        assert_eq!(provider.get("nonexistent"), None);
        assert_eq!(provider.all().len(), 2);
    }
