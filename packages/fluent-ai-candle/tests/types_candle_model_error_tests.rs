use fluent_ai_candle::types::candle_model::error::*;
use fluent_ai_candle::types::candle_model::*;

#[test]
    fn test_model_error_display() {
        assert_eq!(
            ModelError::ModelNotFound {
                provider: "test",
                name: "test"
            }
            .to_string(),
            "Model not found: test:test"
        );
        assert_eq!(
            ModelError::ProviderNotFound("test").to_string(),
            "Provider not found: test"
        );
        assert_eq!(
            ModelError::ModelAlreadyExists {
                provider: "test",
                name: "test"
            }
            .to_string(),
            "Model already registered: test:test"
        );
        assert_eq!(
            ModelError::InvalidConfiguration("test").to_string(),
            "Invalid model configuration: test"
        );
        assert_eq!(
            ModelError::OperationNotSupported("test").to_string(),
            "Operation not supported by model: test"
        );
        assert_eq!(
            ModelError::InvalidInput("test").to_string(),
            "Invalid input: test"
        );
        assert_eq!(
            ModelError::Internal("test").to_string(),
            "Internal error: test"
        );
    }

    #[test]
    fn test_option_ext() {
        let some: Option<u32> = Some(42);
        assert_eq!(some.or_model_not_found("test", "test").unwrap(), 42);

        let none: Option<u32> = None;
        assert!(matches!(
            none.or_model_not_found("test", "test"),
            Err(ModelError::ModelNotFound {
                provider: "test",
                name: "test"
            })
        ));
    }

    #[test]
    fn test_result_ext() {
        let ok: std::result::Result<u32, &str> = Ok(42);
        assert_eq!(ok.invalid_config("test").unwrap(), 42);
        assert_eq!(ok.not_supported("test").unwrap(), 42);

        let err: std::result::Result<u32, &str> = Err("error");
        assert!(matches!(
            err.invalid_config("test"),
            Err(ModelError::InvalidConfiguration("test"))
        ));
        assert!(matches!(
            err.not_supported("test"),
            Err(ModelError::OperationNotSupported("test"))
        ));
    }

    #[test]
    fn test_model_err_macro() {
        assert!(matches!(
            model_err!(not_found: "test", "test"),
            ModelError::ModelNotFound {
                provider: "test",
                name: "test"
            }
        ));
        assert!(matches!(
            model_err!(provider_not_found: "test"),
            ModelError::ProviderNotFound("test")
        ));
        assert!(matches!(
            model_err!(already_exists: "test", "test"),
            ModelError::ModelAlreadyExists {
                provider: "test",
                name: "test"
            }
        ));
        assert!(matches!(
            model_err!(invalid_config: "test"),
            ModelError::InvalidConfiguration("test")
        ));
        assert!(matches!(
            model_err!(not_supported: "test"),
            ModelError::OperationNotSupported("test")
        ));
        assert!(matches!(
            model_err!(invalid_input: "test"),
            ModelError::InvalidInput("test")
        ));
        assert!(matches!(
            model_err!(internal: "test"),
            ModelError::Internal("test")
        ));
    }
