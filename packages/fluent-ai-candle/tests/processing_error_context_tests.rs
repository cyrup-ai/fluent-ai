use fluent_ai_candle::processing::error::context::*;
use fluent_ai_candle::processing::error::*;

#[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new("test_operation")
            .processor("test_processor")
            .array_size(100)
            .position(42)
            .metadata("key", "value");

        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.processor, Some("test_processor".to_string()));
        assert_eq!(context.array_size, Some(100));
        assert_eq!(context.position, Some(42));
        assert_eq!(context.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_error_context_display() {
        let context = ErrorContext::new("test_operation")
            .processor("test_processor")
            .array_size(100);

        let display = format!("{}", context);
        assert!(display.contains("Operation: test_operation"));
        assert!(display.contains("Processor: test_processor"));
        assert!(display.contains("Array size: 100"));
    }

    #[test]
    fn test_contextual_error_creation() {
        let error = ProcessingError::validation("test error");
        let context = ErrorContext::new("test_operation");
        let contextual_error = ContextualError::new(error, context);

        assert_eq!(contextual_error.category(), ErrorCategory::Validation);
        assert_eq!(contextual_error.severity(), ErrorSeverity::Low);
        assert!(!contextual_error.is_recoverable());
    }

    #[test]
    fn test_contextual_error_display() {
        let error = ProcessingError::validation("test error");
        let context = ErrorContext::new("test_operation");
        let contextual_error = ContextualError::new(error, context);

        let display = format!("{}", contextual_error);
        assert!(display.contains("Input validation error: test error"));
        assert!(display.contains("Operation: test_operation"));
    }
