use fluent_ai_candle::processing::error::validation::*;
use fluent_ai_candle::processing::error::utils::*;
use fluent_ai_candle::processing::error::*;

#[test]
    fn test_validate_range() {
        assert!(validate_range(5, 0, 10, "test").is_ok());
        assert!(validate_range(-1, 0, 10, "test").is_err());
        assert!(validate_range(11, 0, 10, "test").is_err());
    }

    #[test]
    fn test_validate_finite() {
        assert!(validate_finite(1.0, "test").is_ok());
        assert!(validate_finite(f32::NAN, "test").is_err());
        assert!(validate_finite(f32::INFINITY, "test").is_err());
    }

    #[test]
    fn test_validate_not_empty() {
        let array = [1, 2, 3];
        let empty_array: [i32; 0] = [];
        
        assert!(validate_not_empty(&array, "test").is_ok());
        assert!(validate_not_empty(&empty_array, "test").is_err());
    }

    #[test]
    fn test_validate_array_sizes() {
        let array1 = [1, 2, 3];
        let array2 = ['a', 'b', 'c'];
        let array3 = [1, 2];
        
        assert!(validate_array_sizes(&array1, &array2, "array1", "array2").is_ok());
        assert!(validate_array_sizes(&array1, &array3, "array1", "array3").is_err());
    }

    #[test]
    fn test_validate_probability() {
        assert!(validate_probability(0.5, "test").is_ok());
        assert!(validate_probability(0.0, "test").is_ok());
        assert!(validate_probability(1.0, "test").is_ok());
        assert!(validate_probability(-0.1, "test").is_err());
        assert!(validate_probability(1.1, "test").is_err());
    }

    #[test]
    fn test_validate_temperature() {
        assert!(validate_temperature(1.0).is_ok());
        assert!(validate_temperature(0.1).is_ok());
        assert!(validate_temperature(0.0).is_err());
        assert!(validate_temperature(-1.0).is_err());
    }

    #[test]
    fn test_validate_top_k() {
        assert!(validate_top_k(1).is_ok());
        assert!(validate_top_k(50).is_ok());
        assert!(validate_top_k(0).is_err());
    }

    #[test]
    fn test_should_shutdown() {
        let internal_error = ProcessingError::internal("critical bug");
        let config_error = ProcessingError::configuration("invalid config");
        
        assert!(should_shutdown(&internal_error));
        assert!(!should_shutdown(&config_error));
    }

    #[test]
    fn test_retry_delay() {
        let resource_error = ProcessingError::resource("out of memory");
        let config_error = ProcessingError::configuration("invalid config");
        
        assert!(retry_delay(&resource_error).is_some());
        assert!(retry_delay(&config_error).is_none());
    }

    #[test]
    fn test_with_operation_context() {
        let error = ProcessingError::validation("test error");
        let contextual_error = with_operation_context(error, "test_operation");
        
        assert_eq!(contextual_error.context.operation, "test_operation");
        assert!(contextual_error.context.processor.is_none());
    }

    #[test]
    fn test_with_processor_context() {
        let error = ProcessingError::validation("test error");
        let contextual_error = with_processor_context(error, "test_operation", "test_processor");
        
        assert_eq!(contextual_error.context.operation, "test_operation");
        assert_eq!(contextual_error.context.processor, Some("test_processor".to_string()));
    }
