use fluent_ai_candle::processing::error::conversion::*;
use fluent_ai_candle::processing::error::*;

use crate::sampling::SamplingError;

    #[test]
    fn test_processing_result_type_alias() {
        let success: ProcessingResult<i32> = Ok(42);
        let failure: ProcessingResult<i32> = Err(ProcessingError::validation("test error"));
        
        assert!(success.is_ok());
        assert!(failure.is_err());
    }

    #[test] 
    fn test_sampling_error_conversion() {
        let sampling_error = SamplingError::InvalidTemperature(0.0);
        let processing_error: ProcessingError = sampling_error.into();
        
        match processing_error {
            ProcessingError::InvalidConfiguration(msg) => {
                assert!(msg.contains("Invalid temperature"));
            }
            _ => panic!("Expected InvalidConfiguration error")}
    }

    #[test]
    fn test_empty_vocabulary_conversion() {
        let sampling_error = SamplingError::EmptyVocabulary;
        let processing_error: ProcessingError = sampling_error.into();
        
        match processing_error {
            ProcessingError::ValidationError(msg) => {
                assert_eq!(msg, "Empty vocabulary");
            }
            _ => panic!("Expected ValidationError")}
    }
