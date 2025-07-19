//! Test module for completion functionality

mod test_utils {
    use super::*;
    
    pub fn create_test_completion_params() -> ValidationResult<CompletionParams> {
        CompletionParams::new()
            .with_temperature(0.7)
            .and_then(|p| p.with_max_tokens(Some(std::num::NonZeroU64::new(100).unwrap())))
    }
    
    pub fn create_test_completion_request() -> Result<CompletionRequest<'static>, CompletionRequestError> {
        CompletionRequest::builder()
            .system_prompt("Test system prompt")
            .chat_history(crate::ZeroOneOrMany::None)
            .documents(crate::ZeroOneOrMany::None)
            .temperature(0.7)?
            .max_tokens(Some(std::num::NonZeroU64::new(100).unwrap()))
            .build()
    }
    
    pub fn create_test_completion_response() -> CompletionResponse<'static> {
        CompletionResponse::new("Test completion response", "test-model")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::*;
    
    #[test]
    fn test_completion_params_validation() {
        // Test valid temperature
        let params = create_test_completion_params().unwrap();
        assert!((0.7 - params.temperature).abs() < f64::EPSILON);
        
        // Test invalid temperature (too low)
        assert!(CompletionParams::new().with_temperature(-0.1).is_err());
        
        // Test invalid temperature (too high)
        assert!(CompletionParams::new().with_temperature(2.1).is_err());
    }
    
    #[test]
    fn test_completion_request_builder() {
        let request = create_test_completion_request().unwrap();
        assert_eq!(request.system_prompt, Some("Test system prompt".into()));
        assert!((0.7 - request.temperature).abs() < f64::EPSILON);
    }
    
    #[test]
    fn test_completion_response() {
        let response = create_test_completion_response();
        assert_eq!(response.text(), "Test completion response");
        assert_eq!(response.model(), "test-model");
    }
    
    #[test]
    fn test_compact_completion_response() {
        let response = create_test_completion_response();
        let compact = response.into_compact();
        let back = compact.into_standard();
        
        assert_eq!(back.text(), "Test completion response");
        assert_eq!(back.model(), "test-model");
    }
}
