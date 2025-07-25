//! Error Validation and Utility Functions Module
//!
//! Provides utility functions for error handling, validation, and error management.
//! Zero-allocation validation with comprehensive helper functions for common error scenarios.

use super::error_types::ProcessingError;
use super::context::{ErrorContext, ContextualError};
use super::classification::{ErrorCategory, ErrorSeverity};
use super::conversion::ProcessingResult;

/// Utility functions for error handling
pub mod utils {
    use super::*;

    /// Validate configuration parameter range
    #[inline(always)]
    pub fn validate_range<T>(value: T, min: T, max: T, param_name: &str) -> ProcessingResult<()>
    where
        T: PartialOrd + std::fmt::Display,
    {
        if value < min || value > max {
            return Err(ProcessingError::configuration(format!(
                "{} value {} is outside valid range [{}, {}]",
                param_name, value, min, max
            )));
        }
        Ok(())
    }

    /// Validate that a value is finite
    #[inline(always)]
    pub fn validate_finite(value: f32, param_name: &str) -> ProcessingResult<()> {
        if !value.is_finite() {
            return Err(ProcessingError::numerical(format!(
                "{} value {} is not finite",
                param_name, value
            )));
        }
        Ok(())
    }

    /// Validate array is not empty
    #[inline(always)]
    pub fn validate_not_empty<T>(array: &[T], array_name: &str) -> ProcessingResult<()> {
        if array.is_empty() {
            return Err(ProcessingError::validation(format!(
                "{} array is empty",
                array_name
            )));
        }
        Ok(())
    }

    /// Validate array sizes match
    #[inline(always)]
    pub fn validate_array_sizes<T, U>(
        array1: &[T],
        array2: &[U],
        name1: &str,
        name2: &str,
    ) -> ProcessingResult<()> {
        if array1.len() != array2.len() {
            return Err(ProcessingError::validation(format!(
                "{} size {} does not match {} size {}",
                name1,
                array1.len(),
                name2,
                array2.len()
            )));
        }
        Ok(())
    }

    /// Convert error to contextual error with operation info
    #[inline(always)]
    pub fn with_operation_context(error: ProcessingError, operation: &str) -> ContextualError {
        let context = ErrorContext::new(operation);
        ContextualError::new(error, context)
    }

    /// Convert error to contextual error with processor info
    #[inline(always)]
    pub fn with_processor_context(
        error: ProcessingError,
        operation: &str,
        processor: &str,
    ) -> ContextualError {
        let context = ErrorContext::new(operation).processor(processor);
        ContextualError::new(error, context)
    }

    /// Check if error should trigger system shutdown
    #[inline(always)]
    pub fn should_shutdown(error: &ProcessingError) -> bool {
        matches!(error, ProcessingError::InternalError(_))
            && error.severity() == ErrorSeverity::Critical
    }

    /// Get retry delay based on error type
    #[inline(always)]
    pub fn retry_delay(error: &ProcessingError) -> Option<std::time::Duration> {
        if !error.is_recoverable() {
            return None;
        }

        match error.category() {
            ErrorCategory::Resource => Some(std::time::Duration::from_millis(100)),
            ErrorCategory::External => Some(std::time::Duration::from_millis(50)),
            ErrorCategory::Numerical => Some(std::time::Duration::from_millis(10)),
            ErrorCategory::Context => Some(std::time::Duration::from_millis(1)),
            _ => None,
        }
    }

    /// Validate probability value (0.0 to 1.0)
    #[inline(always)]
    pub fn validate_probability(value: f32, param_name: &str) -> ProcessingResult<()> {
        validate_finite(value, param_name)?;
        validate_range(value, 0.0, 1.0, param_name)?;
        Ok(())
    }

    /// Validate temperature value (> 0.0)
    #[inline(always)]
    pub fn validate_temperature(value: f32) -> ProcessingResult<()> {
        validate_finite(value, "temperature")?;
        if value <= 0.0 {
            return Err(ProcessingError::configuration(format!(
                "Temperature {} must be greater than 0.0",
                value
            )));
        }
        Ok(())
    }

    /// Validate top-k value (> 0)
    #[inline(always)]
    pub fn validate_top_k(value: usize) -> ProcessingResult<()> {
        if value == 0 {
            return Err(ProcessingError::configuration(
                "Top-k value must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate repetition penalty (> 0.0)
    #[inline(always)]
    pub fn validate_repetition_penalty(value: f32) -> ProcessingResult<()> {
        validate_finite(value, "repetition_penalty")?;
        if value <= 0.0 {
            return Err(ProcessingError::configuration(format!(
                "Repetition penalty {} must be greater than 0.0",
                value
            )));
        }
        Ok(())
    }

    /// Validate buffer capacity
    #[inline(always)]
    pub fn validate_buffer_capacity(capacity: usize, required: usize) -> ProcessingResult<()> {
        if capacity < required {
            return Err(ProcessingError::buffer_overflow(format!(
                "Buffer capacity {} is less than required {}",
                capacity, required
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;
    use super::*;

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
}