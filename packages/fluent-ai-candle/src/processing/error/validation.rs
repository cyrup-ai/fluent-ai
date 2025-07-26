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
            _ => None}
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
