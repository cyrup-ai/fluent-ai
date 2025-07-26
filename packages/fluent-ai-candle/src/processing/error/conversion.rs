//! Error Conversion Module
//!
//! Provides type aliases and From trait implementations for integration with existing error systems.
//! Enables seamless conversion between ProcessingError and external error types.

use super::error_types::ProcessingError;

/// Result type alias for processing operations
pub type ProcessingResult<T> = Result<T, ProcessingError>;

/// Integration with existing Candle error system
impl From<candle_core::Error> for ProcessingError {
    fn from(err: candle_core::Error) -> Self {
        Self::ExternalError(format!("Candle error: {}", err))
    }
}

/// Integration with sampling error system (for migration)
impl From<crate::sampling::SamplingError> for ProcessingError {
    fn from(err: crate::sampling::SamplingError) -> Self {
        match &err {
            crate::sampling::SamplingError::InvalidTemperature(temp) => {
                Self::InvalidConfiguration(format!("Invalid temperature: {}", temp))
            }
            crate::sampling::SamplingError::InvalidTopP(p) => {
                Self::InvalidConfiguration(format!("Invalid top-p: {}", p))
            }
            crate::sampling::SamplingError::InvalidTopK(k) => {
                Self::InvalidConfiguration(format!("Invalid top-k: {}", k))
            }
            crate::sampling::SamplingError::InvalidRepetitionPenalty(penalty) => {
                Self::InvalidConfiguration(format!("Invalid repetition penalty: {}", penalty))
            }
            crate::sampling::SamplingError::TensorError(msg) => {
                Self::ExternalError(format!("Tensor error: {}", msg))
            }
            crate::sampling::SamplingError::NumericalInstability(msg) => {
                Self::NumericalError(msg.clone())
            }
            crate::sampling::SamplingError::EmptyVocabulary => {
                Self::ValidationError("Empty vocabulary".to_string())
            }
            crate::sampling::SamplingError::EmptyLogits => {
                Self::ValidationError("Empty logits".to_string())
            }
            crate::sampling::SamplingError::ProcessingFailed(msg) => {
                Self::InternalError(msg.clone())
            }
            crate::sampling::SamplingError::ProcessorChainError(msg) => {
                Self::ProcessorChainError(msg.clone())
            }
        }
    }
}
