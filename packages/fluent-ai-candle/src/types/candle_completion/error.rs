//! Completion-specific error types
//!
//! Contains error types specific to completion operations.

use thiserror::Error;

/// Completion-specific error types
#[derive(Debug, Error, Clone)]
pub enum CandleCompletionError {
    /// Model not loaded
    #[error("Model not loaded")]
    ModelNotLoaded,
    
    /// Invalid request parameters
    #[error("Invalid request parameters: {message}")]
    InvalidRequest { message: String },
    
    /// Generation failed
    #[error("Generation failed: {reason}")]
    GenerationFailed { reason: String },
    
    /// Context length exceeded
    #[error("Context length exceeded: {current} > {max}")]
    ContextLengthExceeded { current: u32, max: u32 },
    
    /// Tokenization failed
    #[error("Tokenization failed: {message}")]
    TokenizationFailed { message: String },
    
    /// Model inference error
    #[error("Model inference error: {message}")]
    InferenceError { message: String },
    
    /// Device error (GPU/CPU)
    #[error("Device error: {message}")]
    DeviceError { message: String },
    
    /// Memory allocation error
    #[error("Memory allocation error: {message}")]
    MemoryError { message: String },
    
    /// Timeout error
    #[error("Operation timed out after {seconds} seconds")]
    Timeout { seconds: u64 },
    
    /// Unsupported operation
    #[error("Unsupported operation: {operation}")]
    UnsupportedOperation { operation: String },
    
    /// Invalid parameter
    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },
    
    /// Validation error
    #[error("Validation error: {message}")]
    Validation { message: String },
    
    /// Internal error
    #[error("Internal error: {message}")]
    Internal { message: String },
    
    /// Model loading failed
    #[error("Model loading failed: {message}")]
    ModelLoadingFailed { message: String },
}

/// Extraction-specific error types
#[derive(Debug, Error, Clone)]
pub enum CandleExtractionError {
    /// Failed to parse response
    #[error("Failed to parse response: {message}")]
    ParseError { message: String },
    
    /// Missing required field
    #[error("Missing required field: {field}")]
    MissingField { field: String },
    
    /// Invalid format
    #[error("Invalid format: expected {expected}, got {actual}")]
    InvalidFormat { expected: String, actual: String },
    
    /// Schema validation failed
    #[error("Schema validation failed: {message}")]
    SchemaValidation { message: String },
}

/// Result type for completion operations
pub type CandleCompletionResult<T> = Result<T, CandleCompletionError>;

/// Result type for extraction operations
pub type CandleExtractionResult<T> = Result<T, CandleExtractionError>;

impl CandleCompletionError {
    /// Create a new invalid request error
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }
    
    /// Create a new generation failed error
    pub fn generation_failed(reason: impl Into<String>) -> Self {
        Self::GenerationFailed {
            reason: reason.into(),
        }
    }
    
    /// Create a new inference error
    pub fn inference_error(message: impl Into<String>) -> Self {
        Self::InferenceError {
            message: message.into(),
        }
    }
    
    /// Create a new invalid parameter error
    pub fn invalid_parameter(message: impl Into<String>) -> Self {
        Self::InvalidParameter {
            message: message.into(),
        }
    }
    
    /// Create a new validation error
    pub fn validation_error(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }
    
    /// Create a new internal error
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
    
    /// Create a new model loading failed error
    pub fn model_loading_failed(message: impl Into<String>) -> Self {
        Self::ModelLoadingFailed {
            message: message.into(),
        }
    }
}

impl CandleExtractionError {
    /// Create a new parse error
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::ParseError {
            message: message.into(),
        }
    }
    
    /// Create a new missing field error
    pub fn missing_field(field: impl Into<String>) -> Self {
        Self::MissingField {
            field: field.into(),
        }
    }
}

// From trait implementations for error conversion

impl From<crate::model::error::ModelError> for CandleCompletionError {
    fn from(err: crate::model::error::ModelError) -> Self {
        match err {
            crate::model::error::ModelError::Validation(validation_err) => {
                Self::Validation { message: validation_err.to_string() }
            }
            crate::model::error::ModelError::Io(io_err) => {
                Self::Internal { message: format!("IO error: {}", io_err) }
            }
            crate::model::error::ModelError::Serialization(serde_err) => {
                Self::Internal { message: format!("Serialization error: {}", serde_err) }
            }
            crate::model::error::ModelError::Candle { message } => {
                Self::InferenceError { message }
            }
            crate::model::error::ModelError::InitializationFailed { reason } => {
                Self::ModelLoadingFailed { message: reason }
            }
            crate::model::error::ModelError::Inference { message } => {
                Self::InferenceError { message }
            }
            crate::model::error::ModelError::Memory { message } => {
                Self::MemoryError { message }
            }
            crate::model::error::ModelError::Device { message } => {
                Self::DeviceError { message }
            }
            crate::model::error::ModelError::InvalidConfiguration(msg) => {
                Self::InvalidRequest { message: msg.to_string() }
            }
            crate::model::error::ModelError::OperationNotSupported(msg) => {
                Self::UnsupportedOperation { operation: msg.to_string() }
            }
            crate::model::error::ModelError::ModelNotFound { provider, name } => {
                Self::ModelLoadingFailed { 
                    message: format!("Model not found: {}::{}", provider, name) 
                }
            }
            crate::model::error::ModelError::ModelAlreadyExists { name } => {
                Self::InvalidRequest { 
                    message: format!("Model already exists: {}", name) 
                }
            }
        }
    }
}


/// Convert ModelError to CandleCompletionError


impl From<crate::types::candle_model::error::ModelError> for CandleCompletionError {
    fn from(error: crate::types::candle_model::error::ModelError) -> Self {
        match error {
            crate::types::candle_model::error::ModelError::ModelNotFound { provider, name } => {
                Self::InvalidRequest {
                    message: format!("Model not found: {}:{}", provider, name),
                }
            }
            crate::types::candle_model::error::ModelError::ModelAlreadyExists { provider, name } => {
                Self::InvalidRequest {
                    message: format!("Model already exists: {}:{}", provider, name),
                }
            }
            crate::types::candle_model::error::ModelError::ProviderNotFound(provider) => {
                Self::InvalidRequest {
                    message: format!("Provider not found: {}", provider),
                }
            }
            crate::types::candle_model::error::ModelError::InvalidConfiguration(msg) => {
                Self::InvalidParameter {
                    message: msg.to_string(),
                }
            }
            crate::types::candle_model::error::ModelError::OperationNotSupported(msg) => {
                Self::UnsupportedOperation {
                    operation: msg.to_string(),
                }
            }
            crate::types::candle_model::error::ModelError::InvalidInput(msg) => {
                Self::InvalidParameter {
                    message: msg.to_string(),
                }
            }
            crate::types::candle_model::error::ModelError::Internal(msg) => {
                Self::Internal {
                    message: msg.to_string(),
                }
            }
        }
    }
}
