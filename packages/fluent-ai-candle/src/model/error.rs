//! Model-related error types
//!
//! Contains error types for model operations and validation.

use thiserror::Error;

/// Validation errors for model operations
#[derive(Debug, Error, Clone)]
pub enum ValidationError {
    /// Invalid model configuration
    #[error("Invalid model configuration: {message}")]
    InvalidConfig { message: String },
    
    /// Missing required field
    #[error("Missing required field: {field}")]
    MissingField { field: String },
    
    /// Invalid parameter value
    #[error("Invalid parameter value for {parameter}: {value}")]
    InvalidParameter { parameter: String, value: String },
    
    /// Model not found
    #[error("Model not found: {model_name}")]
    ModelNotFound { model_name: String },
    
    /// Unsupported model type
    #[error("Unsupported model type: {model_type}")]
    UnsupportedModelType { model_type: String },
    
    /// Invalid tensor shape
    #[error("Invalid tensor shape: expected {expected}, got {actual}")]
    InvalidTensorShape { expected: String, actual: String },
    
    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource} ({limit})")]
    ResourceLimitExceeded { resource: String, limit: String },
}

/// Result type for validation operations
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Model loading and operation errors
#[derive(Debug, Error)]
pub enum ModelError {
    /// Validation error
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),
    
    /// IO error during model loading
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Candle framework error
    #[error("Candle error: {message}")]
    Candle { message: String },
    
    /// Model initialization failed
    #[error("Model initialization failed: {reason}")]
    InitializationFailed { reason: String },
    
    /// Inference error
    #[error("Inference error: {message}")]
    Inference { message: String },
    
    /// Memory allocation error
    #[error("Memory allocation error: {message}")]
    Memory { message: String },
    
    /// Device error (GPU/CPU)
    #[error("Device error: {message}")]
    Device { message: String },
}

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

impl ValidationError {
    /// Create a new invalid config error
    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }
    
    /// Create a new missing field error
    pub fn missing_field(field: impl Into<String>) -> Self {
        Self::MissingField {
            field: field.into(),
        }
    }
    
    /// Create a new invalid parameter error
    pub fn invalid_parameter(parameter: impl Into<String>, value: impl Into<String>) -> Self {
        Self::InvalidParameter {
            parameter: parameter.into(),
            value: value.into(),
        }
    }
}

impl ModelError {
    /// Create a new Candle error
    pub fn candle(message: impl Into<String>) -> Self {
        Self::Candle {
            message: message.into(),
        }
    }
    
    /// Create a new initialization error
    pub fn initialization_failed(reason: impl Into<String>) -> Self {
        Self::InitializationFailed {
            reason: reason.into(),
        }
    }
    
    /// Create a new inference error
    pub fn inference(message: impl Into<String>) -> Self {
        Self::Inference {
            message: message.into(),
        }
    }
}
