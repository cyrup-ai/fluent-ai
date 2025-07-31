/// Error types for model operations with comprehensive context and zero-allocation performance
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model not found: {provider}:{name}")]
    ModelNotFound { 
        provider: String, 
        name: String 
    },
    
    #[error("Model already exists: {provider}:{name}")]
    ModelAlreadyExists { 
        provider: String, 
        name: String 
    },
    
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),
    
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("HTTP request failed: {0}")]
    HttpError(String),
    
    #[error("JSON parsing failed: {0}")]
    JsonError(String),
    
    #[error("Network timeout: operation took longer than {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    
    #[error("Rate limit exceeded: retry after {retry_after_seconds}s")]
    RateLimit { retry_after_seconds: u64 },
    
    #[error("Authentication failed: {reason}")]
    AuthenticationFailed { reason: String },
    
    #[error("Service unavailable: {provider} is currently down")]
    ServiceUnavailable { provider: String },
}

/// Result type for model operations
pub type Result<T> = std::result::Result<T, ModelError>;

impl ModelError {
    /// Check if error is retryable
    #[inline]
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            ModelError::HttpError(_) 
            | ModelError::Timeout { .. }
            | ModelError::RateLimit { .. }
            | ModelError::ServiceUnavailable { .. }
        )
    }
    
    /// Get recommended retry delay in milliseconds
    #[inline]
    pub const fn retry_delay_ms(&self) -> Option<u64> {
        match self {
            ModelError::RateLimit { retry_after_seconds } => {
                Some(retry_after_seconds * 1000)
            }
            ModelError::HttpError(_) | ModelError::ServiceUnavailable { .. } => {
                Some(1000) // 1 second base delay
            }
            ModelError::Timeout { .. } => {
                Some(5000) // 5 second delay for timeouts
            }
            _ => None,
        }
    }
    
    /// Create a model not found error with zero allocation when possible
    #[inline]
    pub fn model_not_found(provider: impl Into<String>, name: impl Into<String>) -> Self {
        Self::ModelNotFound {
            provider: provider.into(),
            name: name.into(),
        }
    }
    
    /// Create a provider not found error
    #[inline]
    pub fn provider_not_found(provider: impl Into<String>) -> Self {
        Self::ProviderNotFound(provider.into())
    }
    
    /// Create an invalid configuration error
    #[inline]
    pub fn invalid_configuration(message: impl Into<String>) -> Self {
        Self::InvalidConfiguration(message.into())
    }
    
    /// Create a validation failed error
    #[inline]
    pub fn validation_failed(message: impl Into<String>) -> Self {
        Self::ValidationFailed(message.into())
    }
}