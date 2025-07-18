//! Comprehensive error handling for VertexAI client
//!
//! Zero-allocation error types with detailed context and recovery information.

use thiserror::Error;

/// Result type for VertexAI operations
pub type VertexAIResult<T> = Result<T, VertexAIError>;

/// Comprehensive VertexAI error types with detailed context
#[derive(Error, Debug)]
pub enum VertexAIError {
    /// HTTP transport errors from fluent_ai_http3
    #[error("HTTP request failed: {0}")]
    Http(#[from] fluent_ai_http3::HttpError),

    /// Authentication and authorization errors
    #[error("Authentication failed: {message}")]
    Auth { message: String },

    /// JWT token creation or validation errors
    #[error("JWT token error: {details}")]
    JwtToken { details: String },

    /// Service account configuration errors
    #[error("Service account configuration invalid: {reason}")]
    ServiceAccount { reason: String },

    /// OAuth2 token refresh errors
    #[error("Token refresh failed: {error_code} - {description}")]
    TokenRefresh { error_code: String, description: String },

    /// Model-specific errors
    #[error("Model '{model}' not found or not available")]
    ModelNotFound { model: String },

    /// Model capability mismatches
    #[error("Model '{model}' does not support {capability}")]
    ModelCapability { model: String, capability: String },

    /// Project access and configuration errors  
    #[error("Project access denied: {project} in region {region}")]
    ProjectAccess { project: String, region: String },

    /// Region availability errors
    #[error("Region '{region}' not supported for service {service}")]
    RegionNotSupported { region: String, service: String },

    /// Quota and rate limiting errors
    #[error("Quota exceeded: {quota_type}, retry after {retry_after}s")]
    QuotaExceeded { quota_type: String, retry_after: u64 },

    /// Rate limiting with backoff information
    #[error("Rate limited: {requests_per_minute} RPM exceeded, backoff {backoff_ms}ms")]
    RateLimit { requests_per_minute: u32, backoff_ms: u64 },

    /// Request validation errors
    #[error("Request validation failed: {field} - {reason}")]
    RequestValidation { field: String, reason: String },

    /// Response parsing and deserialization errors
    #[error("Response parsing failed: {context}")]
    ResponseParsing { context: String },

    /// Streaming errors
    #[error("Streaming error: {source}")]
    Streaming { source: String },

    /// Server-Sent Events parsing errors
    #[error("SSE parsing error at line {line}: {details}")]
    SseParsing { line: u32, details: String },

    /// Safety filter errors
    #[error("Content blocked by safety filters: {reason}")]
    SafetyFilter { reason: String },

    /// Content policy violations
    #[error("Content policy violation: {policy} - {details}")]
    ContentPolicy { policy: String, details: String },

    /// Configuration errors
    #[error("Configuration error: {parameter} - {issue}")]
    Config { parameter: String, issue: String },

    /// Timeout errors with context
    #[error("Operation timeout: {operation} exceeded {timeout_ms}ms")]
    Timeout { operation: String, timeout_ms: u64 },

    /// Circuit breaker open state
    #[error("Circuit breaker open for VertexAI, too many failures")]
    CircuitBreakerOpen,

    /// Resource not found errors
    #[error("Resource not found: {resource_type} '{resource_id}'")]
    ResourceNotFound { resource_type: String, resource_id: String },

    /// Invalid API key or credentials
    #[error("Invalid credentials: {credential_type}")]
    InvalidCredentials { credential_type: String },

    /// Server errors with detailed context
    #[error("VertexAI server error {status_code}: {error_message}")]
    ServerError { status_code: u16, error_message: String },

    /// Internal client errors
    #[error("Internal client error: {context}")]
    Internal { context: String },

    /// JSON serialization/deserialization errors
    #[error("JSON processing error: {operation} - {details}")]
    Json { operation: String, details: String },

    /// Network connectivity errors
    #[error("Network error: {details}")]
    Network { details: String },

    /// Unsupported operation errors
    #[error("Unsupported operation: {operation} for model {model}")]
    UnsupportedOperation { operation: String, model: String },
}

impl VertexAIError {
    /// Check if error is retryable with exponential backoff
    pub fn is_retryable(&self) -> bool {
        match self {
            VertexAIError::Http(_) => true,
            VertexAIError::TokenRefresh { .. } => true,
            VertexAIError::QuotaExceeded { .. } => true,
            VertexAIError::RateLimit { .. } => true,
            VertexAIError::Timeout { .. } => true,
            VertexAIError::Network { .. } => true,
            VertexAIError::ServerError { status_code, .. } => *status_code >= 500,
            _ => false,
        }
    }

    /// Get suggested retry delay in milliseconds
    pub fn retry_delay_ms(&self) -> Option<u64> {
        match self {
            VertexAIError::QuotaExceeded { retry_after, .. } => Some(*retry_after * 1000),
            VertexAIError::RateLimit { backoff_ms, .. } => Some(*backoff_ms),
            VertexAIError::TokenRefresh { .. } => Some(1000),
            VertexAIError::Timeout { .. } => Some(2000),
            VertexAIError::Network { .. } => Some(1000),
            VertexAIError::ServerError { .. } => Some(5000),
            _ => None,
        }
    }

    /// Check if error requires authentication refresh
    pub fn requires_auth_refresh(&self) -> bool {
        matches!(
            self,
            VertexAIError::Auth { .. } |
            VertexAIError::JwtToken { .. } |
            VertexAIError::TokenRefresh { .. } |
            VertexAIError::InvalidCredentials { .. }
        )
    }

    /// Get error category for metrics and monitoring
    pub fn category(&self) -> &'static str {
        match self {
            VertexAIError::Http(_) => "transport",
            VertexAIError::Auth { .. } |
            VertexAIError::JwtToken { .. } |
            VertexAIError::ServiceAccount { .. } |
            VertexAIError::TokenRefresh { .. } |
            VertexAIError::InvalidCredentials { .. } => "auth",
            VertexAIError::ModelNotFound { .. } |
            VertexAIError::ModelCapability { .. } => "model",
            VertexAIError::ProjectAccess { .. } |
            VertexAIError::RegionNotSupported { .. } => "access",
            VertexAIError::QuotaExceeded { .. } |
            VertexAIError::RateLimit { .. } => "throttling",
            VertexAIError::RequestValidation { .. } |
            VertexAIError::Config { .. } => "validation",
            VertexAIError::ResponseParsing { .. } |
            VertexAIError::SseParsing { .. } |
            VertexAIError::Json { .. } => "parsing",
            VertexAIError::Streaming { .. } => "streaming",
            VertexAIError::SafetyFilter { .. } |
            VertexAIError::ContentPolicy { .. } => "safety",
            VertexAIError::Timeout { .. } => "timeout",
            VertexAIError::CircuitBreakerOpen => "circuit_breaker",
            VertexAIError::ResourceNotFound { .. } => "not_found",
            VertexAIError::ServerError { .. } => "server",
            VertexAIError::Internal { .. } => "internal",
            VertexAIError::Network { .. } => "network",
            VertexAIError::UnsupportedOperation { .. } => "unsupported",
        }
    }
}

impl From<serde_json::Error> for VertexAIError {
    fn from(err: serde_json::Error) -> Self {
        VertexAIError::Json {
            operation: "serde_json".to_string(),
            details: err.to_string(),
        }
    }
}

impl From<std::io::Error> for VertexAIError {
    fn from(err: std::io::Error) -> Self {
        VertexAIError::Network {
            details: err.to_string(),
        }
    }
}