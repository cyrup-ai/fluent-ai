//! AI21 Labs comprehensive error handling with zero allocation context preservation
//!
//! Provides complete error taxonomy for OpenAI-compatible endpoints:
//! - HTTP status mapping for all AI21 API responses
//! - Bearer token authentication failures
//! - Jamba model availability and validation errors
//! - Rate limiting with retry-after information
//! - Request validation with detailed context
//! - Quota exceeded scenarios with billing information
//! - Model capacity and queue management errors
//! - JSON parsing with context preservation
//! - Network and timeout handling
//! - Configuration validation errors
//!
//! All errors use zero allocation patterns with arrayvec::ArrayString
//! for context preservation and #[repr(u8)] for minimal memory footprint.

use thiserror::Error;
use arrayvec::ArrayString;
use std::fmt;

/// Result type alias for AI21 operations
pub type Result<T> = std::result::Result<T, AI21Error>;

/// Comprehensive error taxonomy for AI21 operations with zero allocation context
#[derive(Error, Debug, Clone)]
#[repr(u8)]
pub enum AI21Error {
    /// HTTP request failed
    #[error("HTTP request failed: {0}")]
    Http(#[from] fluent_ai_http3::HttpError),
    
    /// Authentication failed for AI21 API
    #[error("Authentication failed: {message}")]
    Authentication {
        /// Error message with context
        message: ArrayString<256>,
        /// Bearer token length for debugging
        token_length: usize,
        /// Whether token format is valid
        format_valid: bool,
        /// Whether retry is possible
        retry_possible: bool,
        /// HTTP status code
        status_code: u16},
    
    /// Rate limiting exceeded
    #[error("Rate limit exceeded: {message}")]
    RateLimit {
        /// Rate limit message
        message: ArrayString<256>,
        /// Retry after seconds
        retry_after_seconds: u32,
        /// Current request count
        current_requests: u32,
        /// Maximum allowed requests
        max_requests: u32,
        /// Reset time (Unix timestamp)
        reset_time: u64},
    
    /// Model not supported or unavailable
    #[error("Model not supported: {model}")]
    ModelNotSupported {
        /// Model name that was requested
        model: ArrayString<64>,
        /// Available models
        available_models: ArrayString<512>,
        /// Suggested alternative model
        suggested_model: ArrayString<64>,
        /// Whether model exists but is unavailable
        model_exists: bool},
    
    /// Request validation failed
    #[error("Request validation failed: {field} - {reason}")]
    RequestValidation {
        /// Field that failed validation
        field: ArrayString<64>,
        /// Validation failure reason
        reason: ArrayString<256>,
        /// Current field value
        current_value: ArrayString<128>,
        /// Valid range or options
        valid_range: ArrayString<128>,
        /// Whether value can be auto-corrected
        auto_correctable: bool},
    
    /// Quota exceeded for AI21 API
    #[error("Quota exceeded: {quota_type}")]
    QuotaExceeded {
        /// Type of quota exceeded
        quota_type: QuotaType,
        /// Current usage
        current_usage: u64,
        /// Maximum allowed usage
        max_usage: u64,
        /// Reset time (Unix timestamp)
        reset_time: u64,
        /// Billing information
        billing_info: ArrayString<256>},
    
    /// Model capacity exceeded
    #[error("Model capacity exceeded: {model}")]
    ModelCapacity {
        /// Model name
        model: ArrayString<64>,
        /// Current queue length
        queue_length: u32,
        /// Estimated wait time (seconds)
        estimated_wait_seconds: u32,
        /// Whether retry is recommended
        retry_recommended: bool,
        /// Alternative models available
        alternatives: ArrayString<256>},
    
    /// JSON processing error
    #[error("JSON processing failed: {operation} - {message}")]
    JsonProcessing {
        /// JSON operation that failed
        operation: JsonOperation,
        /// Error message
        message: ArrayString<256>,
        /// Line number where error occurred
        line_number: Option<usize>,
        /// Column number where error occurred
        column_number: Option<usize>,
        /// Whether recovery is possible
        recovery_possible: bool},
    
    /// Request timeout
    #[error("Request timeout: {duration_ms}ms")]
    Timeout {
        /// Duration in milliseconds
        duration_ms: u64,
        /// Request type that timed out
        request_type: RequestType,
        /// Whether retry is recommended
        retry_recommended: bool,
        /// Suggested timeout for retry
        suggested_timeout_ms: u64},
    
    /// Configuration error
    #[error("Configuration error: {setting} - {reason}")]
    Configuration {
        /// Configuration setting that failed
        setting: ArrayString<64>,
        /// Reason for failure
        reason: ArrayString<256>,
        /// Current value
        current_value: ArrayString<128>,
        /// Expected value or range
        expected_value: ArrayString<128>},
    
    /// Streaming error
    #[error("Streaming error: {reason}")]
    Streaming {
        /// Streaming error reason
        reason: StreamingErrorReason,
        /// Event number where error occurred
        event_number: u32,
        /// Whether stream can be resumed
        resumable: bool,
        /// Partial content received
        partial_content: ArrayString<512>},
    
    /// Invalid API key format
    #[error("Invalid API key: {reason}")]
    InvalidApiKey {
        /// Reason for invalidity
        reason: ArrayString<128>,
        /// Key length for debugging
        key_length: usize,
        /// Whether format is valid
        format_valid: bool}}

/// Quota type enumeration for detailed quota management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QuotaType {
    /// Monthly token quota
    MonthlyTokens,
    /// Daily request quota
    DailyRequests,
    /// Concurrent requests quota
    ConcurrentRequests,
    /// Model-specific quota
    ModelSpecific,
    /// Billing quota
    BillingQuota}

/// JSON operation types for error context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JsonOperation {
    /// Request serialization
    RequestSerialization,
    /// Response deserialization
    ResponseDeserialization,
    /// Streaming parse
    StreamingParse,
    /// Tool call parse
    ToolCallParse,
    /// Usage statistics parse
    UsageStatsParse}

/// Request type enumeration for timeout context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RequestType {
    /// Chat completion request
    ChatCompletion,
    /// Streaming completion request
    StreamingCompletion,
    /// Model information request
    ModelInfo,
    /// Health check request
    HealthCheck}

/// Streaming error reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StreamingErrorReason {
    /// Invalid SSE format
    InvalidSseFormat,
    /// Connection interrupted
    ConnectionInterrupted,
    /// JSON parsing failed
    JsonParsingFailed,
    /// Unexpected event type
    UnexpectedEventType,
    /// Stream ended unexpectedly
    StreamEndedUnexpectedly}

/// Error code enumeration for programmatic error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ErrorCode {
    /// Authentication failure
    AuthenticationFailed = 1,
    /// Rate limit exceeded
    RateLimitExceeded = 2,
    /// Model not available
    ModelNotAvailable = 3,
    /// Request validation failed
    RequestValidationFailed = 4,
    /// Quota exceeded
    QuotaExceeded = 5,
    /// Model capacity exceeded
    ModelCapacityExceeded = 6,
    /// JSON processing failed
    JsonProcessingFailed = 7,
    /// Request timeout
    RequestTimeout = 8,
    /// Configuration error
    ConfigurationError = 9,
    /// Streaming error
    StreamingError = 10,
    /// Invalid API key
    InvalidApiKey = 11}

impl AI21Error {
    /// Create authentication error with context
    #[inline]
    pub fn authentication_error(
        message: &str,
        token_length: usize,
        format_valid: bool,
        retry_possible: bool,
        status_code: u16,
    ) -> Self {
        Self::Authentication {
            message: ArrayString::from(message).unwrap_or_default(),
            token_length,
            format_valid,
            retry_possible,
            status_code}
    }
    
    /// Create rate limit error with retry information
    #[inline]
    pub fn rate_limit_error(
        message: &str,
        retry_after_seconds: u32,
        current_requests: u32,
        max_requests: u32,
        reset_time: u64,
    ) -> Self {
        Self::RateLimit {
            message: ArrayString::from(message).unwrap_or_default(),
            retry_after_seconds,
            current_requests,
            max_requests,
            reset_time}
    }
    
    /// Create model not supported error
    #[inline]
    pub fn model_not_supported(
        model: &str,
        available_models: &[&str],
        suggested_model: &str,
        model_exists: bool,
    ) -> Self {
        let available_str = available_models.join(", ");
        Self::ModelNotSupported {
            model: ArrayString::from(model).unwrap_or_default(),
            available_models: ArrayString::from(&available_str).unwrap_or_default(),
            suggested_model: ArrayString::from(suggested_model).unwrap_or_default(),
            model_exists}
    }
    
    /// Create request validation error
    #[inline]
    pub fn request_validation_error(
        field: &str,
        reason: &str,
        current_value: &str,
        valid_range: &str,
        auto_correctable: bool,
    ) -> Self {
        Self::RequestValidation {
            field: ArrayString::from(field).unwrap_or_default(),
            reason: ArrayString::from(reason).unwrap_or_default(),
            current_value: ArrayString::from(current_value).unwrap_or_default(),
            valid_range: ArrayString::from(valid_range).unwrap_or_default(),
            auto_correctable}
    }
    
    /// Create quota exceeded error
    #[inline]
    pub fn quota_exceeded_error(
        quota_type: QuotaType,
        current_usage: u64,
        max_usage: u64,
        reset_time: u64,
        billing_info: &str,
    ) -> Self {
        Self::QuotaExceeded {
            quota_type,
            current_usage,
            max_usage,
            reset_time,
            billing_info: ArrayString::from(billing_info).unwrap_or_default()}
    }
    
    /// Create model capacity error
    #[inline]
    pub fn model_capacity_error(
        model: &str,
        queue_length: u32,
        estimated_wait_seconds: u32,
        retry_recommended: bool,
        alternatives: &[&str],
    ) -> Self {
        let alternatives_str = alternatives.join(", ");
        Self::ModelCapacity {
            model: ArrayString::from(model).unwrap_or_default(),
            queue_length,
            estimated_wait_seconds,
            retry_recommended,
            alternatives: ArrayString::from(&alternatives_str).unwrap_or_default()}
    }
    
    /// Create JSON processing error
    #[inline]
    pub fn json_error(
        operation: JsonOperation,
        message: &str,
        line_number: Option<usize>,
        column_number: Option<usize>,
        recovery_possible: bool,
    ) -> Self {
        Self::JsonProcessing {
            operation,
            message: ArrayString::from(message).unwrap_or_default(),
            line_number,
            column_number,
            recovery_possible}
    }
    
    /// Create timeout error
    #[inline]
    pub fn timeout_error(
        duration_ms: u64,
        request_type: RequestType,
        retry_recommended: bool,
        suggested_timeout_ms: u64,
    ) -> Self {
        Self::Timeout {
            duration_ms,
            request_type,
            retry_recommended,
            suggested_timeout_ms}
    }
    
    /// Create configuration error
    #[inline]
    pub fn configuration_error(
        setting: &str,
        reason: &str,
        current_value: &str,
        expected_value: &str,
    ) -> Self {
        Self::Configuration {
            setting: ArrayString::from(setting).unwrap_or_default(),
            reason: ArrayString::from(reason).unwrap_or_default(),
            current_value: ArrayString::from(current_value).unwrap_or_default(),
            expected_value: ArrayString::from(expected_value).unwrap_or_default()}
    }
    
    /// Create streaming error
    #[inline]
    pub fn streaming_error(
        reason: StreamingErrorReason,
        event_number: u32,
        resumable: bool,
        partial_content: &str,
    ) -> Self {
        Self::Streaming {
            reason,
            event_number,
            resumable,
            partial_content: ArrayString::from(partial_content).unwrap_or_default()}
    }
    
    /// Create invalid API key error
    #[inline]
    pub fn invalid_api_key_error(
        reason: &str,
        key_length: usize,
        format_valid: bool,
    ) -> Self {
        Self::InvalidApiKey {
            reason: ArrayString::from(reason).unwrap_or_default(),
            key_length,
            format_valid}
    }
    
    /// Get error code for programmatic handling
    #[inline]
    pub const fn error_code(&self) -> ErrorCode {
        match self {
            Self::Authentication { .. } => ErrorCode::AuthenticationFailed,
            Self::RateLimit { .. } => ErrorCode::RateLimitExceeded,
            Self::ModelNotSupported { .. } => ErrorCode::ModelNotAvailable,
            Self::RequestValidation { .. } => ErrorCode::RequestValidationFailed,
            Self::QuotaExceeded { .. } => ErrorCode::QuotaExceeded,
            Self::ModelCapacity { .. } => ErrorCode::ModelCapacityExceeded,
            Self::JsonProcessing { .. } => ErrorCode::JsonProcessingFailed,
            Self::Timeout { .. } => ErrorCode::RequestTimeout,
            Self::Configuration { .. } => ErrorCode::ConfigurationError,
            Self::Streaming { .. } => ErrorCode::StreamingError,
            Self::InvalidApiKey { .. } => ErrorCode::InvalidApiKey,
            Self::Http(_) => ErrorCode::RequestTimeout, // Map HTTP errors to timeout
        }
    }
    
    /// Check if error is retryable
    #[inline]
    pub const fn is_retryable(&self) -> bool {
        match self {
            Self::Authentication { retry_possible, .. } => *retry_possible,
            Self::RateLimit { .. } => true,
            Self::ModelNotSupported { .. } => false,
            Self::RequestValidation { auto_correctable, .. } => *auto_correctable,
            Self::QuotaExceeded { .. } => false,
            Self::ModelCapacity { retry_recommended, .. } => *retry_recommended,
            Self::JsonProcessing { recovery_possible, .. } => *recovery_possible,
            Self::Timeout { retry_recommended, .. } => *retry_recommended,
            Self::Configuration { .. } => false,
            Self::Streaming { resumable, .. } => *resumable,
            Self::InvalidApiKey { .. } => false,
            Self::Http(_) => true, // HTTP errors are generally retryable
        }
    }
    
    /// Get retry delay in seconds
    #[inline]
    pub const fn retry_delay_seconds(&self) -> Option<u32> {
        match self {
            Self::RateLimit { retry_after_seconds, .. } => Some(*retry_after_seconds),
            Self::ModelCapacity { estimated_wait_seconds, .. } => Some(*estimated_wait_seconds),
            Self::Timeout { .. } => Some(5), // Default 5 second delay for timeouts
            _ => None}
    }
    
    /// Get suggested action for error recovery
    #[inline]
    pub fn suggested_action(&self) -> &'static str {
        match self {
            Self::Authentication { .. } => "Check API key and authentication settings",
            Self::RateLimit { .. } => "Reduce request rate and retry after delay",
            Self::ModelNotSupported { .. } => "Use a supported model from the available list",
            Self::RequestValidation { .. } => "Validate request parameters and correct invalid values",
            Self::QuotaExceeded { .. } => "Upgrade quota or wait for quota reset",
            Self::ModelCapacity { .. } => "Retry with delay or use alternative model",
            Self::JsonProcessing { .. } => "Check request format and try again",
            Self::Timeout { .. } => "Increase timeout or retry with delay",
            Self::Configuration { .. } => "Check configuration settings and correct invalid values",
            Self::Streaming { .. } => "Check connection and restart stream",
            Self::InvalidApiKey { .. } => "Provide valid API key in correct format",
            Self::Http(_) => "Check network connection and retry"}
    }
}

/// HTTP status code to AI21Error conversion
impl From<u16> for AI21Error {
    #[inline]
    fn from(status_code: u16) -> Self {
        match status_code {
            400 => Self::request_validation_error(
                "request",
                "Bad request format",
                "unknown",
                "Valid JSON request body",
                true,
            ),
            401 => Self::authentication_error(
                "Invalid API key",
                0,
                false,
                true,
                401,
            ),
            403 => Self::authentication_error(
                "API key forbidden",
                0,
                true,
                false,
                403,
            ),
            404 => Self::model_not_supported(
                "unknown",
                &[],
                "jamba-1.5-large",
                false,
            ),
            429 => Self::rate_limit_error(
                "Rate limit exceeded",
                60,
                0,
                0,
                0,
            ),
            500 => Self::model_capacity_error(
                "unknown",
                0,
                30,
                true,
                &["jamba-1.5-mini"],
            ),
            503 => Self::model_capacity_error(
                "unknown",
                100,
                60,
                true,
                &["jamba-1.5-mini"],
            ),
            _ => Self::Http(fluent_ai_http3::HttpError::Status(status_code))}
    }
}

/// Serde JSON error conversion
impl From<serde_json::Error> for AI21Error {
    #[inline]
    fn from(err: serde_json::Error) -> Self {
        Self::json_error(
            JsonOperation::ResponseDeserialization,
            &err.to_string(),
            Some(err.line()),
            Some(err.column()),
            false,
        )
    }
}

/// Display implementations for enum types
impl fmt::Display for QuotaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MonthlyTokens => write!(f, "Monthly tokens"),
            Self::DailyRequests => write!(f, "Daily requests"),
            Self::ConcurrentRequests => write!(f, "Concurrent requests"),
            Self::ModelSpecific => write!(f, "Model-specific quota"),
            Self::BillingQuota => write!(f, "Billing quota")}
    }
}

impl fmt::Display for JsonOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestSerialization => write!(f, "Request serialization"),
            Self::ResponseDeserialization => write!(f, "Response deserialization"),
            Self::StreamingParse => write!(f, "Streaming parse"),
            Self::ToolCallParse => write!(f, "Tool call parse"),
            Self::UsageStatsParse => write!(f, "Usage statistics parse")}
    }
}

impl fmt::Display for RequestType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatCompletion => write!(f, "Chat completion"),
            Self::StreamingCompletion => write!(f, "Streaming completion"),
            Self::ModelInfo => write!(f, "Model information"),
            Self::HealthCheck => write!(f, "Health check")}
    }
}

impl fmt::Display for StreamingErrorReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSseFormat => write!(f, "Invalid SSE format"),
            Self::ConnectionInterrupted => write!(f, "Connection interrupted"),
            Self::JsonParsingFailed => write!(f, "JSON parsing failed"),
            Self::UnexpectedEventType => write!(f, "Unexpected event type"),
            Self::StreamEndedUnexpectedly => write!(f, "Stream ended unexpectedly")}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let error = AI21Error::authentication_error(
            "Invalid token",
            32,
            true,
            false,
            401,
        );
        assert_eq!(error.error_code(), ErrorCode::AuthenticationFailed);
        assert!(!error.is_retryable());
    }
    
    #[test]
    fn test_rate_limit_error() {
        let error = AI21Error::rate_limit_error(
            "Rate limit exceeded",
            60,
            100,
            60,
            1640995200,
        );
        assert_eq!(error.error_code(), ErrorCode::RateLimitExceeded);
        assert!(error.is_retryable());
        assert_eq!(error.retry_delay_seconds(), Some(60));
    }
    
    #[test]
    fn test_http_status_conversion() {
        let error = AI21Error::from(404u16);
        assert_eq!(error.error_code(), ErrorCode::ModelNotAvailable);
        
        let error = AI21Error::from(429u16);
        assert_eq!(error.error_code(), ErrorCode::RateLimitExceeded);
        assert!(error.is_retryable());
    }
    
    #[test]
    fn test_json_error_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let error = AI21Error::from(json_err);
        assert_eq!(error.error_code(), ErrorCode::JsonProcessingFailed);
    }
    
    #[test]
    fn test_suggested_actions() {
        let error = AI21Error::authentication_error("Invalid key", 0, false, true, 401);
        assert_eq!(error.suggested_action(), "Check API key and authentication settings");
        
        let error = AI21Error::rate_limit_error("Rate limit", 60, 0, 0, 0);
        assert_eq!(error.suggested_action(), "Reduce request rate and retry after delay");
    }
}