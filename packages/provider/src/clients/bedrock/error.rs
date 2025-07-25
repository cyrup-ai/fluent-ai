//! Comprehensive AWS Bedrock error types with zero allocation context preservation
//!
//! Provides semantic error handling for all AWS Bedrock failure modes including:
//! - HTTP status code mapping to semantic errors
//! - AWS SigV4 authentication failures
//! - Model access and availability errors
//! - Throttling and quota management
//! - Region support validation
//!
//! Uses arrayvec::ArrayString for zero allocation error context preservation.

use std::fmt;

use arrayvec::ArrayString;
use fluent_ai_http3::HttpError;

use crate::completion_provider::CompletionError;

/// Result type for Bedrock operations
pub type Result<T> = std::result::Result<T, BedrockError>;

/// Comprehensive AWS Bedrock error types
#[derive(thiserror::Error, Debug, Clone)]
pub enum BedrockError {
    /// HTTP transport layer errors
    #[error("HTTP request failed: {0}")]
    Http(#[from] HttpError),

    /// AWS authentication and authorization errors
    #[error("AWS authentication failed: {message}")]
    Auth {
        message: ArrayString<256>,
        error_code: Option<ArrayString<32>>},

    /// AWS SigV4 signing errors
    #[error("AWS signature generation failed: {message}")]
    Signature {
        message: ArrayString<256>,
        component: ArrayString<32>, // canonical_request, string_to_sign, signature
    },

    /// Model not found or access denied
    #[error("Model not accessible: {model} in region {region}")]
    ModelAccess {
        model: ArrayString<64>,
        region: ArrayString<32>,
        reason: ModelAccessReason},

    /// AWS region not supported for Bedrock
    #[error("Region not supported: {region}")]
    RegionNotSupported {
        region: ArrayString<32>,
        supported_regions: &'static [&'static str]},

    /// AWS throttling and rate limiting
    #[error("Request throttled: {reason}, retry after {retry_after}s")]
    Throttled {
        reason: ThrottleReason,
        retry_after: u64,
        request_id: Option<ArrayString<64>>},

    /// AWS quota and billing errors  
    #[error("Quota exceeded: {quota_type}, limit: {limit}")]
    QuotaExceeded {
        quota_type: QuotaType,
        limit: u64,
        current_usage: Option<u64>,
        reset_time: Option<u64>, // Unix timestamp
    },

    /// Model input validation errors
    #[error("Model input validation failed: {field} - {message}")]
    InputValidation {
        field: ArrayString<32>,
        message: ArrayString<128>,
        model: ArrayString<64>},

    /// AWS service errors (500-level)
    #[error("AWS service error: {service} - {message}")]
    Service {
        service: ArrayString<32>,
        message: ArrayString<256>,
        error_code: Option<ArrayString<32>>,
        request_id: Option<ArrayString<64>>},

    /// Configuration and setup errors
    #[error("Configuration error: {component} - {message}")]
    Config {
        component: ArrayString<32>,
        message: ArrayString<256>},

    /// JSON serialization/deserialization errors
    #[error("JSON processing failed: {operation} - {context}")]
    Json {
        operation: JsonOperation,
        context: ArrayString<128>},

    /// AWS credentials errors
    #[error("AWS credentials error: {message}")]
    Credentials {
        message: ArrayString<256>,
        source: CredentialsSource},

    /// Network timeout and connectivity
    #[error("Network timeout: {operation} after {timeout_ms}ms")]
    Timeout {
        operation: ArrayString<32>,
        timeout_ms: u64},

    /// Circuit breaker open state
    #[error("Circuit breaker open for {service}, failure rate: {failure_rate}%")]
    CircuitBreakerOpen {
        service: ArrayString<32>,
        failure_rate: f32,
        last_failure: ArrayString<128>}}

/// Model access failure reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelAccessReason {
    /// Model not found in region
    NotFound,
    /// Access denied due to permissions
    AccessDenied,
    /// Model disabled in account
    Disabled,
    /// Model requires additional provisioning
    NotProvisioned,
    /// Model in maintenance mode
    Maintenance}

impl fmt::Display for ModelAccessReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "model not found in region"),
            Self::AccessDenied => write!(f, "access denied"),
            Self::Disabled => write!(f, "model disabled"),
            Self::NotProvisioned => write!(f, "model not provisioned"),
            Self::Maintenance => write!(f, "model in maintenance")}
    }
}

/// AWS throttling reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrottleReason {
    /// Request rate too high
    RateLimit,
    /// Token rate too high
    TokenRateLimit,
    /// Concurrent request limit
    ConcurrencyLimit,
    /// Model-specific throttling
    ModelThrottling,
    /// Account-level throttling
    AccountThrottling}

impl fmt::Display for ThrottleReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RateLimit => write!(f, "request rate limit"),
            Self::TokenRateLimit => write!(f, "token rate limit"),
            Self::ConcurrencyLimit => write!(f, "concurrency limit"),
            Self::ModelThrottling => write!(f, "model throttling"),
            Self::AccountThrottling => write!(f, "account throttling")}
    }
}

/// AWS quota types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaType {
    /// Monthly token quota
    MonthlyTokens,
    /// Daily token quota
    DailyTokens,
    /// Hourly token quota
    HourlyTokens,
    /// Model invocation quota
    ModelInvocations,
    /// Concurrent requests quota
    ConcurrentRequests,
    /// Custom model training quota
    CustomModelTraining}

impl fmt::Display for QuotaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MonthlyTokens => write!(f, "monthly tokens"),
            Self::DailyTokens => write!(f, "daily tokens"),
            Self::HourlyTokens => write!(f, "hourly tokens"),
            Self::ModelInvocations => write!(f, "model invocations"),
            Self::ConcurrentRequests => write!(f, "concurrent requests"),
            Self::CustomModelTraining => write!(f, "custom model training")}
    }
}

/// JSON operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonOperation {
    /// Serializing request to JSON
    Serialize,
    /// Deserializing response from JSON
    Deserialize,
    /// Parsing streaming JSON chunks
    StreamParse,
    /// Validating JSON schema
    Validate}

impl fmt::Display for JsonOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Serialize => write!(f, "serialize"),
            Self::Deserialize => write!(f, "deserialize"),
            Self::StreamParse => write!(f, "stream parse"),
            Self::Validate => write!(f, "validate")}
    }
}

/// AWS credentials source
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialsSource {
    /// Environment variables
    Environment,
    /// AWS profile/config file
    Profile,
    /// IAM role
    Role,
    /// EC2 instance metadata
    InstanceMetadata,
    /// Explicit credentials
    Explicit}

impl fmt::Display for CredentialsSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Environment => write!(f, "environment variables"),
            Self::Profile => write!(f, "AWS profile"),
            Self::Role => write!(f, "IAM role"),
            Self::InstanceMetadata => write!(f, "EC2 instance metadata"),
            Self::Explicit => write!(f, "explicit credentials")}
    }
}

impl BedrockError {
    /// Create authentication error with zero allocation
    pub fn auth_error(message: &str) -> Self {
        let mut msg = ArrayString::new();
        let _ = msg.try_push_str(message);
        Self::Auth {
            message: msg,
            error_code: None}
    }

    /// Create signature error with zero allocation
    pub fn signature_error(message: &str, component: &str) -> Self {
        let mut msg = ArrayString::new();
        let mut comp = ArrayString::new();
        let _ = msg.try_push_str(message);
        let _ = comp.try_push_str(component);
        Self::Signature {
            message: msg,
            component: comp}
    }

    /// Create model access error with zero allocation
    pub fn model_access_error(model: &str, region: &str, reason: ModelAccessReason) -> Self {
        let mut mod_name = ArrayString::new();
        let mut reg_name = ArrayString::new();
        let _ = mod_name.try_push_str(model);
        let _ = reg_name.try_push_str(region);
        Self::ModelAccess {
            model: mod_name,
            region: reg_name,
            reason}
    }

    /// Create configuration error with zero allocation
    pub fn config_error(component: &str, message: &str) -> Self {
        let mut comp = ArrayString::new();
        let mut msg = ArrayString::new();
        let _ = comp.try_push_str(component);
        let _ = msg.try_push_str(message);
        Self::Config {
            component: comp,
            message: msg}
    }

    /// Create credentials error with zero allocation
    pub fn credentials_error(message: &str, source: CredentialsSource) -> Self {
        let mut msg = ArrayString::new();
        let _ = msg.try_push_str(message);
        Self::Credentials {
            message: msg,
            source}
    }

    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Throttled { .. } => true,
            Self::Service { .. } => true,
            Self::Timeout { .. } => true,
            Self::CircuitBreakerOpen { .. } => false,
            Self::Http(http_err) => {
                // Retryable HTTP status codes
                matches!(http_err, HttpError::StatusCode(code) if *code >= 500)
            }
            _ => false}
    }

    /// Get retry delay in seconds
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            Self::Throttled { retry_after, .. } => Some(*retry_after),
            Self::Service { .. } => Some(1), // 1 second for service errors
            Self::Timeout { .. } => Some(2), // 2 seconds for timeouts
            _ => None}
    }

    /// Extract AWS request ID if available
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::Throttled {
                request_id: Some(id),
                ..
            } => Some(id.as_str()),
            Self::Service {
                request_id: Some(id),
                ..
            } => Some(id.as_str()),
            _ => None}
    }
}

/// Convert from HTTP status codes to Bedrock errors
impl From<u16> for BedrockError {
    fn from(status: u16) -> Self {
        match status {
            400 => Self::InputValidation {
                field: ArrayString::from("request").unwrap_or_default(),
                message: ArrayString::from("bad request").unwrap_or_default(),
                model: ArrayString::new()},
            401 => Self::Auth {
                message: ArrayString::from("unauthorized").unwrap_or_default(),
                error_code: None},
            403 => Self::Auth {
                message: ArrayString::from("forbidden").unwrap_or_default(),
                error_code: None},
            404 => Self::ModelAccess {
                model: ArrayString::new(),
                region: ArrayString::new(),
                reason: ModelAccessReason::NotFound},
            429 => Self::Throttled {
                reason: ThrottleReason::RateLimit,
                retry_after: 60,
                request_id: None},
            500..=599 => Self::Service {
                service: ArrayString::from("bedrock").unwrap_or_default(),
                message: ArrayString::from("internal server error").unwrap_or_default(),
                error_code: None,
                request_id: None},
            _ => Self::Http(HttpError::StatusCode(status))}
    }
}

/// Convert to CompletionError for compatibility
impl From<BedrockError> for CompletionError {
    fn from(err: BedrockError) -> Self {
        match err {
            BedrockError::Auth { .. } => Self::AuthError,
            BedrockError::ModelAccess { .. } => Self::ModelError,
            BedrockError::InputValidation { .. } => Self::ValidationError,
            BedrockError::Throttled { .. } => Self::RateLimitError,
            BedrockError::Service { .. } => Self::ProviderError,
            BedrockError::Config { .. } => Self::ConfigError,
            _ => Self::ProviderError}
    }
}

/// Convert from serde_json errors
impl From<serde_json::Error> for BedrockError {
    fn from(err: serde_json::Error) -> Self {
        let operation = if err.is_data() {
            JsonOperation::Deserialize
        } else if err.is_syntax() {
            JsonOperation::StreamParse
        } else {
            JsonOperation::Validate
        };

        let mut context = ArrayString::new();
        let _ = context.try_push_str(&format!("line {}, column {}", err.line(), err.column()));

        Self::Json { operation, context }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = BedrockError::auth_error("Invalid API key");
        assert!(matches!(error, BedrockError::Auth { .. }));

        let error = BedrockError::model_access_error(
            "claude-3-5-sonnet",
            "us-east-1",
            ModelAccessReason::NotFound,
        );
        assert!(matches!(error, BedrockError::ModelAccess { .. }));
    }

    #[test]
    fn test_retryable_errors() {
        let throttled = BedrockError::Throttled {
            reason: ThrottleReason::RateLimit,
            retry_after: 60,
            request_id: None};
        assert!(throttled.is_retryable());
        assert_eq!(throttled.retry_delay(), Some(60));

        let auth_error = BedrockError::auth_error("Invalid token");
        assert!(!auth_error.is_retryable());
        assert_eq!(auth_error.retry_delay(), None);
    }

    #[test]
    fn test_status_code_conversion() {
        let error = BedrockError::from(429u16);
        assert!(matches!(error, BedrockError::Throttled { .. }));
        assert!(error.is_retryable());
    }
}
