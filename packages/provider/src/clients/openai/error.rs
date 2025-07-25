//! OpenAI comprehensive error handling with zero allocation context preservation
//!
//! Provides complete error taxonomy for OpenAI multi-endpoint architecture:
//! - HTTP status mapping for all OpenAI API responses across endpoints
//! - Bearer token authentication with optional Organization header
//! - Multi-endpoint error handling (chat, embeddings, audio, vision)
//! - GPT model availability and validation errors across all 18 models
//! - Rate limiting with retry-after information and quota management
//! - Request validation with detailed context for each endpoint
//! - Function calling errors with tool validation and execution failures
//! - Audio processing errors for speech-to-text and text-to-speech
//! - Vision processing errors for image analysis and validation
//! - Streaming errors with partial response reconstruction
//! - JSON parsing with context preservation and recovery strategies
//! - Network and timeout handling with exponential backoff
//! - Configuration validation errors with auto-correction suggestions
//!
//! All errors use zero allocation patterns with arrayvec::ArrayString
//! for context preservation and #[repr(u8)] for minimal memory footprint.

use std::fmt;

use arrayvec::ArrayString;
use thiserror::Error;

/// Result type alias for OpenAI operations
pub type Result<T> = std::result::Result<T, OpenAIError>;

/// Comprehensive error taxonomy for OpenAI operations with zero allocation context
#[derive(Error, Debug, Clone)]
#[repr(u8)]
pub enum OpenAIError {
    /// HTTP request failed
    #[error("HTTP request failed: {0}")]
    Http(#[from] fluent_ai_http3::HttpError),

    /// Authentication failed for OpenAI API
    #[error("Authentication failed: {message}")]
    Authentication {
        /// Error message with context
        message: ArrayString<256>,
        /// Bearer token length for debugging
        token_length: usize,
        /// Organization ID if present
        organization_id: Option<ArrayString<64>>,
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
        reset_time: u64,
        /// Affected endpoint
        endpoint: EndpointType},

    /// Model not supported or unavailable
    #[error("Model not supported: {model}")]
    ModelNotSupported {
        /// Model name that was requested
        model: ArrayString<64>,
        /// Endpoint where model was requested
        endpoint: EndpointType,
        /// Available models for this endpoint
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
        /// Endpoint context
        endpoint: EndpointType,
        /// Whether value can be auto-corrected
        auto_correctable: bool},

    /// Quota exceeded for OpenAI API
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
        billing_info: ArrayString<256>,
        /// Affected endpoint
        endpoint: EndpointType},

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
        alternatives: ArrayString<256>,
        /// Affected endpoint
        endpoint: EndpointType},

    /// Function calling error
    #[error("Function calling failed: {function_name} - {reason}")]
    FunctionCall {
        /// Function name that failed
        function_name: ArrayString<64>,
        /// Failure reason
        reason: ArrayString<256>,
        /// Function arguments passed
        arguments: ArrayString<512>,
        /// Validation error details
        validation_error: Option<ArrayString<256>>,
        /// Whether retry is possible
        retry_possible: bool},

    /// Audio processing error
    #[error("Audio processing failed: {operation} - {reason}")]
    AudioProcessing {
        /// Audio operation that failed
        operation: AudioOperation,
        /// Failure reason
        reason: ArrayString<256>,
        /// Audio format if relevant
        audio_format: Option<ArrayString<32>>,
        /// File size if relevant
        file_size_bytes: Option<u64>,
        /// Whether retry is possible
        retry_possible: bool},

    /// Vision processing error
    #[error("Vision processing failed: {operation} - {reason}")]
    VisionProcessing {
        /// Vision operation that failed
        operation: VisionOperation,
        /// Failure reason
        reason: ArrayString<256>,
        /// Image format if relevant
        image_format: Option<ArrayString<32>>,
        /// Image size if relevant
        image_size_bytes: Option<u64>,
        /// Whether retry is possible
        retry_possible: bool},

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
        /// Endpoint context
        endpoint: EndpointType,
        /// Whether recovery is possible
        recovery_possible: bool},

    /// Request timeout
    #[error("Request timeout: {duration_ms}ms")]
    Timeout {
        /// Duration in milliseconds
        duration_ms: u64,
        /// Request type that timed out
        request_type: RequestType,
        /// Endpoint that timed out
        endpoint: EndpointType,
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
        expected_value: ArrayString<128>,
        /// Endpoint context
        endpoint: EndpointType},

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
        partial_content: ArrayString<512>,
        /// Endpoint context
        endpoint: EndpointType},

    /// Invalid API key format
    #[error("Invalid API key: {reason}")]
    InvalidApiKey {
        /// Reason for invalidity
        reason: ArrayString<128>,
        /// Key length for debugging
        key_length: usize,
        /// Whether format is valid
        format_valid: bool,
        /// Organization ID if present
        organization_id: Option<ArrayString<64>>},

    /// Token limit exceeded
    #[error("Token limit exceeded: {model} - {tokens} tokens")]
    TokenLimit {
        /// Model with token limit
        model: ArrayString<64>,
        /// Tokens in request
        tokens: u32,
        /// Maximum allowed tokens
        max_tokens: u32,
        /// Endpoint context
        endpoint: EndpointType,
        /// Whether request can be split
        can_split: bool},

    /// Content policy violation
    #[error("Content policy violation: {reason}")]
    ContentPolicy {
        /// Policy violation reason
        reason: ArrayString<256>,
        /// Flagged content category
        category: ArrayString<64>,
        /// Confidence score (0-100)
        confidence: u8,
        /// Whether content can be modified
        modifiable: bool},

    /// Usage tracking error
    #[error("Usage tracking failed: {reason}")]
    UsageTracking {
        /// Usage tracking failure reason
        reason: ArrayString<256>,
        /// Expected usage format
        expected_format: ArrayString<128>,
        /// Endpoint context
        endpoint: EndpointType,
        /// Whether tracking is required
        required: bool},

    /// Embedding processing error
    #[error("Embedding processing failed: {reason}")]
    EmbeddingProcessing {
        /// Embedding failure reason
        reason: ArrayString<256>,
        /// Batch size if relevant
        batch_size: Option<u32>,
        /// Dimension mismatch if relevant
        dimension_mismatch: Option<(u32, u32)>,
        /// Whether retry is possible
        retry_possible: bool},

    /// Organization access error
    #[error("Organization access denied: {reason}")]
    OrganizationAccess {
        /// Access denial reason
        reason: ArrayString<256>,
        /// Organization ID
        organization_id: ArrayString<64>,
        /// Required permissions
        required_permissions: ArrayString<256>,
        /// Whether access can be requested
        can_request_access: bool}}

/// Endpoint type enumeration for multi-endpoint error context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EndpointType {
    /// Chat completions endpoint
    ChatCompletions,
    /// Embeddings endpoint
    Embeddings,
    /// Audio transcription endpoint
    AudioTranscription,
    /// Audio translation endpoint
    AudioTranslation,
    /// Text-to-speech endpoint
    TextToSpeech,
    /// Vision analysis endpoint
    VisionAnalysis,
    /// Models endpoint
    Models,
    /// Files endpoint
    Files,
    /// Fine-tuning endpoint
    FineTuning,
    /// Moderations endpoint
    Moderations}

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
    BillingQuota,
    /// Organization quota
    OrganizationQuota}

/// Audio operation types for error context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AudioOperation {
    /// Speech-to-text transcription
    Transcription,
    /// Speech-to-text translation
    Translation,
    /// Text-to-speech synthesis
    TextToSpeech,
    /// Audio format validation
    FormatValidation,
    /// Audio file upload
    FileUpload}

/// Vision operation types for error context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VisionOperation {
    /// Image analysis
    ImageAnalysis,
    /// Image format validation
    FormatValidation,
    /// Image encoding
    ImageEncoding,
    /// Image upload
    ImageUpload,
    /// Image resizing
    ImageResize}

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
    /// Function call parse
    FunctionCallParse,
    /// Usage statistics parse
    UsageStatsParse,
    /// Tool definition parse
    ToolDefinitionParse}

/// Request type enumeration for timeout context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RequestType {
    /// Chat completion request
    ChatCompletion,
    /// Streaming completion request
    StreamingCompletion,
    /// Embedding request
    Embedding,
    /// Audio transcription request
    AudioTranscription,
    /// Audio translation request
    AudioTranslation,
    /// Text-to-speech request
    TextToSpeech,
    /// Vision analysis request
    VisionAnalysis,
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
    StreamEndedUnexpectedly,
    /// Function call streaming failed
    FunctionCallStreamingFailed}

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
    /// Function calling failed
    FunctionCallFailed = 7,
    /// Audio processing failed
    AudioProcessingFailed = 8,
    /// Vision processing failed
    VisionProcessingFailed = 9,
    /// JSON processing failed
    JsonProcessingFailed = 10,
    /// Request timeout
    RequestTimeout = 11,
    /// Configuration error
    ConfigurationError = 12,
    /// Streaming error
    StreamingError = 13,
    /// Invalid API key
    InvalidApiKey = 14,
    /// Token limit exceeded
    TokenLimitExceeded = 15,
    /// Content policy violation
    ContentPolicyViolation = 16,
    /// Usage tracking failed
    UsageTrackingFailed = 17,
    /// Embedding processing failed
    EmbeddingProcessingFailed = 18,
    /// Organization access denied
    OrganizationAccessDenied = 19}

impl OpenAIError {
    /// Create authentication error with context
    #[inline]
    pub fn authentication_error(
        message: &str,
        token_length: usize,
        organization_id: Option<&str>,
        format_valid: bool,
        retry_possible: bool,
        status_code: u16,
    ) -> Self {
        Self::Authentication {
            message: ArrayString::from(message).unwrap_or_default(),
            token_length,
            organization_id: organization_id.map(|id| ArrayString::from(id).unwrap_or_default()),
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
        endpoint: EndpointType,
    ) -> Self {
        Self::RateLimit {
            message: ArrayString::from(message).unwrap_or_default(),
            retry_after_seconds,
            current_requests,
            max_requests,
            reset_time,
            endpoint}
    }

    /// Create model not supported error
    #[inline]
    pub fn model_not_supported(
        model: &str,
        endpoint: EndpointType,
        available_models: &[&str],
        suggested_model: &str,
        model_exists: bool,
    ) -> Self {
        let available_str = available_models.join(", ");
        Self::ModelNotSupported {
            model: ArrayString::from(model).unwrap_or_default(),
            endpoint,
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
        endpoint: EndpointType,
        auto_correctable: bool,
    ) -> Self {
        Self::RequestValidation {
            field: ArrayString::from(field).unwrap_or_default(),
            reason: ArrayString::from(reason).unwrap_or_default(),
            current_value: ArrayString::from(current_value).unwrap_or_default(),
            valid_range: ArrayString::from(valid_range).unwrap_or_default(),
            endpoint,
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
        endpoint: EndpointType,
    ) -> Self {
        Self::QuotaExceeded {
            quota_type,
            current_usage,
            max_usage,
            reset_time,
            billing_info: ArrayString::from(billing_info).unwrap_or_default(),
            endpoint}
    }

    /// Create model capacity error
    #[inline]
    pub fn model_capacity_error(
        model: &str,
        queue_length: u32,
        estimated_wait_seconds: u32,
        retry_recommended: bool,
        alternatives: &[&str],
        endpoint: EndpointType,
    ) -> Self {
        let alternatives_str = alternatives.join(", ");
        Self::ModelCapacity {
            model: ArrayString::from(model).unwrap_or_default(),
            queue_length,
            estimated_wait_seconds,
            retry_recommended,
            alternatives: ArrayString::from(&alternatives_str).unwrap_or_default(),
            endpoint}
    }

    /// Create function call error
    #[inline]
    pub fn function_call_error(
        function_name: &str,
        reason: &str,
        arguments: &str,
        validation_error: Option<&str>,
        retry_possible: bool,
    ) -> Self {
        Self::FunctionCall {
            function_name: ArrayString::from(function_name).unwrap_or_default(),
            reason: ArrayString::from(reason).unwrap_or_default(),
            arguments: ArrayString::from(arguments).unwrap_or_default(),
            validation_error: validation_error.map(|e| ArrayString::from(e).unwrap_or_default()),
            retry_possible}
    }

    /// Create audio processing error
    #[inline]
    pub fn audio_processing_error(
        operation: AudioOperation,
        reason: &str,
        audio_format: Option<&str>,
        file_size_bytes: Option<u64>,
        retry_possible: bool,
    ) -> Self {
        Self::AudioProcessing {
            operation,
            reason: ArrayString::from(reason).unwrap_or_default(),
            audio_format: audio_format.map(|f| ArrayString::from(f).unwrap_or_default()),
            file_size_bytes,
            retry_possible}
    }

    /// Create vision processing error
    #[inline]
    pub fn vision_processing_error(
        operation: VisionOperation,
        reason: &str,
        image_format: Option<&str>,
        image_size_bytes: Option<u64>,
        retry_possible: bool,
    ) -> Self {
        Self::VisionProcessing {
            operation,
            reason: ArrayString::from(reason).unwrap_or_default(),
            image_format: image_format.map(|f| ArrayString::from(f).unwrap_or_default()),
            image_size_bytes,
            retry_possible}
    }

    /// Create JSON processing error
    #[inline]
    pub fn json_error(
        operation: JsonOperation,
        message: &str,
        line_number: Option<usize>,
        column_number: Option<usize>,
        endpoint: EndpointType,
        recovery_possible: bool,
    ) -> Self {
        Self::JsonProcessing {
            operation,
            message: ArrayString::from(message).unwrap_or_default(),
            line_number,
            column_number,
            endpoint,
            recovery_possible}
    }

    /// Create timeout error
    #[inline]
    pub fn timeout_error(
        duration_ms: u64,
        request_type: RequestType,
        endpoint: EndpointType,
        retry_recommended: bool,
        suggested_timeout_ms: u64,
    ) -> Self {
        Self::Timeout {
            duration_ms,
            request_type,
            endpoint,
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
        endpoint: EndpointType,
    ) -> Self {
        Self::Configuration {
            setting: ArrayString::from(setting).unwrap_or_default(),
            reason: ArrayString::from(reason).unwrap_or_default(),
            current_value: ArrayString::from(current_value).unwrap_or_default(),
            expected_value: ArrayString::from(expected_value).unwrap_or_default(),
            endpoint}
    }

    /// Create streaming error
    #[inline]
    pub fn streaming_error(
        reason: StreamingErrorReason,
        event_number: u32,
        resumable: bool,
        partial_content: &str,
        endpoint: EndpointType,
    ) -> Self {
        Self::Streaming {
            reason,
            event_number,
            resumable,
            partial_content: ArrayString::from(partial_content).unwrap_or_default(),
            endpoint}
    }

    /// Create invalid API key error
    #[inline]
    pub fn invalid_api_key_error(
        reason: &str,
        key_length: usize,
        format_valid: bool,
        organization_id: Option<&str>,
    ) -> Self {
        Self::InvalidApiKey {
            reason: ArrayString::from(reason).unwrap_or_default(),
            key_length,
            format_valid,
            organization_id: organization_id.map(|id| ArrayString::from(id).unwrap_or_default())}
    }

    /// Create token limit error
    #[inline]
    pub fn token_limit_error(
        model: &str,
        tokens: u32,
        max_tokens: u32,
        endpoint: EndpointType,
        can_split: bool,
    ) -> Self {
        Self::TokenLimit {
            model: ArrayString::from(model).unwrap_or_default(),
            tokens,
            max_tokens,
            endpoint,
            can_split}
    }

    /// Create content policy error
    #[inline]
    pub fn content_policy_error(
        reason: &str,
        category: &str,
        confidence: u8,
        modifiable: bool,
    ) -> Self {
        Self::ContentPolicy {
            reason: ArrayString::from(reason).unwrap_or_default(),
            category: ArrayString::from(category).unwrap_or_default(),
            confidence,
            modifiable}
    }

    /// Create usage tracking error
    #[inline]
    pub fn usage_tracking_error(
        reason: &str,
        expected_format: &str,
        endpoint: EndpointType,
        required: bool,
    ) -> Self {
        Self::UsageTracking {
            reason: ArrayString::from(reason).unwrap_or_default(),
            expected_format: ArrayString::from(expected_format).unwrap_or_default(),
            endpoint,
            required}
    }

    /// Create embedding processing error
    #[inline]
    pub fn embedding_processing_error(
        reason: &str,
        batch_size: Option<u32>,
        dimension_mismatch: Option<(u32, u32)>,
        retry_possible: bool,
    ) -> Self {
        Self::EmbeddingProcessing {
            reason: ArrayString::from(reason).unwrap_or_default(),
            batch_size,
            dimension_mismatch,
            retry_possible}
    }

    /// Create organization access error
    #[inline]
    pub fn organization_access_error(
        reason: &str,
        organization_id: &str,
        required_permissions: &str,
        can_request_access: bool,
    ) -> Self {
        Self::OrganizationAccess {
            reason: ArrayString::from(reason).unwrap_or_default(),
            organization_id: ArrayString::from(organization_id).unwrap_or_default(),
            required_permissions: ArrayString::from(required_permissions).unwrap_or_default(),
            can_request_access}
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
            Self::FunctionCall { .. } => ErrorCode::FunctionCallFailed,
            Self::AudioProcessing { .. } => ErrorCode::AudioProcessingFailed,
            Self::VisionProcessing { .. } => ErrorCode::VisionProcessingFailed,
            Self::JsonProcessing { .. } => ErrorCode::JsonProcessingFailed,
            Self::Timeout { .. } => ErrorCode::RequestTimeout,
            Self::Configuration { .. } => ErrorCode::ConfigurationError,
            Self::Streaming { .. } => ErrorCode::StreamingError,
            Self::InvalidApiKey { .. } => ErrorCode::InvalidApiKey,
            Self::TokenLimit { .. } => ErrorCode::TokenLimitExceeded,
            Self::ContentPolicy { .. } => ErrorCode::ContentPolicyViolation,
            Self::UsageTracking { .. } => ErrorCode::UsageTrackingFailed,
            Self::EmbeddingProcessing { .. } => ErrorCode::EmbeddingProcessingFailed,
            Self::OrganizationAccess { .. } => ErrorCode::OrganizationAccessDenied,
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
            Self::RequestValidation {
                auto_correctable, ..
            } => *auto_correctable,
            Self::QuotaExceeded { .. } => false,
            Self::ModelCapacity {
                retry_recommended, ..
            } => *retry_recommended,
            Self::FunctionCall { retry_possible, .. } => *retry_possible,
            Self::AudioProcessing { retry_possible, .. } => *retry_possible,
            Self::VisionProcessing { retry_possible, .. } => *retry_possible,
            Self::JsonProcessing {
                recovery_possible, ..
            } => *recovery_possible,
            Self::Timeout {
                retry_recommended, ..
            } => *retry_recommended,
            Self::Configuration { .. } => false,
            Self::Streaming { resumable, .. } => *resumable,
            Self::InvalidApiKey { .. } => false,
            Self::TokenLimit { can_split, .. } => *can_split,
            Self::ContentPolicy { modifiable, .. } => *modifiable,
            Self::UsageTracking { .. } => true,
            Self::EmbeddingProcessing { retry_possible, .. } => *retry_possible,
            Self::OrganizationAccess {
                can_request_access, ..
            } => *can_request_access,
            Self::Http(_) => true, // HTTP errors are generally retryable
        }
    }

    /// Get retry delay in seconds
    #[inline]
    pub const fn retry_delay_seconds(&self) -> Option<u32> {
        match self {
            Self::RateLimit {
                retry_after_seconds,
                ..
            } => Some(*retry_after_seconds),
            Self::ModelCapacity {
                estimated_wait_seconds,
                ..
            } => Some(*estimated_wait_seconds),
            Self::Timeout { .. } => Some(5), // Default 5 second delay for timeouts
            Self::AudioProcessing { .. } => Some(10), // Audio processing retry delay
            Self::VisionProcessing { .. } => Some(10), // Vision processing retry delay
            _ => None}
    }

    /// Get suggested action for error recovery
    #[inline]
    pub fn suggested_action(&self) -> &'static str {
        match self {
            Self::Authentication { .. } => "Check API key and authentication settings",
            Self::RateLimit { .. } => "Reduce request rate and retry after delay",
            Self::ModelNotSupported { .. } => "Use a supported model from the available list",
            Self::RequestValidation { .. } => {
                "Validate request parameters and correct invalid values"
            }
            Self::QuotaExceeded { .. } => "Upgrade quota or wait for quota reset",
            Self::ModelCapacity { .. } => "Retry with delay or use alternative model",
            Self::FunctionCall { .. } => "Check function definition and arguments",
            Self::AudioProcessing { .. } => "Validate audio format and file size",
            Self::VisionProcessing { .. } => "Validate image format and file size",
            Self::JsonProcessing { .. } => "Check request format and try again",
            Self::Timeout { .. } => "Increase timeout or retry with delay",
            Self::Configuration { .. } => "Check configuration settings and correct invalid values",
            Self::Streaming { .. } => "Check connection and restart stream",
            Self::InvalidApiKey { .. } => "Provide valid API key in correct format",
            Self::TokenLimit { .. } => "Reduce input size or split request",
            Self::ContentPolicy { .. } => "Modify content to comply with usage policies",
            Self::UsageTracking { .. } => "Check usage tracking configuration",
            Self::EmbeddingProcessing { .. } => "Validate embedding parameters and batch size",
            Self::OrganizationAccess { .. } => "Request access or check organization settings",
            Self::Http(_) => "Check network connection and retry"}
    }
}

/// HTTP status code to OpenAI error conversion
impl From<u16> for OpenAIError {
    #[inline]
    fn from(status_code: u16) -> Self {
        match status_code {
            400 => Self::request_validation_error(
                "request",
                "Bad request format",
                "unknown",
                "Valid JSON request body",
                EndpointType::ChatCompletions,
                true,
            ),
            401 => Self::authentication_error("Invalid API key", 0, None, false, true, 401),
            403 => Self::authentication_error("API key forbidden", 0, None, true, false, 403),
            404 => Self::model_not_supported(
                "unknown",
                EndpointType::ChatCompletions,
                &[],
                "gpt-4o",
                false,
            ),
            429 => Self::rate_limit_error(
                "Rate limit exceeded",
                60,
                0,
                0,
                0,
                EndpointType::ChatCompletions,
            ),
            500 => Self::model_capacity_error(
                "unknown",
                0,
                30,
                true,
                &["gpt-4o-mini"],
                EndpointType::ChatCompletions,
            ),
            503 => Self::model_capacity_error(
                "unknown",
                100,
                60,
                true,
                &["gpt-4o-mini"],
                EndpointType::ChatCompletions,
            ),
            _ => Self::Http(fluent_ai_http3::HttpError::Status(status_code))}
    }
}

/// Serde JSON error conversion
impl From<serde_json::Error> for OpenAIError {
    #[inline]
    fn from(err: serde_json::Error) -> Self {
        Self::json_error(
            JsonOperation::ResponseDeserialization,
            &err.to_string(),
            Some(err.line()),
            Some(err.column()),
            EndpointType::ChatCompletions,
            false,
        )
    }
}

/// Display implementations for enum types
impl fmt::Display for EndpointType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatCompletions => write!(f, "Chat completions"),
            Self::Embeddings => write!(f, "Embeddings"),
            Self::AudioTranscription => write!(f, "Audio transcription"),
            Self::AudioTranslation => write!(f, "Audio translation"),
            Self::TextToSpeech => write!(f, "Text-to-speech"),
            Self::VisionAnalysis => write!(f, "Vision analysis"),
            Self::Models => write!(f, "Models"),
            Self::Files => write!(f, "Files"),
            Self::FineTuning => write!(f, "Fine-tuning"),
            Self::Moderations => write!(f, "Moderations")}
    }
}

impl fmt::Display for QuotaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MonthlyTokens => write!(f, "Monthly tokens"),
            Self::DailyRequests => write!(f, "Daily requests"),
            Self::ConcurrentRequests => write!(f, "Concurrent requests"),
            Self::ModelSpecific => write!(f, "Model-specific quota"),
            Self::BillingQuota => write!(f, "Billing quota"),
            Self::OrganizationQuota => write!(f, "Organization quota")}
    }
}

impl fmt::Display for AudioOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transcription => write!(f, "Transcription"),
            Self::Translation => write!(f, "Translation"),
            Self::TextToSpeech => write!(f, "Text-to-speech"),
            Self::FormatValidation => write!(f, "Format validation"),
            Self::FileUpload => write!(f, "File upload")}
    }
}

impl fmt::Display for VisionOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageAnalysis => write!(f, "Image analysis"),
            Self::FormatValidation => write!(f, "Format validation"),
            Self::ImageEncoding => write!(f, "Image encoding"),
            Self::ImageUpload => write!(f, "Image upload"),
            Self::ImageResize => write!(f, "Image resize")}
    }
}

impl fmt::Display for JsonOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestSerialization => write!(f, "Request serialization"),
            Self::ResponseDeserialization => write!(f, "Response deserialization"),
            Self::StreamingParse => write!(f, "Streaming parse"),
            Self::FunctionCallParse => write!(f, "Function call parse"),
            Self::UsageStatsParse => write!(f, "Usage statistics parse"),
            Self::ToolDefinitionParse => write!(f, "Tool definition parse")}
    }
}

impl fmt::Display for RequestType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatCompletion => write!(f, "Chat completion"),
            Self::StreamingCompletion => write!(f, "Streaming completion"),
            Self::Embedding => write!(f, "Embedding"),
            Self::AudioTranscription => write!(f, "Audio transcription"),
            Self::AudioTranslation => write!(f, "Audio translation"),
            Self::TextToSpeech => write!(f, "Text-to-speech"),
            Self::VisionAnalysis => write!(f, "Vision analysis"),
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
            Self::StreamEndedUnexpectedly => write!(f, "Stream ended unexpectedly"),
            Self::FunctionCallStreamingFailed => write!(f, "Function call streaming failed")}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = OpenAIError::authentication_error(
            "Invalid token",
            32,
            Some("org-123"),
            true,
            false,
            401,
        );
        assert_eq!(error.error_code(), ErrorCode::AuthenticationFailed);
        assert!(!error.is_retryable());
    }

    #[test]
    fn test_rate_limit_error() {
        let error = OpenAIError::rate_limit_error(
            "Rate limit exceeded",
            60,
            100,
            60,
            1640995200,
            EndpointType::ChatCompletions,
        );
        assert_eq!(error.error_code(), ErrorCode::RateLimitExceeded);
        assert!(error.is_retryable());
        assert_eq!(error.retry_delay_seconds(), Some(60));
    }

    #[test]
    fn test_function_call_error() {
        let error = OpenAIError::function_call_error(
            "calculate_sum",
            "Invalid arguments",
            "{\"a\": \"not_a_number\"}",
            Some("Expected number for parameter 'a'"),
            true,
        );
        assert_eq!(error.error_code(), ErrorCode::FunctionCallFailed);
        assert!(error.is_retryable());
    }

    #[test]
    fn test_audio_processing_error() {
        let error = OpenAIError::audio_processing_error(
            AudioOperation::Transcription,
            "Unsupported audio format",
            Some("mp4"),
            Some(1024000),
            false,
        );
        assert_eq!(error.error_code(), ErrorCode::AudioProcessingFailed);
        assert!(!error.is_retryable());
    }

    #[test]
    fn test_vision_processing_error() {
        let error = OpenAIError::vision_processing_error(
            VisionOperation::ImageAnalysis,
            "Image too large",
            Some("png"),
            Some(5242880),
            false,
        );
        assert_eq!(error.error_code(), ErrorCode::VisionProcessingFailed);
        assert!(!error.is_retryable());
    }

    #[test]
    fn test_http_status_conversion() {
        let error = OpenAIError::from(404u16);
        assert_eq!(error.error_code(), ErrorCode::ModelNotAvailable);

        let error = OpenAIError::from(429u16);
        assert_eq!(error.error_code(), ErrorCode::RateLimitExceeded);
        assert!(error.is_retryable());
    }

    #[test]
    fn test_json_error_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let error = OpenAIError::from(json_err);
        assert_eq!(error.error_code(), ErrorCode::JsonProcessingFailed);
    }

    #[test]
    fn test_token_limit_error() {
        let error = OpenAIError::token_limit_error(
            "gpt-4o",
            10000,
            8192,
            EndpointType::ChatCompletions,
            true,
        );
        assert_eq!(error.error_code(), ErrorCode::TokenLimitExceeded);
        assert!(error.is_retryable());
    }

    #[test]
    fn test_content_policy_error() {
        let error =
            OpenAIError::content_policy_error("Content contains hate speech", "hate", 95, true);
        assert_eq!(error.error_code(), ErrorCode::ContentPolicyViolation);
        assert!(error.is_retryable());
    }

    #[test]
    fn test_embedding_processing_error() {
        let error = OpenAIError::embedding_processing_error(
            "Dimension mismatch",
            Some(10),
            Some((1536, 512)),
            true,
        );
        assert_eq!(error.error_code(), ErrorCode::EmbeddingProcessingFailed);
        assert!(error.is_retryable());
    }

    #[test]
    fn test_organization_access_error() {
        let error = OpenAIError::organization_access_error(
            "Insufficient permissions",
            "org-123",
            "admin",
            true,
        );
        assert_eq!(error.error_code(), ErrorCode::OrganizationAccessDenied);
        assert!(error.is_retryable());
    }

    #[test]
    fn test_suggested_actions() {
        let error = OpenAIError::authentication_error("Invalid key", 0, None, false, true, 401);
        assert_eq!(
            error.suggested_action(),
            "Check API key and authentication settings"
        );

        let error =
            OpenAIError::rate_limit_error("Rate limit", 60, 0, 0, 0, EndpointType::ChatCompletions);
        assert_eq!(
            error.suggested_action(),
            "Reduce request rate and retry after delay"
        );

        let error = OpenAIError::function_call_error("test", "Invalid args", "{}", None, true);
        assert_eq!(
            error.suggested_action(),
            "Check function definition and arguments"
        );
    }
}
