//! Comprehensive error handling for Cohere multi-endpoint client
//!
//! Provides complete error taxonomy covering:
//! - Chat completion errors (Command models)
//! - Embedding errors (Embed models)
//! - Reranking errors (Rerank models)
//! - HTTP and authentication failures
//! - Model validation and capability errors
//! - Rate limiting and quota management
//!
//! All errors preserve context without allocations using arrayvec::ArrayString

use arrayvec::ArrayString;
use smallvec::SmallVec;
use thiserror::Error;

/// Result type for all Cohere operations
pub type Result<T> = std::result::Result<T, CohereError>;

/// Comprehensive Cohere error taxonomy with zero allocation context preservation
#[derive(Error, Debug, Clone)]
pub enum CohereError {
    /// HTTP transport layer errors
    #[error("HTTP request failed: {0}")]
    Http(#[from] fluent_ai_http3::HttpError),
    
    /// Authentication and authorization failures
    #[error("Authentication failed for endpoint {endpoint}: {message}")]
    Authentication {
        endpoint: ArrayString<32>,
        message: ArrayString<256>,
        error_code: AuthErrorCode,
        retry_possible: bool},
    
    /// API key validation errors
    #[error("Invalid API key: {reason}")]
    InvalidApiKey {
        reason: ArrayString<128>,
        key_length: usize,
        format_valid: bool},
    
    /// Model validation and capability errors
    #[error("Model {model} not supported for {operation}")]
    ModelNotSupported {
        model: ArrayString<64>,
        operation: CohereOperation,
        suggested_models: SmallVec<[&'static str; 4]>,
        endpoint: ArrayString<32>},
    
    /// Model capability validation failures
    #[error("Model {model} does not support {capability}")]
    CapabilityNotSupported {
        model: ArrayString<64>,
        capability: ModelCapability,
        available_capabilities: SmallVec<[ModelCapability; 8]>},
    
    /// Rate limiting errors with retry information
    #[error("Rate limit exceeded for {endpoint}: retry after {retry_after_ms}ms")]
    RateLimited {
        endpoint: ArrayString<32>,
        retry_after_ms: u64,
        current_rpm: u32,
        limit_rpm: u32,
        reset_time: u64},
    
    /// Quota and billing errors
    #[error("Quota exceeded for {resource}: {details}")]
    QuotaExceeded {
        resource: QuotaResource,
        details: ArrayString<256>,
        current_usage: u64,
        limit: u64,
        reset_date: Option<ArrayString<32>>},
    
    /// Chat completion specific errors
    #[error("Chat completion failed: {reason}")]
    ChatCompletion {
        reason: ChatErrorReason,
        model: ArrayString<64>,
        context: ArrayString<512>,
        recoverable: bool},
    
    /// Embedding specific errors
    #[error("Embedding operation failed: {reason}")]
    EmbeddingOperation {
        reason: EmbeddingErrorReason,
        model: ArrayString<64>,
        text_count: usize,
        batch_size: usize,
        failed_texts: SmallVec<[usize; 16]>},
    
    /// Reranking specific errors
    #[error("Reranking operation failed: {reason}")]
    RerankingOperation {
        reason: RerankingErrorReason,
        model: ArrayString<64>,
        query_length: usize,
        document_count: usize,
        failed_documents: SmallVec<[usize; 16]>},
    
    /// JSON processing errors
    #[error("JSON processing failed at {operation}: {details}")]
    JsonProcessing {
        operation: JsonOperation,
        details: ArrayString<256>,
        position: Option<usize>,
        recovery_possible: bool},
    
    /// Streaming errors
    #[error("Streaming failed for {endpoint}: {reason}")]
    Streaming {
        endpoint: ArrayString<32>,
        reason: StreamingErrorReason,
        chunk_count: u32,
        last_successful_chunk: Option<u32>,
        reconnect_possible: bool},
    
    /// Request validation errors
    #[error("Request validation failed: {field} - {reason}")]
    RequestValidation {
        field: ArrayString<64>,
        reason: ArrayString<256>,
        provided_value: ArrayString<128>,
        expected_format: ArrayString<128>,
        correction_hint: Option<ArrayString<256>>},
    
    /// Response parsing errors
    #[error("Response parsing failed for {endpoint}: {reason}")]
    ResponseParsing {
        endpoint: ArrayString<32>,
        reason: ArrayString<256>,
        content_type: ArrayString<64>,
        content_length: usize,
        partial_data: Option<ArrayString<512>>},
    
    /// Circuit breaker errors
    #[error("Circuit breaker {state} for {endpoint}")]
    CircuitBreaker {
        endpoint: ArrayString<32>,
        state: CircuitBreakerState,
        failure_count: u32,
        success_count: u32,
        next_retry_ms: u64},
    
    /// Configuration errors
    #[error("Configuration error: {setting} - {reason}")]
    Configuration {
        setting: ArrayString<64>,
        reason: ArrayString<256>,
        current_value: ArrayString<128>,
        valid_range: Option<ArrayString<64>>},
    
    /// Timeout errors with context
    #[error("Operation timed out after {duration_ms}ms for {operation}")]
    Timeout {
        operation: CohereOperation,
        duration_ms: u64,
        expected_duration_ms: u64,
        stage: OperationStage,
        partial_result: bool},
    
    /// Network connectivity errors
    #[error("Network error for {endpoint}: {reason}")]
    Network {
        endpoint: ArrayString<32>,
        reason: NetworkErrorReason,
        retry_count: u8,
        last_attempt_ms: u64,
        dns_resolution: bool},
    
    /// Server errors with status codes
    #[error("Server error {status_code}: {message}")]
    Server {
        status_code: u16,
        message: ArrayString<256>,
        endpoint: ArrayString<32>,
        request_id: Option<ArrayString<64>>,
        retry_recommended: bool}}

/// Authentication error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthErrorCode {
    InvalidApiKey,
    ExpiredApiKey,
    InsufficientPermissions,
    AccountSuspended,
    RegionRestricted,
    UnknownAuthError}

/// Cohere operation types for error context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohereOperation {
    Chat,
    Embedding,
    Reranking,
    ModelValidation,
    Authentication}

/// Model capabilities for validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCapability {
    Streaming,
    Tools,
    Vision,
    LongContext,
    BatchProcessing,
    FineTuning}

/// Quota resource types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaResource {
    RequestsPerMinute,
    TokensPerMonth,
    ConcurrentRequests,
    EmbeddingDimensions,
    RerankingDocuments,
    StorageGB}

/// Chat completion error reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatErrorReason {
    InvalidMessage,
    ContextTooLong,
    TemperatureOutOfRange,
    MaxTokensInvalid,
    ToolCallFailed,
    ModelOverloaded,
    ContentFiltered,
    UnknownChatError}

/// Embedding operation error reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingErrorReason {
    TextTooLong,
    BatchTooLarge,
    EmptyText,
    InvalidEncoding,
    DimensionMismatch,
    ModelOverloaded,
    UnknownEmbeddingError}

/// Reranking operation error reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankingErrorReason {
    QueryTooLong,
    TooManyDocuments,
    DocumentTooLong,
    EmptyQuery,
    EmptyDocuments,
    InvalidRelevanceThreshold,
    ModelOverloaded,
    UnknownRerankingError}

/// JSON operation types for error context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonOperation {
    RequestSerialization,
    ResponseDeserialization,
    StreamingParse,
    ToolCallParse,
    EmbeddingParse,
    RerankingParse}

/// Streaming error reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingErrorReason {
    ConnectionLost,
    InvalidSSEFormat,
    ChunkTooLarge,
    MalformedChunk,
    StreamTimeout,
    BufferOverflow,
    UnknownStreamingError}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen}

/// Operation stages for timeout context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationStage {
    Authentication,
    RequestSerialization,
    NetworkTransmission,
    ServerProcessing,
    ResponseReceiving,
    ResponseParsing,
    PostProcessing}

/// Network error reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkErrorReason {
    DNSResolutionFailed,
    ConnectionRefused,
    ConnectionTimeout,
    SSLHandshakeFailed,
    NetworkUnreachable,
    ProxyError,
    UnknownNetworkError}

impl CohereError {
    /// Create authentication error with context
    #[inline]
    pub fn auth_error(
        endpoint: &str,
        message: &str,
        error_code: AuthErrorCode,
        retry_possible: bool,
    ) -> Self {
        Self::Authentication {
            endpoint: ArrayString::from(endpoint).unwrap_or_default(),
            message: ArrayString::from(message).unwrap_or_default(),
            error_code,
            retry_possible}
    }
    
    /// Create model not supported error with suggestions
    #[inline]
    pub fn model_not_supported(
        model: &str,
        operation: CohereOperation,
        suggested_models: &[&'static str],
        endpoint: &str,
    ) -> Self {
        let mut suggestions = SmallVec::new();
        for &model in suggested_models.iter().take(4) {
            suggestions.push(model);
        }
        
        Self::ModelNotSupported {
            model: ArrayString::from(model).unwrap_or_default(),
            operation,
            suggested_models: suggestions,
            endpoint: ArrayString::from(endpoint).unwrap_or_default()}
    }
    
    /// Create rate limit error with retry information
    #[inline]
    pub fn rate_limited(
        endpoint: &str,
        retry_after_ms: u64,
        current_rpm: u32,
        limit_rpm: u32,
        reset_time: u64,
    ) -> Self {
        Self::RateLimited {
            endpoint: ArrayString::from(endpoint).unwrap_or_default(),
            retry_after_ms,
            current_rpm,
            limit_rpm,
            reset_time}
    }
    
    /// Create chat completion error
    #[inline]
    pub fn chat_error(
        reason: ChatErrorReason,
        model: &str,
        context: &str,
        recoverable: bool,
    ) -> Self {
        Self::ChatCompletion {
            reason,
            model: ArrayString::from(model).unwrap_or_default(),
            context: ArrayString::from(context).unwrap_or_default(),
            recoverable}
    }
    
    /// Create embedding error with batch context
    #[inline]
    pub fn embedding_error(
        reason: EmbeddingErrorReason,
        model: &str,
        text_count: usize,
        batch_size: usize,
        failed_indices: &[usize],
    ) -> Self {
        let mut failed_texts = SmallVec::new();
        for &index in failed_indices.iter().take(16) {
            failed_texts.push(index);
        }
        
        Self::EmbeddingOperation {
            reason,
            model: ArrayString::from(model).unwrap_or_default(),
            text_count,
            batch_size,
            failed_texts}
    }
    
    /// Create reranking error with document context
    #[inline]
    pub fn reranking_error(
        reason: RerankingErrorReason,
        model: &str,
        query_length: usize,
        document_count: usize,
        failed_documents: &[usize],
    ) -> Self {
        let mut failed_docs = SmallVec::new();
        for &index in failed_documents.iter().take(16) {
            failed_docs.push(index);
        }
        
        Self::RerankingOperation {
            reason,
            model: ArrayString::from(model).unwrap_or_default(),
            query_length,
            document_count,
            failed_documents: failed_docs}
    }
    
    /// Create JSON processing error
    #[inline]
    pub fn json_error(
        operation: JsonOperation,
        details: &str,
        position: Option<usize>,
        recovery_possible: bool,
    ) -> Self {
        Self::JsonProcessing {
            operation,
            details: ArrayString::from(details).unwrap_or_default(),
            position,
            recovery_possible}
    }
    
    /// Create streaming error
    #[inline]
    pub fn streaming_error(
        endpoint: &str,
        reason: StreamingErrorReason,
        chunk_count: u32,
        last_successful_chunk: Option<u32>,
        reconnect_possible: bool,
    ) -> Self {
        Self::Streaming {
            endpoint: ArrayString::from(endpoint).unwrap_or_default(),
            reason,
            chunk_count,
            last_successful_chunk,
            reconnect_possible}
    }
    
    /// Create request validation error
    #[inline]
    pub fn validation_error(
        field: &str,
        reason: &str,
        provided_value: &str,
        expected_format: &str,
        correction_hint: Option<&str>,
    ) -> Self {
        Self::RequestValidation {
            field: ArrayString::from(field).unwrap_or_default(),
            reason: ArrayString::from(reason).unwrap_or_default(),
            provided_value: ArrayString::from(provided_value).unwrap_or_default(),
            expected_format: ArrayString::from(expected_format).unwrap_or_default(),
            correction_hint: correction_hint.map(|h| ArrayString::from(h).unwrap_or_default())}
    }
    
    /// Create timeout error
    #[inline]
    pub fn timeout_error(
        operation: CohereOperation,
        duration_ms: u64,
        expected_duration_ms: u64,
        stage: OperationStage,
        partial_result: bool,
    ) -> Self {
        Self::Timeout {
            operation,
            duration_ms,
            expected_duration_ms,
            stage,
            partial_result}
    }
    
    /// Check if error is retryable
    #[inline]
    pub const fn is_retryable(&self) -> bool {
        match self {
            Self::Authentication { retry_possible, .. } => *retry_possible,
            Self::RateLimited { .. } => true,
            Self::Server { retry_recommended, .. } => *retry_recommended,
            Self::Network { .. } => true,
            Self::Timeout { .. } => true,
            Self::CircuitBreaker { state, .. } => matches!(state, CircuitBreakerState::HalfOpen),
            Self::Streaming { reconnect_possible, .. } => *reconnect_possible,
            _ => false}
    }
    
    /// Get retry delay in milliseconds
    #[inline]
    pub const fn retry_delay_ms(&self) -> Option<u64> {
        match self {
            Self::RateLimited { retry_after_ms, .. } => Some(*retry_after_ms),
            Self::CircuitBreaker { next_retry_ms, .. } => Some(*next_retry_ms),
            Self::Network { retry_count, .. } => {
                // Exponential backoff: 1s, 2s, 4s, 8s, 16s
                let delay = 1000u64 << (*retry_count as u64).min(4);
                Some(delay)
            }
            Self::Server { status_code, .. } => {
                match *status_code {
                    429 => Some(1000), // Rate limited
                    500..=599 => Some(2000), // Server error
                    _ => None}
            }
            _ => None}
    }
    
    /// Get error severity for logging and monitoring
    #[inline]
    pub const fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Authentication { .. } => ErrorSeverity::High,
            Self::InvalidApiKey { .. } => ErrorSeverity::High,
            Self::QuotaExceeded { .. } => ErrorSeverity::High,
            Self::Configuration { .. } => ErrorSeverity::High,
            Self::RateLimited { .. } => ErrorSeverity::Medium,
            Self::ModelNotSupported { .. } => ErrorSeverity::Medium,
            Self::RequestValidation { .. } => ErrorSeverity::Medium,
            Self::Timeout { .. } => ErrorSeverity::Medium,
            Self::Network { .. } => ErrorSeverity::Low,
            Self::Streaming { .. } => ErrorSeverity::Low,
            _ => ErrorSeverity::Medium}
    }
}

/// Error severity levels for monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical}

/// HTTP status code to CohereError conversion
impl From<u16> for CohereError {
    #[inline]
    fn from(status_code: u16) -> Self {
        let (message, retry_recommended) = match status_code {
            400 => ("Bad Request - Invalid parameters", false),
            401 => ("Unauthorized - Invalid API key", false),
            403 => ("Forbidden - Insufficient permissions", false),
            404 => ("Not Found - Invalid endpoint or model", false),
            422 => ("Unprocessable Entity - Request validation failed", false),
            429 => ("Too Many Requests - Rate limit exceeded", true),
            500 => ("Internal Server Error", true),
            502 => ("Bad Gateway", true),
            503 => ("Service Unavailable", true),
            504 => ("Gateway Timeout", true),
            _ => ("Unknown HTTP error", false)};
        
        Self::Server {
            status_code,
            message: ArrayString::from(message).unwrap_or_default(),
            endpoint: ArrayString::new(),
            request_id: None,
            retry_recommended}
    }
}

/// JSON serialization error conversion
impl From<serde_json::Error> for CohereError {
    #[inline]
    fn from(err: serde_json::Error) -> Self {
        let details = err.to_string();
        let position = if err.is_syntax() {
            err.line().map(|line| line.saturating_sub(1))
        } else {
            None
        };
        
        Self::json_error(
            JsonOperation::ResponseDeserialization,
            &details,
            position,
            false,
        )
    }
}