//! Zero-allocation error handling for OpenAI API provider
//!
//! Comprehensive error types for all OpenAI API operations with semantic error handling
//! and optimal performance patterns.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Result type for OpenAI operations
pub type OpenAIResult<T> = Result<T, OpenAIError>;

/// Comprehensive OpenAI API error type with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpenAIError {
    /// Authentication failed - invalid API key
    AuthenticationFailed(String),
    /// Authorization failed - insufficient permissions
    AuthorizationFailed(String),
    /// Rate limit exceeded - too many requests
    RateLimitExceeded {
        retry_after: Option<u64>,
        reset_time: Option<u64>,
        limit_type: String,
    },
    /// Request quota exceeded - billing related
    QuotaExceeded(String),
    /// Invalid request parameters
    InvalidRequest {
        param: Option<String>,
        message: String,
    },
    /// Model not found or unsupported
    ModelNotFound(String),
    /// Content policy violation
    ContentPolicyViolation(String),
    /// Context length exceeded for model
    ContextLengthExceeded {
        max_tokens: u64,
        requested_tokens: u64,
    },
    /// Server error from OpenAI
    ServerError {
        status: u16,
        error_type: String,
        message: String,
    },
    /// Network connection error
    NetworkError(String),
    /// Request timeout
    Timeout(String),
    /// JSON parsing error
    JsonError(String),
    /// Feature not supported by model
    FeatureNotSupported {
        feature: String,
        model: String,
    },
    /// Tool/function calling error
    ToolError(String),
    /// Vision processing error
    VisionError(String),
    /// Audio processing error
    AudioError(String),
    /// Embedding generation error
    EmbeddingError(String),
    /// Moderation error
    ModerationError(String),
    /// Generic API error
    ApiError(String),
}

/// OpenAI API error response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIErrorResponse {
    pub error: OpenAIErrorDetail,
}

/// OpenAI API error detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl fmt::Display for OpenAIError {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthenticationFailed(msg) => write!(f, "Authentication failed: {}", msg),
            Self::AuthorizationFailed(msg) => write!(f, "Authorization failed: {}", msg),
            Self::RateLimitExceeded { limit_type, retry_after, .. } => {
                if let Some(retry) = retry_after {
                    write!(f, "Rate limit exceeded for {}: retry after {} seconds", limit_type, retry)
                } else {
                    write!(f, "Rate limit exceeded for {}", limit_type)
                }
            }
            Self::QuotaExceeded(msg) => write!(f, "Quota exceeded: {}", msg),
            Self::InvalidRequest { param, message } => {
                if let Some(p) = param {
                    write!(f, "Invalid request parameter '{}': {}", p, message)
                } else {
                    write!(f, "Invalid request: {}", message)
                }
            }
            Self::ModelNotFound(model) => write!(f, "Model not found: {}", model),
            Self::ContentPolicyViolation(msg) => write!(f, "Content policy violation: {}", msg),
            Self::ContextLengthExceeded { max_tokens, requested_tokens } => {
                write!(f, "Context length exceeded: requested {} tokens, max {}", requested_tokens, max_tokens)
            }
            Self::ServerError { status, error_type, message } => {
                write!(f, "Server error {}: {} - {}", status, error_type, message)
            }
            Self::NetworkError(msg) => write!(f, "Network error: {}", msg),
            Self::Timeout(msg) => write!(f, "Request timeout: {}", msg),
            Self::JsonError(msg) => write!(f, "JSON error: {}", msg),
            Self::FeatureNotSupported { feature, model } => {
                write!(f, "Feature '{}' not supported by model '{}'", feature, model)
            }
            Self::ToolError(msg) => write!(f, "Tool error: {}", msg),
            Self::VisionError(msg) => write!(f, "Vision error: {}", msg),
            Self::AudioError(msg) => write!(f, "Audio error: {}", msg),
            Self::EmbeddingError(msg) => write!(f, "Embedding error: {}", msg),
            Self::ModerationError(msg) => write!(f, "Moderation error: {}", msg),
            Self::ApiError(msg) => write!(f, "API error: {}", msg),
        }
    }
}

impl std::error::Error for OpenAIError {}

impl OpenAIError {
    /// Create error from HTTP status and response body
    #[inline(always)]
    pub fn from_status_and_body(status: u16, body: &str) -> Self {
        match status {
            401 => Self::AuthenticationFailed("Invalid API key".to_string()),
            403 => Self::AuthorizationFailed("Insufficient permissions".to_string()),
            429 => Self::RateLimitExceeded {
                retry_after: None,
                reset_time: None,
                limit_type: "requests".to_string(),
            },
            400 => {
                if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(body) {
                    let detail = error_response.error;
                    match detail.error_type.as_str() {
                        "invalid_request_error" => Self::InvalidRequest {
                            param: detail.param,
                            message: detail.message,
                        },
                        "content_policy_violation" => Self::ContentPolicyViolation(detail.message),
                        _ => Self::ApiError(detail.message),
                    }
                } else {
                    Self::InvalidRequest {
                        param: None,
                        message: body.to_string(),
                    }
                }
            }
            404 => Self::ModelNotFound("Model not found".to_string()),
            413 => Self::ContextLengthExceeded {
                max_tokens: 0,
                requested_tokens: 0,
            },
            500..=599 => Self::ServerError {
                status,
                error_type: "server_error".to_string(),
                message: body.to_string(),
            },
            _ => Self::ApiError(format!("HTTP {} - {}", status, body)),
        }
    }

    /// Create error from reqwest error
    #[inline(always)]
    pub fn from_reqwest_error(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            Self::Timeout(err.to_string())
        } else if err.is_connect() {
            Self::NetworkError(err.to_string())
        } else {
            Self::ApiError(err.to_string())
        }
    }

    /// Create error from serde JSON error
    #[inline(always)]
    pub fn from_json_error(err: serde_json::Error) -> Self {
        Self::JsonError(err.to_string())
    }

    /// Check if error is retryable
    #[inline(always)]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimitExceeded { .. } => true,
            Self::ServerError { status, .. } if *status >= 500 => true,
            Self::NetworkError(_) => true,
            Self::Timeout(_) => true,
            _ => false,
        }
    }

    /// Get retry delay in seconds if applicable
    #[inline(always)]
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            Self::RateLimitExceeded { retry_after, .. } => *retry_after,
            Self::ServerError { .. } => Some(2), // 2 second default retry
            Self::NetworkError(_) => Some(1),
            Self::Timeout(_) => Some(3),
            _ => None,
        }
    }

    /// Check if error indicates quota/billing issues
    #[inline(always)]
    pub fn is_quota_error(&self) -> bool {
        matches!(self, Self::QuotaExceeded(_))
    }

    /// Check if error indicates authentication issues
    #[inline(always)]
    pub fn is_auth_error(&self) -> bool {
        matches!(self, Self::AuthenticationFailed(_) | Self::AuthorizationFailed(_))
    }

    /// Check if error indicates rate limiting
    #[inline(always)]
    pub fn is_rate_limit_error(&self) -> bool {
        matches!(self, Self::RateLimitExceeded { .. })
    }
}

/// Create OpenAI provider from environment variables
#[inline(always)]
pub fn from_env() -> OpenAIResult<super::OpenAIProvider> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| OpenAIError::AuthenticationFailed(
            "OPENAI_API_KEY environment variable not set".to_string()
        ))?;
    super::OpenAIProvider::new(api_key)
}

/// Create OpenAI provider from environment with custom base URL
#[inline(always)]
pub fn from_env_with_base_url(base_url: impl Into<String>) -> OpenAIResult<super::OpenAIProvider> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| OpenAIError::AuthenticationFailed(
            "OPENAI_API_KEY environment variable not set".to_string()
        ))?;
    super::OpenAIProvider::with_config(api_key, base_url, fluent_ai_provider::Models::Gpt4O)
}