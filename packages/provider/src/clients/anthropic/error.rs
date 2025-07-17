//! Zero-allocation error handling for Anthropic API
//!
//! Provides comprehensive error handling with static dispatch and minimal allocations.

// BadTraitImpl trait removed - not needed for provider error handling
use serde::{Deserialize, Serialize};
use std::fmt;

/// Zero-allocation error type for Anthropic API operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnthropicError {
    /// API authentication failed
    AuthenticationFailed(String),
    /// Rate limit exceeded
    RateLimited { retry_after: Option<u64> },
    /// Model not found or not accessible
    ModelNotFound(String),
    /// Invalid request parameters
    InvalidRequest(String),
    /// Server error from Anthropic
    ServerError { status: u16, message: String },
    /// Network error during request
    NetworkError(String),
    /// JSON parsing error
    ParseError(String),
    /// Tool execution failed
    ToolExecutionError { tool_name: String, error: String },
    /// Streaming error
    StreamError(String),
    /// HTTP error
    HttpError(String),
    /// File operation error
    FileError(String),
    /// API error
    ApiError(String),
    /// Generic error fallback
    Unknown(String),
}

impl fmt::Display for AnthropicError {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnthropicError::AuthenticationFailed(msg) => {
                write!(f, "Anthropic authentication failed: {}", msg)
            }
            AnthropicError::RateLimited { retry_after } => {
                if let Some(seconds) = retry_after {
                    write!(f, "Anthropic rate limit exceeded, retry after {} seconds", seconds)
                } else {
                    write!(f, "Anthropic rate limit exceeded")
                }
            }
            AnthropicError::ModelNotFound(model) => {
                write!(f, "Anthropic model not found: {}", model)
            }
            AnthropicError::InvalidRequest(msg) => {
                write!(f, "Invalid Anthropic request: {}", msg)
            }
            AnthropicError::ServerError { status, message } => {
                write!(f, "Anthropic server error {}: {}", status, message)
            }
            AnthropicError::NetworkError(msg) => {
                write!(f, "Anthropic network error: {}", msg)
            }
            AnthropicError::ParseError(msg) => {
                write!(f, "Anthropic JSON parse error: {}", msg)
            }
            AnthropicError::ToolExecutionError { tool_name, error } => {
                write!(f, "Tool '{}' execution failed: {}", tool_name, error)
            }
            AnthropicError::StreamError(msg) => {
                write!(f, "Anthropic stream error: {}", msg)
            }
            AnthropicError::HttpError(msg) => {
                write!(f, "Anthropic HTTP error: {}", msg)
            }
            AnthropicError::FileError(msg) => {
                write!(f, "Anthropic file error: {}", msg)
            }
            AnthropicError::ApiError(msg) => {
                write!(f, "Anthropic API error: {}", msg)
            }
            AnthropicError::Unknown(msg) => {
                write!(f, "Unknown Anthropic error: {}", msg)
            }
        }
    }
}

impl std::error::Error for AnthropicError {}

// BadTraitImpl implementation removed - not needed for provider error handling

/// API response envelope for error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(flatten)]
    pub data: Option<T>,
    pub error: Option<ApiError>,
}

/// Detailed API error structure from Anthropic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// Response with proper error handling for all Anthropic operations
pub type AnthropicResult<T> = Result<T, AnthropicError>;

/// Convert HTTP status and response to AnthropicError
#[inline(always)]
pub fn handle_http_error(status: u16, body: &str) -> AnthropicError {
    match status {
        401 => AnthropicError::AuthenticationFailed(
            "Invalid API key or authentication failed".to_string()
        ),
        429 => {
            // Try to parse retry-after header from body
            let retry_after = parse_retry_after(body);
            AnthropicError::RateLimited { retry_after }
        }
        404 => AnthropicError::ModelNotFound(
            "Model not found or not accessible".to_string()
        ),
        400 => AnthropicError::InvalidRequest(body.to_string()),
        500..=599 => AnthropicError::ServerError {
            status,
            message: body.to_string(),
        },
        _ => AnthropicError::Unknown(format!("HTTP {}: {}", status, body)),
    }
}

/// Parse retry-after from error response body
#[inline(always)]
fn parse_retry_after(body: &str) -> Option<u64> {
    // Try to parse JSON and extract retry-after
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(retry) = json.get("retry_after") {
            return retry.as_u64();
        }
    }
    None
}

/// Convert HTTP3 error to AnthropicError
#[inline(always)]
pub fn handle_http3_error(error: fluent_ai_http3::HttpError) -> AnthropicError {
    match error {
        fluent_ai_http3::HttpError::Timeout => {
            AnthropicError::NetworkError("Request timeout".to_string())
        }
        fluent_ai_http3::HttpError::Connection(_) => {
            AnthropicError::NetworkError("Connection failed".to_string())
        }
        fluent_ai_http3::HttpError::Http(status, msg) => {
            AnthropicError::ServerError {
                status: status as u16,
                message: msg,
            }
        }
        _ => {
            AnthropicError::NetworkError(error.to_string())
        }
    }
}

/// Convert JSON parsing error to AnthropicError
#[inline(always)]
pub fn handle_json_error(error: serde_json::Error) -> AnthropicError {
    AnthropicError::ParseError(error.to_string())
}