//! Comprehensive error handling for Gemini completion operations
//!
//! This module provides production-ready error types with proper context,
//! error chains, and semantic error handling without unwrap/expect usage.

use std::fmt;

use crate::completion_provider::CompletionError as ProviderError;

/// Comprehensive error type for Gemini completion operations
#[derive(Debug, thiserror::Error)]
pub enum GeminiError {
    #[error("HTTP request failed: {context}")]
    HttpError { context: String },

    #[error("Authentication failed: {reason}")]
    AuthError { reason: String },

    #[error("Request payload too large: {size_bytes} bytes")]
    RequestTooLarge { size_bytes: usize },

    #[error("Rate limit exceeded: {retry_after_seconds}s")]
    RateLimited { retry_after_seconds: u64 },

    #[error("JSON parsing failed: {context}")]
    ParseError { context: String },

    #[error("Stream processing error: {context}")]
    StreamError { context: String },

    #[error("Invalid API response: {context}")]
    InvalidResponse { context: String },

    #[error("Model configuration error: {model_name} - {reason}")]
    ModelConfigError { model_name: String, reason: String },

    #[error("Content filtering triggered: {category}")]
    ContentFiltered { category: String },

    #[error("Request timeout after {timeout_seconds}s")]
    Timeout { timeout_seconds: u64 },

    #[error("Internal Gemini API error: {code} - {message}")]
    InternalError { code: u16, message: String },

    #[error("Tool call parsing failed: {tool_name} - {reason}")]
    ToolCallError { tool_name: String, reason: String },

    #[error("Schema validation failed: {field} - {reason}")]
    SchemaError { field: String, reason: String }}

impl GeminiError {
    /// Create HTTP error with context
    #[inline(always)]
    pub fn http_error(context: impl Into<String>) -> Self {
        Self::HttpError {
            context: context.into()}
    }

    /// Create authentication error with reason
    #[inline(always)]
    pub fn auth_error(reason: impl Into<String>) -> Self {
        Self::AuthError {
            reason: reason.into()}
    }

    /// Create request too large error
    #[inline(always)]
    pub fn request_too_large(size_bytes: usize) -> Self {
        Self::RequestTooLarge { size_bytes }
    }

    /// Create rate limit error
    #[inline(always)]
    pub fn rate_limited(retry_after_seconds: u64) -> Self {
        Self::RateLimited {
            retry_after_seconds}
    }

    /// Create parsing error with context
    #[inline(always)]
    pub fn parse_error(context: impl Into<String>) -> Self {
        Self::ParseError {
            context: context.into()}
    }

    /// Create stream error with context
    #[inline(always)]
    pub fn stream_error(context: impl Into<String>) -> Self {
        Self::StreamError {
            context: context.into()}
    }

    /// Create invalid response error
    #[inline(always)]
    pub fn invalid_response(context: impl Into<String>) -> Self {
        Self::InvalidResponse {
            context: context.into()}
    }

    /// Create model configuration error
    #[inline(always)]
    pub fn model_config_error(model_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ModelConfigError {
            model_name: model_name.into(),
            reason: reason.into()}
    }

    /// Create content filtered error
    #[inline(always)]
    pub fn content_filtered(category: impl Into<String>) -> Self {
        Self::ContentFiltered {
            category: category.into()}
    }

    /// Create timeout error
    #[inline(always)]
    pub fn timeout(timeout_seconds: u64) -> Self {
        Self::Timeout { timeout_seconds }
    }

    /// Create internal error
    #[inline(always)]
    pub fn internal_error(code: u16, message: impl Into<String>) -> Self {
        Self::InternalError {
            code,
            message: message.into()}
    }

    /// Create tool call error
    #[inline(always)]
    pub fn tool_call_error(tool_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ToolCallError {
            tool_name: tool_name.into(),
            reason: reason.into()}
    }

    /// Create schema error
    #[inline(always)]
    pub fn schema_error(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::SchemaError {
            field: field.into(),
            reason: reason.into()}
    }

    /// Check if error is retriable
    #[inline(always)]
    pub fn is_retriable(&self) -> bool {
        matches!(self,
            Self::HttpError { .. } |
            Self::RateLimited { .. } |
            Self::Timeout { .. } |
            Self::InternalError { code, .. } if *code >= 500
        )
    }

    /// Get error message without context for logging
    #[inline(always)]
    pub fn message(&self) -> &str {
        match self {
            Self::HttpError { .. } => "HTTP request failed",
            Self::AuthError { .. } => "Authentication failed",
            Self::RequestTooLarge { .. } => "Request too large",
            Self::RateLimited { .. } => "Rate limit exceeded",
            Self::ParseError { .. } => "JSON parsing failed",
            Self::StreamError { .. } => "Stream processing error",
            Self::InvalidResponse { .. } => "Invalid API response",
            Self::ModelConfigError { .. } => "Model configuration error",
            Self::ContentFiltered { .. } => "Content filtering triggered",
            Self::Timeout { .. } => "Request timeout",
            Self::InternalError { .. } => "Internal API error",
            Self::ToolCallError { .. } => "Tool call parsing failed",
            Self::SchemaError { .. } => "Schema validation failed"}
    }
}

/// Convert from HTTP errors
impl From<fluent_ai_http3::HttpError> for GeminiError {
    fn from(error: fluent_ai_http3::HttpError) -> Self {
        Self::http_error(error.to_string())
    }
}

/// Convert from JSON parsing errors
impl From<serde_json::Error> for GeminiError {
    fn from(error: serde_json::Error) -> Self {
        Self::parse_error(error.to_string())
    }
}

/// Convert to provider completion error
impl From<GeminiError> for ProviderError {
    fn from(error: GeminiError) -> Self {
        match error {
            GeminiError::HttpError { .. } => ProviderError::HttpError,
            GeminiError::AuthError { .. } => ProviderError::AuthError,
            GeminiError::RequestTooLarge { .. } => ProviderError::RequestTooLarge,
            GeminiError::RateLimited { .. } => ProviderError::RateLimited,
            GeminiError::ParseError { .. } => ProviderError::ParseError,
            GeminiError::StreamError { .. } => ProviderError::StreamError,
            GeminiError::InvalidResponse { context } => ProviderError::InvalidResponse(context),
            GeminiError::ModelConfigError { reason, .. } => {
                ProviderError::ConfigurationError(reason)
            }
            GeminiError::ContentFiltered { category } => ProviderError::ContentFiltered(category),
            GeminiError::Timeout { .. } => ProviderError::Timeout,
            GeminiError::InternalError { message, .. } => ProviderError::ProviderError(message),
            GeminiError::ToolCallError { reason, .. } => ProviderError::ToolCallError(reason),
            GeminiError::SchemaError { reason, .. } => ProviderError::ValidationError(reason)}
    }
}

/// Result type for Gemini operations
pub type GeminiResult<T> = Result<T, GeminiError>;

/// Parse HTTP status code to appropriate Gemini error
#[inline(always)]
pub fn parse_http_status_error(status_code: u16, response_body: Option<&str>) -> GeminiError {
    let body_context = response_body.unwrap_or("No response body");

    match status_code {
        400 => GeminiError::invalid_response(format!("Bad request: {}", body_context)),
        401 => GeminiError::auth_error(format!("Invalid API key: {}", body_context)),
        403 => GeminiError::auth_error(format!("Forbidden: {}", body_context)),
        413 => GeminiError::request_too_large(0), // Size not provided in response
        429 => {
            // Try to parse retry-after from response
            let retry_after = if body_context.contains("retry-after") {
                60 // Default 60 seconds if present but not parseable
            } else {
                30 // Default 30 seconds
            };
            GeminiError::rate_limited(retry_after)
        }
        500..=599 => GeminiError::internal_error(status_code, body_context.to_string()),
        _ => GeminiError::http_error(format!("HTTP {}: {}", status_code, body_context))}
}

/// Extract error details from Gemini API response
#[inline(always)]
pub fn parse_api_error_response(response: &str) -> GeminiError {
    // Try to parse structured error response
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response) {
        if let Some(error_obj) = parsed.get("error") {
            let code = error_obj.get("code").and_then(|c| c.as_u64()).unwrap_or(0) as u16;

            let message = error_obj
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");

            let status = error_obj
                .get("status")
                .and_then(|s| s.as_str())
                .unwrap_or("");

            return match status {
                "INVALID_ARGUMENT" => GeminiError::invalid_response(message.to_string()),
                "UNAUTHENTICATED" => GeminiError::auth_error(message.to_string()),
                "PERMISSION_DENIED" => GeminiError::auth_error(message.to_string()),
                "RESOURCE_EXHAUSTED" => GeminiError::rate_limited(60),
                "CONTENT_FILTERED" => GeminiError::content_filtered(message.to_string()),
                _ => GeminiError::internal_error(code, message.to_string())};
        }
    }

    // Fallback to generic parsing error
    GeminiError::parse_error(format!("Unable to parse error response: {}", response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = GeminiError::http_error("Connection refused");
        assert!(matches!(error, GeminiError::HttpError { .. }));
        assert_eq!(error.message(), "HTTP request failed");
        assert!(error.is_retriable());
    }

    #[test]
    fn test_status_code_parsing() {
        let error = parse_http_status_error(401, Some("Invalid API key"));
        assert!(matches!(error, GeminiError::AuthError { .. }));
        assert!(!error.is_retriable());
    }

    #[test]
    fn test_api_error_parsing() {
        let response = r#"{"error": {"code": 400, "message": "Invalid request", "status": "INVALID_ARGUMENT"}}"#;
        let error = parse_api_error_response(response);
        assert!(matches!(error, GeminiError::InvalidResponse { .. }));
    }

    #[test]
    fn test_error_conversion() {
        let gemini_error = GeminiError::auth_error("Invalid API key");
        let provider_error: ProviderError = gemini_error.into();
        assert!(matches!(provider_error, ProviderError::AuthError));
    }
}
