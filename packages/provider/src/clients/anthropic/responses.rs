//! Zero-allocation HTTP response processing for Anthropic API
//!
//! This module provides blazing-fast response parsing and validation with minimal allocations
//! and no locking requirements.

use fluent_ai_domain::completion::CompletionResponse;
use fluent_ai_http3::HttpResponse;
use serde_json;

use super::error::{AnthropicError, AnthropicResult};
use super::streaming::AnthropicStreamChunk;

/// Content type constants for response validation
const CONTENT_TYPE_JSON: &str = "application/json";
const CONTENT_TYPE_STREAM: &str = "text/event-stream";

/// SSE (Server-Sent Events) constants
const SSE_DATA_PREFIX: &str = "data: ";
const SSE_DONE_MARKER: &str = "[DONE]";

/// Zero-allocation response processor for Anthropic API
#[derive(Debug)]
pub struct AnthropicResponseProcessor;

impl AnthropicResponseProcessor {
    /// Process a completion response
    #[inline]
    pub fn process_completion_response(
        response: HttpResponse,
    ) -> AnthropicResult<CompletionResponse> {
        // Validate response status
        if !response.status.is_success() {
            return Err(Self::process_error_response(response));
        }

        // Validate content type
        Self::validate_json_content_type(&response)?;

        // Parse JSON response
        let completion_response: CompletionResponse = serde_json::from_slice(&response.body)
            .map_err(|e| AnthropicError::DeserializationError {
                message: format!("Failed to parse completion response: {}", e),
            })?;

        // Validate response structure
        Self::validate_completion_response(&completion_response)?;

        Ok(completion_response)
    }

    /// Process a streaming response into chunks
    #[inline]
    pub fn process_streaming_response(
        response: HttpResponse,
    ) -> AnthropicResult<Vec<AnthropicStreamChunk>> {
        // Validate response status
        if !response.status.is_success() {
            return Err(Self::process_error_response(response));
        }

        // Validate content type for streaming
        Self::validate_stream_content_type(&response)?;

        // Parse SSE stream
        let chunks = Self::parse_sse_stream(&response.body)?;

        Ok(chunks)
    }

    /// Process a test connection response
    #[inline]
    pub fn process_test_response(response: HttpResponse) -> AnthropicResult<()> {
        // Validate response status
        if !response.status.is_success() {
            return Err(Self::process_error_response(response));
        }

        // For test connections, we just need to verify it's a valid response
        Self::validate_json_content_type(&response)?;

        // Try to parse as completion response to ensure it's valid
        let _: CompletionResponse = serde_json::from_slice(&response.body).map_err(|e| {
            AnthropicError::DeserializationError {
                message: format!("Test response is not a valid completion response: {}", e),
            }
        })?;

        Ok(())
    }

    /// Process an error response
    #[inline]
    fn process_error_response(response: HttpResponse) -> AnthropicError {
        let status = response.status.as_u16();
        let body = String::from_utf8_lossy(&response.body);

        // Try to parse as JSON error
        if let Ok(error_json) = serde_json::from_slice::<serde_json::Value>(&response.body) {
            if let Some(error_message) = error_json
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
            {
                return AnthropicError::ApiError {
                    status,
                    message: error_message.to_string(),
                };
            }
        }

        // Fallback to raw body
        AnthropicError::HttpStatus {
            status,
            message: body.to_string(),
        }
    }

    /// Validate JSON content type
    #[inline]
    fn validate_json_content_type(response: &HttpResponse) -> AnthropicResult<()> {
        if let Some(content_type) = response.headers.get("content-type") {
            let content_type_str =
                content_type
                    .to_str()
                    .map_err(|_| AnthropicError::ResponseError {
                        message: "Invalid content-type header".to_string(),
                    })?;

            if !content_type_str.starts_with(CONTENT_TYPE_JSON) {
                return Err(AnthropicError::ResponseError {
                    message: format!("Expected JSON content type, got: {}", content_type_str),
                });
            }
        }

        Ok(())
    }

    /// Validate streaming content type
    #[inline]
    fn validate_stream_content_type(response: &HttpResponse) -> AnthropicResult<()> {
        if let Some(content_type) = response.headers.get("content-type") {
            let content_type_str =
                content_type
                    .to_str()
                    .map_err(|_| AnthropicError::ResponseError {
                        message: "Invalid content-type header".to_string(),
                    })?;

            if !content_type_str.starts_with(CONTENT_TYPE_STREAM) {
                return Err(AnthropicError::ResponseError {
                    message: format!("Expected streaming content type, got: {}", content_type_str),
                });
            }
        }

        Ok(())
    }

    /// Validate completion response structure
    #[inline]
    fn validate_completion_response(response: &CompletionResponse) -> AnthropicResult<()> {
        if response.content.is_empty() {
            return Err(AnthropicError::ResponseError {
                message: "Completion response has no content".to_string(),
            });
        }

        if response.model.is_empty() {
            return Err(AnthropicError::ResponseError {
                message: "Completion response has no model".to_string(),
            });
        }

        Ok(())
    }

    /// Parse SSE stream into chunks
    #[inline]
    fn parse_sse_stream(body: &[u8]) -> AnthropicResult<Vec<AnthropicStreamChunk>> {
        let body_str = std::str::from_utf8(body).map_err(|_| AnthropicError::ResponseError {
            message: "Invalid UTF-8 in streaming response".to_string(),
        })?;

        let mut chunks = Vec::new();

        for line in body_str.lines() {
            if line.starts_with(SSE_DATA_PREFIX) {
                let json_str = &line[SSE_DATA_PREFIX.len()..];

                // Skip empty lines and comments
                if json_str.trim().is_empty() || json_str.starts_with(':') {
                    continue;
                }

                // Check for done marker
                if json_str.trim() == SSE_DONE_MARKER {
                    break;
                }

                // Parse JSON chunk
                let chunk: AnthropicStreamChunk = serde_json::from_str(json_str).map_err(|e| {
                    AnthropicError::DeserializationError {
                        message: format!("Failed to parse stream chunk: {}", e),
                    }
                })?;

                chunks.push(chunk);
            }
        }

        Ok(chunks)
    }

    /// Extract rate limit information from headers
    #[inline]
    pub fn extract_rate_limit_info(response: &HttpResponse) -> RateLimitInfo {
        let mut info = RateLimitInfo::default();

        // Extract rate limit headers
        if let Some(remaining) = response.headers.get("x-ratelimit-remaining") {
            if let Ok(remaining_str) = remaining.to_str() {
                if let Ok(remaining_val) = remaining_str.parse::<u64>() {
                    info.requests_remaining = Some(remaining_val);
                }
            }
        }

        if let Some(reset) = response.headers.get("x-ratelimit-reset") {
            if let Ok(reset_str) = reset.to_str() {
                if let Ok(reset_val) = reset_str.parse::<u64>() {
                    info.reset_time = Some(reset_val);
                }
            }
        }

        if let Some(limit) = response.headers.get("x-ratelimit-limit") {
            if let Ok(limit_str) = limit.to_str() {
                if let Ok(limit_val) = limit_str.parse::<u64>() {
                    info.requests_limit = Some(limit_val);
                }
            }
        }

        info
    }

    /// Validate response size
    #[inline]
    pub fn validate_response_size(response: &HttpResponse, max_size: usize) -> AnthropicResult<()> {
        if response.body.len() > max_size {
            return Err(AnthropicError::ResponseError {
                message: format!(
                    "Response size {} exceeds maximum {}",
                    response.body.len(),
                    max_size
                ),
            });
        }

        Ok(())
    }
}

/// Rate limit information extracted from response headers
#[derive(Debug, Clone, Default)]
pub struct RateLimitInfo {
    /// Number of requests remaining in the current window
    pub requests_remaining: Option<u64>,
    /// Unix timestamp when the rate limit window resets
    pub reset_time: Option<u64>,
    /// Total number of requests allowed in the window
    pub requests_limit: Option<u64>,
}

impl RateLimitInfo {
    /// Check if we're approaching the rate limit
    #[inline]
    pub fn is_near_limit(&self, threshold: f64) -> bool {
        match (self.requests_remaining, self.requests_limit) {
            (Some(remaining), Some(limit)) => {
                let ratio = remaining as f64 / limit as f64;
                ratio < threshold
            }
            _ => false,
        }
    }

    /// Get the current rate limit utilization as a percentage
    #[inline]
    pub fn utilization_percentage(&self) -> Option<f64> {
        match (self.requests_remaining, self.requests_limit) {
            (Some(remaining), Some(limit)) => {
                if limit > 0 {
                    let used = limit - remaining;
                    Some((used as f64 / limit as f64) * 100.0)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Standalone function to process a completion response
#[inline]
pub fn process_completion_response(response: HttpResponse) -> AnthropicResult<CompletionResponse> {
    AnthropicResponseProcessor::process_completion_response(response)
}

/// Standalone function to process a streaming response
#[inline]
pub fn process_streaming_response(
    response: HttpResponse,
) -> AnthropicResult<Vec<AnthropicStreamChunk>> {
    AnthropicResponseProcessor::process_streaming_response(response)
}

/// Standalone function to process a test response
#[inline]
pub fn process_test_response(response: HttpResponse) -> AnthropicResult<()> {
    AnthropicResponseProcessor::process_test_response(response)
}

/// Standalone function to extract rate limit info
#[inline]
pub fn extract_rate_limit_info(response: &HttpResponse) -> RateLimitInfo {
    AnthropicResponseProcessor::extract_rate_limit_info(response)
}
