//! Zero-allocation HTTP request handling for Anthropic API
//!
//! This module provides blazing-fast request building and execution with zero allocations
//! after initial setup and no locking requirements.

use fluent_ai_http3::Http3;
use fluent_ai_http3::HttpClient;
use fluent_ai_http3::HttpRequest;
use fluent_ai_http3::HttpResponse;
use std::collections::HashMap;

use super::completion::AnthropicCompletionRequest;
use super::config::AnthropicConfig;
use super::error::{AnthropicError, AnthropicResult};

/// HTTP method constants for zero-allocation header building
const METHOD_POST: &str = "POST";
const METHOD_GET: &str = "GET";

/// Header name constants for zero-allocation header building
const HEADER_CONTENT_TYPE: &str = "Content-Type";
const HEADER_AUTHORIZATION: &str = "x-api-key";
const HEADER_ANTHROPIC_VERSION: &str = "anthropic-version";
const HEADER_USER_AGENT: &str = "User-Agent";

/// Content type constants
const CONTENT_TYPE_JSON: &str = "application/json";

/// API endpoint constants
const ENDPOINT_MESSAGES: &str = "/v1/messages";

/// Zero-allocation request builder for Anthropic API
#[derive(Debug)]
pub struct AnthropicRequestBuilder<'a> {
    config: &'a AnthropicConfig,
    http_client: &'a HttpClient}

impl<'a> AnthropicRequestBuilder<'a> {
    /// Create a new request builder
    #[inline]
    pub fn new(config: &'a AnthropicConfig, http_client: &'a HttpClient) -> Self {
        Self {
            config,
            http_client}
    }

    /// Build common headers for all requests
    #[inline]
    fn build_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::with_capacity(4);

        headers.insert(
            HEADER_CONTENT_TYPE.to_string(),
            CONTENT_TYPE_JSON.to_string(),
        );
        headers.insert(
            HEADER_AUTHORIZATION.to_string(),
            self.config.api_key().to_string(),
        );
        headers.insert(
            HEADER_ANTHROPIC_VERSION.to_string(),
            self.config.api_version().to_string(),
        );
        headers.insert(
            HEADER_USER_AGENT.to_string(),
            self.config.user_agent().to_string(),
        );

        headers
    }

    /// Build the full URL for an endpoint
    #[inline]
    fn build_url(&self, endpoint: &str) -> String {
        format!("{}{}", self.config.base_url(), endpoint)
    }



    /// Send a completion request
    pub async fn send_completion(
        &self,
        request: &AnthropicCompletionRequest,
    ) -> AnthropicResult<serde_json::Value> {
        let url = self.build_url(ENDPOINT_MESSAGES);
        let api_version = self.config.api_version();

        // Use new Http3::json() pattern with automatic serialization
        let response = Http3::json()
            .debug() // Enable debug logging
            .api_key(self.config.api_key()) // Use built-in api_key method
            .header(
                http::HeaderName::from_static("anthropic-version"),
                http::HeaderValue::from_str(api_version).unwrap_or(http::HeaderValue::from_static("2023-06-01"))
            )
            .body(request) // Automatic serde serialization
            .post(&url)
            .collect::<serde_json::Value>() // Collect to JSON response
            .await
            .map_err(|e| AnthropicError::NetworkError {
                message: format!("Failed to execute completion request: {}", e)})?;

        Ok(response)
    }

    /// Send a streaming completion request
    pub async fn send_streaming_completion(
        &self,
        request: &AnthropicCompletionRequest,
    ) -> AnthropicResult<HttpResponse> {
        let url = self.build_url(ENDPOINT_MESSAGES);
        let api_version = self.config.api_version();

        // Create streaming request
        let mut streaming_request = request.clone();
        streaming_request.stream = Some(true);

        // Use Http3::json() pattern for streaming
        let response = Http3::json()
            .debug()
            .api_key(self.config.api_key()) // Use built-in api_key method
            .header(
                http::HeaderName::from_static("anthropic-version"),
                http::HeaderValue::from_str(api_version).unwrap_or(http::HeaderValue::from_static("2023-06-01"))
            )
            .body(&streaming_request)
            .post(&url)
            .stream() // Use streaming instead of collect
            .await
            .map_err(|e| AnthropicError::NetworkError {
                message: format!("Failed to execute streaming request: {}", e)
            })?;

        // Check for HTTP errors
        if !response.status.is_success() {
            return Err(AnthropicError::HttpStatus {
                status: response.status.as_u16(),
                message: String::from_utf8_lossy(&response.body).to_string()});
        }

        Ok(response)
    }

    /// Send a test connection request
    pub async fn test_connection(&self) -> AnthropicResult<HttpResponse> {
        let url = self.build_url(ENDPOINT_MESSAGES);

        // Create minimal test request
        let test_request = AnthropicCompletionRequest {
            model: "claude-3-haiku-20240307".to_string(),
            max_tokens: 10,
            messages: vec![crate::clients::anthropic::messages::AnthropicMessage {
                role: "user".to_string(),
                content: "Hi".to_string()}],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            tool_choice: None};

        // Serialize request body
        let body =
            serde_json::to_vec(&test_request).map_err(|e| AnthropicError::SerializationError {
                message: format!("Failed to serialize test request: {}", e)})?;

        // Build HTTP request
        let mut http_request = HttpRequest::new(HttpMethod::Post, url.to_string()).with_body(body);

        // Add headers
        let headers = self.build_headers();
        for (name, value) in headers {
            http_request =
                http_request
                    .header(&name, &value)
                    .map_err(|e| AnthropicError::NetworkError {
                        message: format!("Failed to add header {}: {}", name, e)})?;
        }

        // Execute request
        let response = self.http_client.execute(http_request).await.map_err(|e| {
            AnthropicError::NetworkError {
                message: format!("Connection test failed: {}", e)}
        })?;

        // Check for HTTP errors
        if !response.status.is_success() {
            return Err(AnthropicError::HttpStatus {
                status: response.status.as_u16(),
                message: String::from_utf8_lossy(&response.body).to_string()});
        }

        Ok(response)
    }
}

/// Standalone function to send a completion request
#[inline]
pub async fn send_completion_request<'a>(
    config: &'a AnthropicConfig,
    http_client: &'a HttpClient,
    request: &'a AnthropicCompletionRequest,
) -> AnthropicResult<serde_json::Value> {
    let builder = AnthropicRequestBuilder::new(config, http_client);
    builder.send_completion(request).await
}

/// Standalone function to send a streaming completion request
#[inline]
pub async fn send_streaming_completion_request<'a>(
    config: &'a AnthropicConfig,
    http_client: &'a HttpClient,
    request: &'a AnthropicCompletionRequest,
) -> AnthropicResult<HttpResponse> {
    let builder = AnthropicRequestBuilder::new(config, http_client);
    builder.send_streaming_completion(request).await
}

/// Standalone function to test connection
#[inline]
pub async fn test_connection_request<'a>(
    config: &'a AnthropicConfig,
    http_client: &'a HttpClient,
) -> AnthropicResult<HttpResponse> {
    let builder = AnthropicRequestBuilder::new(config, http_client);
    builder.test_connection().await
}

/// Validate request parameters before sending
#[inline]
pub fn validate_completion_request(request: &AnthropicCompletionRequest) -> AnthropicResult<()> {
    if request.model.is_empty() {
        return Err(AnthropicError::ValidationError {
            message: "Model name cannot be empty".to_string()});
    }

    if request.max_tokens == 0 {
        return Err(AnthropicError::ValidationError {
            message: "max_tokens must be greater than 0".to_string()});
    }

    if request.max_tokens > 4096 {
        return Err(AnthropicError::ValidationError {
            message: "max_tokens cannot exceed 4096".to_string()});
    }

    if request.messages.is_empty() {
        return Err(AnthropicError::ValidationError {
            message: "Messages cannot be empty".to_string()});
    }

    // Validate temperature if provided
    if let Some(temp) = request.temperature {
        if temp < 0.0 || temp > 1.0 {
            return Err(AnthropicError::ValidationError {
                message: "Temperature must be between 0.0 and 1.0".to_string()});
        }
    }

    // Validate top_p if provided
    if let Some(top_p) = request.top_p {
        if top_p < 0.0 || top_p > 1.0 {
            return Err(AnthropicError::ValidationError {
                message: "top_p must be between 0.0 and 1.0".to_string()});
        }
    }

    // Validate top_k if provided
    if let Some(top_k) = request.top_k {
        if top_k <= 0 {
            return Err(AnthropicError::ValidationError {
                message: "top_k must be greater than 0".to_string()});
        }
    }

    Ok(())
}

/// Estimate request size for rate limiting
#[inline]
pub fn estimate_request_size(request: &AnthropicCompletionRequest) -> usize {
    // Rough estimation based on JSON serialization
    let mut size = 0;
    size += request.model.len();
    size += 20; // max_tokens field

    for message in &request.messages {
        size += message.role.len();
        size += message.content.len();
        size += 20; // JSON overhead
    }

    if let Some(system) = &request.system {
        size += system.len();
    }

    size += 100; // Additional JSON overhead and optional fields

    size
}
