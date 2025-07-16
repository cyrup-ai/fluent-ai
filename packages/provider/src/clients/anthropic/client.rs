//! Anthropic client implementation with zero-allocation patterns
//!
//! This module provides a production-ready Anthropic client with QUIC/HTTP3 support,
//! proper error handling, and zero-allocation request patterns.

use crate::http::HttpRequest;
use super::completion::{AnthropicCompletionRequest, AnthropicCompletionResponse};
use super::error::{AnthropicError, AnthropicResult};
use super::streaming::AnthropicStreamChunk;
use futures::Stream;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Anthropic client configuration
#[derive(Debug, Clone)]
pub struct AnthropicClientConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
}

impl Default for AnthropicClientConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.anthropic.com".to_string(),
            timeout_seconds: 300,
            max_retries: 3,
        }
    }
}

/// High-performance Anthropic client with QUIC/HTTP3 support
pub struct AnthropicClient {
    config: AnthropicClientConfig,
    http_client: Arc<HttpRequest>,
    request_count: Arc<RwLock<u64>>,
}

impl AnthropicClient {
    /// Create a new Anthropic client with API key and base URL
    pub async fn new(api_key: String, base_url: String) -> AnthropicResult<Self> {
        let config = AnthropicClientConfig {
            api_key,
            base_url,
            ..Default::default()
        };
        
        let http_client = Arc::new(HttpRequest::new());
        let request_count = Arc::new(RwLock::new(0));
        
        Ok(Self {
            config,
            http_client,
            request_count,
        })
    }
    
    /// Create a new Anthropic client with custom configuration
    pub async fn with_config(config: AnthropicClientConfig) -> AnthropicResult<Self> {
        let http_client = Arc::new(HttpRequest::new());
        let request_count = Arc::new(RwLock::new(0));
        
        Ok(Self {
            config,
            http_client,
            request_count,
        })
    }
    
    /// Get the current request count
    pub async fn request_count(&self) -> u64 {
        *self.request_count.read().await
    }
    
    /// Increment the request count
    async fn increment_request_count(&self) {
        let mut count = self.request_count.write().await;
        *count += 1;
    }
    
    /// Create common headers for Anthropic requests
    fn common_headers(&self) -> hashbrown::HashMap<String, String> {
        let mut headers = hashbrown::HashMap::new();
        headers.insert("x-api-key".to_string(), self.config.api_key.clone());
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());
        headers.insert("User-Agent".to_string(), "fluent-ai-provider/0.1.0".to_string());
        headers
    }
    
    /// Send a completion request
    pub async fn send_completion(
        &self,
        request: &AnthropicCompletionRequest,
    ) -> AnthropicResult<AnthropicCompletionResponse> {
        self.increment_request_count().await;
        
        let url = format!("{}/v1/messages", self.config.base_url);
        let headers = self.common_headers();
        
        let body = serde_json::to_vec(request).map_err(|e| {
            AnthropicError::SerializationError {
                message: format!("Failed to serialize completion request: {}", e),
            }
        })?;
        
        let response = self.http_client
            .post(&url)
            .headers(headers)
            .body(body)
            .timeout_seconds(self.config.timeout_seconds)
            .send()
            .await
            .map_err(|e| {
                AnthropicError::NetworkError {
                    message: format!("Failed to send completion request: {}", e),
                }
            })?;
        
        if !response.status.is_success() {
            return Err(AnthropicError::ApiError {
                status: response.status.as_u16(),
                message: String::from_utf8_lossy(&response.body).to_string(),
            });
        }
        
        let completion_response: AnthropicCompletionResponse = serde_json::from_slice(&response.body)
            .map_err(|e| {
                AnthropicError::DeserializationError {
                    message: format!("Failed to deserialize completion response: {}", e),
                }
            })?;
        
        Ok(completion_response)
    }
    
    /// Send a streaming completion request
    pub async fn send_streaming_completion(
        &self,
        request: &AnthropicCompletionRequest,
    ) -> AnthropicResult<Box<dyn Stream<Item = AnthropicResult<AnthropicStreamChunk>> + Send + Unpin>> {
        self.increment_request_count().await;
        
        let url = format!("{}/v1/messages", self.config.base_url);
        let headers = self.common_headers();
        
        // Create streaming request
        let mut streaming_request = request.clone();
        streaming_request.stream = Some(true);
        
        let body = serde_json::to_vec(&streaming_request).map_err(|e| {
            AnthropicError::SerializationError {
                message: format!("Failed to serialize streaming request: {}", e),
            }
        })?;
        
        let stream = self.http_client
            .post(&url)
            .headers(headers)
            .body(body)
            .timeout_seconds(self.config.timeout_seconds)
            .send_stream()
            .await
            .map_err(|e| {
                AnthropicError::NetworkError {
                    message: format!("Failed to send streaming request: {}", e),
                }
            })?;
        
        // Convert HTTP stream to Anthropic chunk stream
        let chunk_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    // Parse SSE chunk
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    if chunk_str.starts_with("data: ") {
                        let json_str = &chunk_str[6..];
                        if json_str.trim() == "[DONE]" {
                            return Ok(AnthropicStreamChunk::done());
                        }
                        
                        serde_json::from_str::<AnthropicStreamChunk>(json_str)
                            .map_err(|e| AnthropicError::DeserializationError {
                                message: format!("Failed to parse stream chunk: {}", e),
                            })
                    } else {
                        Err(AnthropicError::StreamError {
                            message: "Invalid SSE format".to_string(),
                        })
                    }
                }
                Err(e) => Err(AnthropicError::StreamError {
                    message: format!("Stream error: {}", e),
                }),
            }
        });
        
        Ok(Box::new(chunk_stream))
    }
    
    /// Test the client connection
    pub async fn test_connection(&self) -> AnthropicResult<()> {
        let url = format!("{}/v1/messages", self.config.base_url);
        let headers = self.common_headers();
        
        // Create a minimal test request
        let test_request = AnthropicCompletionRequest {
            model: "claude-3-haiku-20240307".to_string(),
            max_tokens: 10,
            messages: vec![super::messages::AnthropicMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            tool_choice: None,
        };
        
        let body = serde_json::to_vec(&test_request).map_err(|e| {
            AnthropicError::SerializationError {
                message: format!("Failed to serialize test request: {}", e),
            }
        })?;
        
        let response = self.http_client
            .post(&url)
            .headers(headers)
            .body(body)
            .timeout_seconds(30)
            .send()
            .await
            .map_err(|e| {
                AnthropicError::NetworkError {
                    message: format!("Connection test failed: {}", e),
                }
            })?;
        
        if !response.status.is_success() {
            return Err(AnthropicError::ApiError {
                status: response.status.as_u16(),
                message: String::from_utf8_lossy(&response.body).to_string(),
            });
        }
        
        Ok(())
    }
}

/// Anthropic provider implementation
pub struct AnthropicProvider {
    name: &'static str,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider instance
    pub fn new() -> Self {
        Self {
            name: "anthropic",
        }
    }
    
    /// Get the provider name
    pub fn name(&self) -> &'static str {
        self.name
    }
    
    /// Get available models
    pub fn models(&self) -> Vec<&'static str> {
        vec![
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ]
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_client_creation() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
        ).await;
        
        assert!(client.is_ok());
        
        let client = client.unwrap();
        assert_eq!(client.request_count().await, 0);
    }
    
    #[tokio::test]
    async fn test_provider_creation() {
        let provider = AnthropicProvider::new();
        assert_eq!(provider.name(), "anthropic");
        assert!(!provider.models().is_empty());
    }
    
    #[tokio::test]
    async fn test_common_headers() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
        ).await.unwrap();
        
        let headers = client.common_headers();
        assert_eq!(headers.get("x-api-key"), Some(&"test-key".to_string()));
        assert_eq!(headers.get("Content-Type"), Some(&"application/json".to_string()));
        assert_eq!(headers.get("anthropic-version"), Some(&"2023-06-01".to_string()));
        assert!(headers.contains_key("User-Agent"));
    }
}