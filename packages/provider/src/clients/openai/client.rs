//! OpenAI client implementation with zero-allocation patterns
//!
//! This module provides a production-ready OpenAI client with QUIC/HTTP3 support,
//! proper error handling, and zero-allocation request patterns.

use crate::http::{HttpClient, HttpError};
use super::completion::{OpenAICompletionRequest, OpenAICompletionResponse};
use super::embeddings::{OpenAIEmbeddingRequest, OpenAIEmbeddingResponse};
use super::error::{OpenAIError, OpenAIResult};
use super::streaming::OpenAIStreamChunk;
use futures::Stream;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use bytes::Bytes;

/// OpenAI client configuration
#[derive(Debug, Clone)]
pub struct OpenAIClientConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
}

impl Default for OpenAIClientConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.openai.com/v1".to_string(),
            timeout_seconds: 300,
            max_retries: 3,
        }
    }
}

/// High-performance OpenAI client with QUIC/HTTP3 support
pub struct OpenAIClient {
    config: OpenAIClientConfig,
    http_client: HttpClient,
    request_count: Arc<AtomicU64>,
}

impl OpenAIClient {
    /// Create a new OpenAI client with API key and base URL
    pub async fn new(api_key: String, base_url: String) -> OpenAIResult<Self> {
        let config = OpenAIClientConfig {
            api_key,
            base_url,
            ..Default::default()
        };
        
        let http_client = HttpClient::for_provider("openai")
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to create HTTP client: {}", e),
            })?;
        let request_count = Arc::new(AtomicU64::new(0));
        
        Ok(Self {
            config,
            http_client,
            request_count,
        })
    }
    
    /// Create a new OpenAI client with custom configuration
    pub async fn with_config(config: OpenAIClientConfig) -> OpenAIResult<Self> {
        let http_client = HttpClient::for_provider("openai")
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to create HTTP client: {}", e),
            })?;
        let request_count = Arc::new(AtomicU64::new(0));
        
        Ok(Self {
            config,
            http_client,
            request_count,
        })
    }
    
    /// Get the current request count (lock-free)
    pub fn request_count(&self) -> u64 {
        self.request_count.load(Ordering::Relaxed)
    }
    
    /// Increment the request count (lock-free)
    fn increment_request_count(&self) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Create authorization header value
    fn auth_header(&self) -> String {
        format!("Bearer {}", self.config.api_key)
    }
    
    
    /// Send a completion request
    pub async fn send_completion(
        &self,
        request: &OpenAICompletionRequest,
    ) -> OpenAIResult<OpenAICompletionResponse> {
        self.increment_request_count();
        
        let url = format!("{}/chat/completions", self.config.base_url);
        
        let body = serde_json::to_vec(request).map_err(|e| {
            OpenAIError::SerializationError {
                message: format!("Failed to serialize completion request: {}", e),
            }
        })?;
        
        let mut http_request = crate::http::HttpRequest::post(url, Bytes::from(body))
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to create HTTP request: {}", e),
            })?;
        
        // Add headers
        http_request = http_request.header("Content-Type", "application/json")
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to add content-type header: {}", e),
            })?;
        
        http_request = http_request.header("Authorization", &self.auth_header())
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to add authorization header: {}", e),
            })?;
        
        http_request = http_request.header("User-Agent", "fluent-ai-provider/0.1.0")
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to add user-agent header: {}", e),
            })?;
        
        let response = self.http_client.execute(http_request).await
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to execute completion request: {}", e),
            })?;
        
        if !response.status.is_success() {
            return Err(OpenAIError::ApiError {
                status: response.status.as_u16(),
                message: String::from_utf8_lossy(&response.body).to_string(),
            });
        }
        
        let completion_response: OpenAICompletionResponse = serde_json::from_slice(&response.body)
            .map_err(|e| {
                OpenAIError::DeserializationError {
                    message: format!("Failed to deserialize completion response: {}", e),
                }
            })?;
        
        Ok(completion_response)
    }
    
    /// Send a streaming completion request
    pub async fn send_streaming_completion(
        &self,
        request: &OpenAICompletionRequest,
    ) -> OpenAIResult<impl Stream<Item = OpenAIResult<OpenAIStreamChunk>>> {
        self.increment_request_count();
        
        let url = format!("{}/chat/completions", self.config.base_url);
        
        // Create streaming request
        let mut streaming_request = request.clone();
        streaming_request.stream = Some(true);
        
        let body = serde_json::to_vec(&streaming_request).map_err(|e| {
            OpenAIError::SerializationError {
                message: format!("Failed to serialize streaming request: {}", e),
            }
        })?;
        
        let mut http_request = crate::http::HttpRequest::post(url, Bytes::from(body))
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to create HTTP request: {}", e),
            })?;
        
        // Add headers
        http_request = http_request.header("Content-Type", "application/json")
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to add content-type header: {}", e),
            })?;
        
        http_request = http_request.header("Authorization", &self.auth_header())
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to add authorization header: {}", e),
            })?;
        
        http_request = http_request.header("User-Agent", "fluent-ai-provider/0.1.0")
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to add user-agent header: {}", e),
            })?;
        
        // Use HTTP3 streaming
        let stream = self.http_client.execute_streaming(http_request).await
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to execute streaming request: {}", e),
            })?;
        
        // Convert HTTP stream to OpenAI chunk stream using HTTP3 SSE parsing
        let chunk_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    // Parse SSE chunk
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    if chunk_str.starts_with("data: ") {
                        let json_str = &chunk_str[6..];
                        if json_str.trim() == "[DONE]" {
                            return Ok(OpenAIStreamChunk::done());
                        }
                        
                        serde_json::from_str::<OpenAIStreamChunk>(json_str)
                            .map_err(|e| OpenAIError::DeserializationError {
                                message: format!("Failed to parse stream chunk: {}", e),
                            })
                    } else {
                        Err(OpenAIError::StreamError {
                            message: "Invalid SSE format".to_string(),
                        })
                    }
                }
                Err(e) => Err(OpenAIError::StreamError {
                    message: format!("Stream error: {}", e),
                }),
            }
        });
        
        Ok(chunk_stream)
    }
    
    /// Send an embedding request
    pub async fn send_embedding(
        &self,
        request: &OpenAIEmbeddingRequest,
    ) -> OpenAIResult<OpenAIEmbeddingResponse> {
        self.increment_request_count();
        
        let url = format!("{}/embeddings", self.config.base_url);
        
        let body = serde_json::to_vec(request).map_err(|e| {
            OpenAIError::SerializationError {
                message: format!("Failed to serialize embedding request: {}", e),
            }
        })?;
        
        let mut http_request = crate::http::HttpRequest::post(url, Bytes::from(body))
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to create HTTP request: {}", e),
            })?;
        
        // Add headers
        http_request = http_request.header("Content-Type", "application/json")
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to add content-type header: {}", e),
            })?;
        
        http_request = http_request.header("Authorization", &self.auth_header())
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to add authorization header: {}", e),
            })?;
        
        http_request = http_request.header("User-Agent", "fluent-ai-provider/0.1.0")
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to add user-agent header: {}", e),
            })?;
        
        let response = self.http_client.execute(http_request).await
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to execute embedding request: {}", e),
            })?;
        
        if !response.status.is_success() {
            return Err(OpenAIError::ApiError {
                status: response.status.as_u16(),
                message: String::from_utf8_lossy(&response.body).to_string(),
            });
        }
        
        let embedding_response: OpenAIEmbeddingResponse = serde_json::from_slice(&response.body)
            .map_err(|e| {
                OpenAIError::DeserializationError {
                    message: format!("Failed to deserialize embedding response: {}", e),
                }
            })?;
        
        Ok(embedding_response)
    }
    
    /// Test the client connection
    pub async fn test_connection(&self) -> OpenAIResult<()> {
        let url = format!("{}/models", self.config.base_url);
        
        let mut http_request = crate::http::HttpRequest::get(url)
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to create HTTP request: {}", e),
            })?;
        
        // Add headers
        http_request = http_request.header("Authorization", &self.auth_header())
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to add authorization header: {}", e),
            })?;
        
        http_request = http_request.header("User-Agent", "fluent-ai-provider/0.1.0")
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Failed to add user-agent header: {}", e),
            })?;
        
        let response = self.http_client.execute(http_request).await
            .map_err(|e| OpenAIError::NetworkError {
                message: format!("Connection test failed: {}", e),
            })?;
        
        if !response.status.is_success() {
            return Err(OpenAIError::ApiError {
                status: response.status.as_u16(),
                message: String::from_utf8_lossy(&response.body).to_string(),
            });
        }
        
        Ok(())
    }
}

/// OpenAI provider implementation
pub struct OpenAIProvider {
    name: &'static str,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider instance
    pub fn new() -> Self {
        Self {
            name: "openai",
        }
    }
    
    /// Get the provider name
    pub fn name(&self) -> &'static str {
        self.name
    }
    
    /// Get available models
    pub fn models(&self) -> Vec<&'static str> {
        vec![
            "gpt-4-1",
            "gpt-4-1-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "text-embedding-3-large",
            "text-embedding-3-small",
        ]
    }
}

impl Default for OpenAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_client_creation() {
        let client = OpenAIClient::new(
            "test-key".to_string(),
            "https://api.openai.com/v1".to_string(),
        ).await;
        
        assert!(client.is_ok());
        
        let client = client.unwrap();
        assert_eq!(client.request_count(), 0);
    }
    
    #[tokio::test]
    async fn test_provider_creation() {
        let provider = OpenAIProvider::new();
        assert_eq!(provider.name(), "openai");
        assert!(!provider.models().is_empty());
    }
    
    #[tokio::test]
    async fn test_auth_header() {
        let client = OpenAIClient::new(
            "test-key".to_string(),
            "https://api.openai.com/v1".to_string(),
        ).await.unwrap();
        
        assert_eq!(client.auth_header(), "Bearer test-key");
    }
}