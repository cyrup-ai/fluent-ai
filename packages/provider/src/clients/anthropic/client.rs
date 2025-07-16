//! Zero-allocation Anthropic client with modular architecture
//!
//! This module provides a blazing-fast Anthropic client that orchestrates
//! the modular components with zero allocations and no locking.

use crate::http::{HttpClient, HttpError};
use super::config::AnthropicConfig;
use super::requests::AnthropicRequestBuilder;
use super::responses::AnthropicResponseProcessor;
use super::completion::{AnthropicCompletionRequest, AnthropicCompletionResponse};
use super::error::{AnthropicError, AnthropicResult};
use super::streaming::AnthropicStreamChunk;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// High-performance Anthropic client with zero-allocation patterns
///
/// This client uses the HTTP3 library's streaming capabilities and modular
/// components for maximum performance with zero allocations after initialization.
#[derive(Debug)]
pub struct AnthropicClient {
    /// Immutable configuration
    config: AnthropicConfig,
    /// HTTP3 client for requests
    http_client: HttpClient,
    /// Atomic request counter (lock-free)
    request_count: Arc<AtomicU64>,
}

impl AnthropicClient {
    /// Create a new Anthropic client with the given API key
    #[inline]
    pub fn new(api_key: impl Into<String>) -> AnthropicResult<Self> {
        let config = AnthropicConfig::new(api_key)?;
        let http_client = HttpClient::for_provider("anthropic")
            .map_err(|e| AnthropicError::NetworkError {
                message: format!("Failed to create HTTP client: {}", e),
            })?;
        
        Ok(Self {
            config,
            http_client,
            request_count: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Create a new Anthropic client with custom configuration
    #[inline]
    pub fn with_config(config: AnthropicConfig) -> AnthropicResult<Self> {
        let http_client = HttpClient::for_provider("anthropic")
            .map_err(|e| AnthropicError::NetworkError {
                message: format!("Failed to create HTTP client: {}", e),
            })?;
        
        Ok(Self {
            config,
            http_client,
            request_count: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Create a client from environment variables
    #[inline]
    pub fn from_env() -> AnthropicResult<Self> {
        let config = super::config::from_env()?;
        Self::with_config(config)
    }

    /// Get the current request count (lock-free)
    #[inline]
    pub fn request_count(&self) -> u64 {
        self.request_count.load(Ordering::Relaxed)
    }

    /// Get the configuration
    #[inline]
    pub fn config(&self) -> &AnthropicConfig {
        &self.config
    }

    /// Increment request count (lock-free)
    #[inline]
    fn increment_request_count(&self) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Send a completion request using the modular components
    pub async fn send_completion(
        &self,
        request: &AnthropicCompletionRequest,
    ) -> AnthropicResult<AnthropicCompletionResponse> {
        self.increment_request_count();
        
        // Use the request builder to send the request
        let request_builder = AnthropicRequestBuilder::new(&self.config, &self.http_client);
        let response = request_builder.send_completion(request).await?;
        
        // Use the response processor to process the response
        AnthropicResponseProcessor::process_completion_response(response)
    }

    /// Send a streaming completion request using HTTP3 streaming
    pub async fn send_streaming_completion(
        &self,
        request: &AnthropicCompletionRequest,
    ) -> AnthropicResult<impl futures::Stream<Item = AnthropicResult<AnthropicStreamChunk>>> {
        self.increment_request_count();
        
        // Use the request builder to send the streaming request
        let request_builder = AnthropicRequestBuilder::new(&self.config, &self.http_client);
        let response = request_builder.send_streaming_completion(request).await?;
        
        // Use the response processor to process the streaming response
        let chunks = AnthropicResponseProcessor::process_streaming_response(response)?;
        
        // Convert the chunks into a stream
        let stream = futures::stream::iter(chunks.into_iter().map(Ok));
        
        Ok(stream)
    }

    /// Test the connection using the modular components
    pub async fn test_connection(&self) -> AnthropicResult<()> {
        let request_builder = AnthropicRequestBuilder::new(&self.config, &self.http_client);
        let response = request_builder.test_connection().await?;
        
        AnthropicResponseProcessor::process_test_response(response)
    }

    /// Get available models
    #[inline]
    pub fn models(&self) -> &[&'static str] {
        &[
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514", 
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ]
    }

    /// Get the provider name
    #[inline]
    pub fn provider_name(&self) -> &'static str {
        "anthropic"
    }

    /// Check if streaming is supported
    #[inline]
    pub fn supports_streaming(&self) -> bool {
        true
    }

    /// Get request statistics
    #[inline]
    pub fn stats(&self) -> AnthropicClientStats {
        AnthropicClientStats {
            requests_sent: self.request_count(),
            provider: self.provider_name(),
        }
    }
}

/// Client statistics (zero-allocation)
#[derive(Debug, Clone)]
pub struct AnthropicClientStats {
    /// Total requests sent
    pub requests_sent: u64,
    /// Provider name
    pub provider: &'static str,
}

impl AnthropicClientStats {
    /// Get requests per second (requires external timing)
    #[inline]
    pub fn requests_per_second(&self, elapsed_seconds: f64) -> f64 {
        if elapsed_seconds > 0.0 {
            self.requests_sent as f64 / elapsed_seconds
        } else {
            0.0
        }
    }
}

/// Anthropic provider metadata
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    name: &'static str,
    models: &'static [&'static str],
}

impl AnthropicProvider {
    /// Create a new provider instance
    #[inline]
    pub const fn new() -> Self {
        Self {
            name: "anthropic",
            models: &[
                "claude-opus-4-20250514",
                "claude-sonnet-4-20250514",
                "claude-3-5-sonnet-20241022", 
                "claude-3-5-haiku-20241022",
            ],
        }
    }

    /// Get the provider name
    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Get available models
    #[inline]
    pub fn models(&self) -> &[&'static str] {
        self.models
    }

    /// Check if a model is supported
    #[inline]
    pub fn supports_model(&self, model: &str) -> bool {
        self.models.iter().any(|&m| m == model)
    }
}

impl Default for AnthropicProvider {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to create a client with an API key
#[inline]
pub fn create_client(api_key: impl Into<String>) -> AnthropicResult<AnthropicClient> {
    AnthropicClient::new(api_key)
}

/// Convenience function to create a client from environment
#[inline]
pub fn create_client_from_env() -> AnthropicResult<AnthropicClient> {
    AnthropicClient::from_env()
}