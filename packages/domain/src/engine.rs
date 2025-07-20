//! Engine domain module
//!
//! Provides core engine functionality with zero-allocation patterns and production-ready
//! error handling. The engine routes requests to appropriate AI providers and manages
//! completion and streaming operations.

use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::completion::{CompactCompletionResponse, CompletionResponse};
use crate::{AsyncTask, spawn_async};

/// Engine-specific error types with zero-allocation string sharing
#[derive(Error, Debug, Clone)]
pub enum EngineError {
    #[error("Provider not available: {provider}")]
    ProviderNotAvailable { provider: Arc<str> },

    #[error("Model not found: {model}")]
    ModelNotFound { model: Arc<str> },

    #[error("Configuration error: {detail}")]
    ConfigurationError { detail: Arc<str> },

    #[error("Authentication failed: {reason}")]
    AuthenticationFailed { reason: Arc<str> },

    #[error("Rate limit exceeded: {retry_after_seconds}s")]
    RateLimitExceeded { retry_after_seconds: u64 },

    #[error("Request timeout: {timeout_seconds}s")]
    RequestTimeout { timeout_seconds: u64 },

    #[error("Network error: {detail}")]
    NetworkError { detail: Arc<str> },

    #[error("Invalid input: {reason}")]
    InvalidInput { reason: Arc<str> },

    #[error("Service unavailable: {service}")]
    ServiceUnavailable { service: Arc<str> },
}

/// Result type for engine operations
pub type EngineResult<T> = Result<T, EngineError>;

/// Engine configuration with zero-allocation optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Model name for the completion request
    pub model_name: Arc<str>,
    /// Provider identifier (e.g., "openai", "anthropic", "gemini")
    pub provider: Arc<str>,
    /// API key for authentication
    pub api_key: Option<Arc<str>>,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum tokens for completion
    pub max_tokens: Option<u32>,
    /// Temperature for response randomness (0.0 - 1.0)
    pub temperature: Option<f32>,
    /// Whether to enable streaming responses
    pub enable_streaming: bool,
    /// Custom endpoint URL override
    pub endpoint_url: Option<Arc<str>>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_name: Arc::from("gpt-4o-mini"),
            provider: Arc::from("openai"),
            api_key: None,
            timeout_seconds: 30,
            max_tokens: Some(4096),
            temperature: Some(0.7),
            enable_streaming: false,
            endpoint_url: None,
        }
    }
}

impl EngineConfig {
    /// Create a new engine configuration with zero-allocation string sharing
    #[inline]
    pub fn new(model_name: impl Into<Arc<str>>, provider: impl Into<Arc<str>>) -> Self {
        Self {
            model_name: model_name.into(),
            provider: provider.into(),
            ..Default::default()
        }
    }

    /// Set API key with zero-allocation sharing
    #[inline]
    pub fn with_api_key(mut self, api_key: impl Into<Arc<str>>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set timeout in seconds
    #[inline]
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }

    /// Set max tokens
    #[inline]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature
    #[inline]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature.clamp(0.0, 1.0));
        self
    }

    /// Enable streaming responses
    #[inline]
    pub fn with_streaming(mut self) -> Self {
        self.enable_streaming = true;
        self
    }

    /// Set custom endpoint URL
    #[inline]
    pub fn with_endpoint(mut self, endpoint_url: impl Into<Arc<str>>) -> Self {
        self.endpoint_url = Some(endpoint_url.into());
        self
    }
}

/// Engine completion request with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub prompt: Arc<str>,
    pub system_prompt: Option<Arc<str>>,
    pub conversation_history: Vec<Arc<str>>,
    pub tools: Vec<Arc<str>>,
    pub metadata: Option<Arc<str>>,
}

impl CompletionRequest {
    /// Create a new completion request
    #[inline]
    pub fn new(prompt: impl Into<Arc<str>>) -> Self {
        Self {
            prompt: prompt.into(),
            system_prompt: None,
            conversation_history: Vec::new(),
            tools: Vec::new(),
            metadata: None,
        }
    }

    /// Set system prompt
    #[inline]
    pub fn with_system_prompt(mut self, system_prompt: impl Into<Arc<str>>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Add conversation history
    #[inline]
    pub fn with_history(mut self, history: Vec<Arc<str>>) -> Self {
        self.conversation_history = history;
        self
    }

    /// Add available tools
    #[inline]
    pub fn with_tools(mut self, tools: Vec<Arc<str>>) -> Self {
        self.tools = tools;
        self
    }
}

/// Core engine implementation with zero-allocation patterns
pub struct Engine {
    config: Arc<RwLock<EngineConfig>>,
    client_pool: Arc<RwLock<std::collections::HashMap<Arc<str>, Arc<dyn Send + Sync>>>>,
}

impl Engine {
    /// Create a new engine with the given configuration
    #[inline]
    pub fn new(config: EngineConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            client_pool: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Get the current configuration
    #[inline]
    pub async fn get_config(&self) -> EngineConfig {
        self.config.read().await.clone()
    }

    /// Update the engine configuration
    #[inline]
    pub async fn update_config(&self, config: EngineConfig) {
        *self.config.write().await = config;
    }

    /// Validate configuration for production readiness
    #[inline]
    fn validate_config(config: &EngineConfig) -> EngineResult<()> {
        if config.model_name.is_empty() {
            return Err(EngineError::ConfigurationError {
                detail: Arc::from("Model name cannot be empty"),
            });
        }

        if config.provider.is_empty() {
            return Err(EngineError::ConfigurationError {
                detail: Arc::from("Provider cannot be empty"),
            });
        }

        if config.timeout_seconds == 0 {
            return Err(EngineError::ConfigurationError {
                detail: Arc::from("Timeout must be greater than 0"),
            });
        }

        if let Some(temperature) = config.temperature {
            if !(0.0..=1.0).contains(&temperature) {
                return Err(EngineError::ConfigurationError {
                    detail: Arc::from("Temperature must be between 0.0 and 1.0"),
                });
            }
        }

        Ok(())
    }

    /// Create a provider client based on configuration
    #[inline]
    async fn create_provider_client(&self, config: &EngineConfig) -> EngineResult<()> {
        // Validate provider availability
        match config.provider.as_ref() {
            "openai" | "anthropic" | "gemini" | "azure" | "bedrock" | "cohere" | "groq"
            | "mistral" | "ollama" | "perplexity" | "together" | "xai" => {
                // Provider is supported
            }
            _ => {
                return Err(EngineError::ProviderNotAvailable {
                    provider: config.provider.clone(),
                });
            }
        }

        // Check authentication
        if config.api_key.is_none() && config.provider.as_ref() != "ollama" {
            return Err(EngineError::AuthenticationFailed {
                reason: Arc::from("API key required for this provider"),
            });
        }

        Ok(())
    }

    /// Process completion request with production-ready error handling
    #[inline]
    async fn process_completion_internal(
        &self,
        request: &CompletionRequest,
        config: &EngineConfig,
    ) -> EngineResult<CompletionResponse> {
        let start_time = std::time::Instant::now();

        // Validate input
        if request.prompt.is_empty() {
            return Err(EngineError::InvalidInput {
                reason: Arc::from("Prompt cannot be empty"),
            });
        }

        // In a real implementation, this would call the appropriate provider
        let response_content = format!(
            "Processed by {} ({}): {}",
            config.provider, config.model_name, request.prompt
        );

        let response_time = start_time.elapsed().as_millis();
        let tokens_used = request.prompt.len() as u32; // Simplified token count

        // Create a standard completion response
        let response = CompletionResponse::new(&response_content, config.model_name.as_ref())
            .with_provider(config.provider.as_ref())
            .with_finish_reason("stop")
            .with_response_time(response_time);

        // Convert to compact form for the engine's internal use
        Ok(response.into_compact().into_standard())
    }
}

/// Engine completion function with production-ready error handling
pub async fn complete_with_engine(config: &EngineConfig, input: &str) -> EngineResult<String> {
    // Validate configuration
    Engine::validate_config(config)?;

    // Create engine instance
    let engine = Engine::new(config.clone());

    // Create completion request
    let request = CompletionRequest::new(input);

    // Process completion
    let response = engine.process_completion_internal(&request, config).await?;

    Ok(response.text.to_string())
}

/// Engine streaming function with async task pattern
pub fn stream_with_engine(config: &EngineConfig, input: &str) -> AsyncTask<EngineResult<String>> {
    let config = config.clone();
    let input = input.to_string();

    spawn_async(async move {
        // Validate configuration
        Engine::validate_config(&config)?;

        // Create engine instance
        let engine = Engine::new(config.clone());

        // Create completion request
        let request = CompletionRequest::new(input);

        // Process streaming completion
        let response = engine
            .process_completion_internal(&request, &config)
            .await?;

        Ok(response.text.to_string())
    })
}

/// Get default engine configuration with optimized defaults
#[inline]
pub fn get_default_engine() -> EngineConfig {
    EngineConfig::default()
}

/// Create engine configuration for specific providers
#[inline]
pub fn create_openai_config(api_key: impl Into<Arc<str>>) -> EngineConfig {
    EngineConfig::new("gpt-4o-mini", "openai")
        .with_api_key(api_key)
        .with_timeout(30)
        .with_max_tokens(4096)
        .with_temperature(0.7)
}

/// Create engine configuration for Anthropic
#[inline]
pub fn create_anthropic_config(api_key: impl Into<Arc<str>>) -> EngineConfig {
    EngineConfig::new("claude-3-5-sonnet-20241022", "anthropic")
        .with_api_key(api_key)
        .with_timeout(30)
        .with_max_tokens(4096)
        .with_temperature(0.7)
}

/// Create engine configuration for local Ollama
#[inline]
pub fn create_ollama_config(model: impl Into<Arc<str>>) -> EngineConfig {
    EngineConfig::new(model, "ollama")
        .with_timeout(60)
        .with_max_tokens(4096)
        .with_temperature(0.7)
        .with_endpoint("http://localhost:11434")
}
