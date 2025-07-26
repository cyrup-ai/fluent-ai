//! Provider-specific model types and traits for dynamic model discovery
//!
//! This module provides types and traits specifically for the model automation system
//! that dynamically fetches model information from providers at build-time and runtime.

use anyhow::Result;
use crate::model::info::ModelInfo;
use fluent_ai_async::AsyncStream;

/// Core trait for strongly-typed provider model enums
///
/// This trait is implemented by the build-time generated enums for each provider,
/// providing compile-time access to model metadata.
pub trait ProviderModel: Send + Sync + std::fmt::Debug + Clone + Copy + PartialEq + Eq + std::hash::Hash + 'static {
    /// Get the model's name/identifier
    fn name(&self) -> &'static str;
    
    /// Get the maximum context length in tokens
    fn max_context_length(&self) -> u64;
    
    /// Get the input pricing per million tokens
    fn pricing_input(&self) -> f64;
    
    /// Get the output pricing per million tokens
    fn pricing_output(&self) -> f64;
    
    /// Check if this is a thinking/reasoning model
    fn is_thinking(&self) -> bool;
    
    /// Get the required temperature for this model (if any)
    fn required_temperature(&self) -> Option<f64>;
}

/// Simplified ModelInfo for provider automation system
///
/// This is a lightweight version of ModelInfo specifically for the model automation
/// system that focuses on the essential metadata needed for model selection.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderModelInfo {
    /// The model's name/identifier
    pub name: String,
    /// Maximum context length in tokens
    pub max_context: u64,
    /// Input pricing per million tokens
    pub pricing_input: f64,
    /// Output pricing per million tokens
    pub pricing_output: f64,
    /// Whether this is a thinking/reasoning model
    pub is_thinking: bool,
    /// Required temperature for this model (if any)
    pub required_temperature: Option<f64>,
}

impl ProviderModelInfo {
    /// Create a new ProviderModelInfo
    pub fn new(
        name: String,
        max_context: u64,
        pricing_input: f64,
        pricing_output: f64,
        is_thinking: bool,
        required_temperature: Option<f64>,
    ) -> Self {
        Self {
            name,
            max_context,
            pricing_input,
            pricing_output,
            is_thinking,
            required_temperature,
        }
    }

    /// Convert to the full domain ModelInfo
    pub fn to_domain_model_info(&self, provider_name: &'static str) -> ModelInfo {
        use std::num::NonZeroU32;
        
        ModelInfo {
            provider_name,
            name: Box::leak(self.name.clone().into_boxed_str()),
            max_input_tokens: NonZeroU32::new(self.max_context as u32),
            max_output_tokens: NonZeroU32::new((self.max_context / 4) as u32), // Rough estimate
            input_price: Some(self.pricing_input),
            output_price: Some(self.pricing_output),
            supports_vision: false, // Default, can be overridden
            supports_function_calling: true, // Default for most modern models
            supports_streaming: true, // Default for most modern models
            supports_embeddings: false, // Default, specific to embedding models
            requires_max_tokens: false, // Default
            supports_thinking: self.is_thinking,
            optimal_thinking_budget: if self.is_thinking { Some(100_000) } else { None },
            system_prompt_prefix: None,
            real_name: None,
            model_type: None,
            patch: None,
        }
    }
}

/// Trait for runtime provider implementations
///
/// This trait is implemented by runtime provider clients that can fetch
/// model information from provider APIs using HTTP3 and AsyncStream patterns.
pub trait ProviderTrait: Send + Sync + std::fmt::Debug + 'static {
    /// Get model information for a specific model
    ///
    /// Returns an AsyncStream that yields the model information.
    /// This follows the fluent-ai AsyncStream pattern for all async operations.
    fn get_model_info(&self, model: &str) -> AsyncStream<ProviderModelInfo>;
    
    /// Get all available models from this provider
    ///
    /// Returns an AsyncStream that yields model information for all available models.
    fn list_models(&self) -> AsyncStream<ProviderModelInfo>;
    
    /// Get the provider name
    fn provider_name(&self) -> &'static str;
}

/// Error types specific to provider operations
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// Model was not found
    #[error("Model '{model}' not found for provider '{provider}'")]
    ModelNotFound { provider: String, model: String },
    
    /// Provider API error
    #[error("Provider API error: {message}")]
    ApiError { message: String },
    
    /// Authentication error
    #[error("Authentication failed for provider '{provider}': {message}")]
    AuthenticationFailed { provider: String, message: String },
    
    /// Network error
    #[error("Network error: {message}")]
    NetworkError { message: String },
    
    /// Parsing error
    #[error("Failed to parse response: {message}")]
    ParseError { message: String },
    
    /// Rate limit exceeded
    #[error("Rate limit exceeded for provider '{provider}'")]
    RateLimitExceeded { provider: String },
    
    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },
}

impl ProviderError {
    /// Create a model not found error
    pub fn model_not_found(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self::ModelNotFound {
            provider: provider.into(),
            model: model.into(),
        }
    }
    
    /// Create an API error
    pub fn api_error(message: impl Into<String>) -> Self {
        Self::ApiError {
            message: message.into(),
        }
    }
    
    /// Create an authentication error
    pub fn auth_error(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::AuthenticationFailed {
            provider: provider.into(),
            message: message.into(),
        }
    }
    
    /// Create a network error
    pub fn network_error(message: impl Into<String>) -> Self {
        Self::NetworkError {
            message: message.into(),
        }
    }
    
    /// Create a parse error
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::ParseError {
            message: message.into(),
        }
    }
    
    /// Create a rate limit error
    pub fn rate_limit(provider: impl Into<String>) -> Self {
        Self::RateLimitExceeded {
            provider: provider.into(),
        }
    }
    
    /// Create a configuration error
    pub fn config_error(message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            message: message.into(),
        }
    }
}

/// Result type for provider operations
pub type ProviderResult<T> = Result<T, ProviderError>;