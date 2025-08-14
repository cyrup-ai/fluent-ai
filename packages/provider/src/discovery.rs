//! Provider discovery and model enumeration capabilities
//!
//! Provides utilities for discovering available models and providers
//! across all supported AI providers in the fluent-ai ecosystem.

use std::collections::HashMap;

use fluent_ai_domain::model::ModelInfo;
use serde::{Deserialize, Serialize};

/// Discovery error types
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Authentication error")]
    AuthError,
}

/// Result type for discovery operations
pub type DiscoveryResult<T> = Result<T, DiscoveryError>;

/// Provider model discovery trait
pub trait ProviderModelDiscovery {
    /// Discover available models for this provider
    fn discover_models(&self) -> DiscoveryResult<Vec<String>>;

    /// Get detailed model information
    fn get_model_info(&self, model_name: &str) -> DiscoveryResult<ModelInfo>;

    /// Check if a model is available
    fn is_model_available(&self, model_name: &str) -> bool;
}

/// Provider discovery information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Provider name
    pub name: String,
    /// Provider base URL
    pub base_url: String,
    /// Available models
    pub models: Vec<String>,
    /// Whether the provider requires an API key
    pub requires_api_key: bool,
    /// Provider capabilities
    pub capabilities: ProviderCapabilities,
}

/// Provider capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Supports text completion
    pub completion: bool,
    /// Supports embeddings
    pub embeddings: bool,
    /// Supports image generation
    pub image_generation: bool,
    /// Supports audio generation
    pub audio_generation: bool,
    /// Supports vision/image understanding
    pub vision: bool,
    /// Supports function calling/tools
    pub function_calling: bool,
    /// Supports streaming
    pub streaming: bool,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            completion: true,
            embeddings: false,
            image_generation: false,
            audio_generation: false,
            vision: false,
            function_calling: false,
            streaming: true,
        }
    }
}

/// Discover available providers
pub fn discover_providers() -> HashMap<String, ProviderInfo> {
    let mut providers = HashMap::new();

    // OpenAI
    providers.insert(
        "openai".to_string(),
        ProviderInfo {
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            models: vec![
                "gpt-4o".to_string(),
                "gpt-4o-mini".to_string(),
                "gpt-4-turbo".to_string(),
                "gpt-3.5-turbo".to_string(),
            ],
            requires_api_key: true,
            capabilities: ProviderCapabilities {
                completion: true,
                embeddings: true,
                image_generation: true,
                audio_generation: true,
                vision: true,
                function_calling: true,
                streaming: true,
            },
        },
    );

    // Anthropic
    providers.insert(
        "anthropic".to_string(),
        ProviderInfo {
            name: "Anthropic".to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            models: vec![
                "claude-3.5-sonnet".to_string(),
                "claude-3-opus".to_string(),
                "claude-3-haiku".to_string(),
            ],
            requires_api_key: true,
            capabilities: ProviderCapabilities {
                completion: true,
                vision: true,
                function_calling: true,
                streaming: true,
                ..Default::default()
            },
        },
    );

    // Mistral
    providers.insert(
        "mistral".to_string(),
        ProviderInfo {
            name: "Mistral AI".to_string(),
            base_url: "https://api.mistral.ai/v1".to_string(),
            models: vec![
                "mistral-large-latest".to_string(),
                "mistral-small-latest".to_string(),
                "codestral-latest".to_string(),
            ],
            requires_api_key: true,
            capabilities: ProviderCapabilities {
                completion: true,
                embeddings: true,
                function_calling: true,
                streaming: true,
                ..Default::default()
            },
        },
    );

    // HuggingFace
    providers.insert(
        "huggingface".to_string(),
        ProviderInfo {
            name: "HuggingFace".to_string(),
            base_url: "https://api-inference.huggingface.co".to_string(),
            models: vec![
                "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string(),
                "microsoft/Phi-4".to_string(),
                "google/gemma-2-9b-it".to_string(),
            ],
            requires_api_key: true,
            capabilities: ProviderCapabilities::default(),
        },
    );

    providers
}

/// Get provider information by name
pub fn get_provider_info(name: &str) -> Option<ProviderInfo> {
    discover_providers().get(name).cloned()
}

/// List all available models across all providers
pub fn list_all_models() -> Vec<(String, String)> {
    let mut models = Vec::new();

    for (provider_name, provider_info) in discover_providers() {
        for model in &provider_info.models {
            models.push((provider_name.clone(), model.clone()));
        }
    }

    models
}
