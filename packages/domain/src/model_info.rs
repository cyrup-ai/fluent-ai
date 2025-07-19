//! Model information types for AI models
//!
//! This module contains the core data structures and functionality for working with
//! AI model metadata, including capabilities, pricing, and token limits.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::collections::HashMap;

/// Core metadata and capabilities for an AI model
/// 
/// This struct provides a standardized way to represent model capabilities,
/// limitations, and pricing information across different providers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[must_use = "ModelInfoData should be used to make informed decisions about model selection"]
pub struct ModelInfoData {
    /// The name of the provider (e.g., "openai", "anthropic", "google")
    pub provider_name: String,
    
    /// The name of the model (e.g., "gpt-4.1", "claude-opus-4-20250514")
    pub name: String,
    
    /// Maximum number of input tokens supported by the model
    pub max_input_tokens: Option<u64>,
    
    /// Maximum number of output tokens that can be generated
    pub max_output_tokens: Option<u64>,
    
    /// Price per 1M input tokens in USD
    pub input_price: Option<f64>,
    
    /// Price per 1M output tokens in USD
    pub output_price: Option<f64>,
    
    /// Whether the model supports image/video input (multimodal)
    pub supports_vision: Option<bool>,
    
    /// Whether the model supports function calling/tool use
    pub supports_function_calling: Option<bool>,
    
    /// Whether the model requires max_tokens to be specified
    pub require_max_tokens: Option<bool>,
    
    /// Whether the model supports thinking/reasoning capabilities
    pub supports_thinking: Option<bool>,
    
    /// Optimal thinking budget for this model in tokens
    pub optimal_thinking_budget: Option<u32>,
}

impl Default for ModelInfoData {
    /// Create a default ModelInfoData with safe defaults
    fn default() -> Self {
        Self {
            provider_name: String::new(),
            name: String::new(),
            max_input_tokens: None,
            max_output_tokens: None,
            input_price: None,
            output_price: None,
            supports_vision: Some(false),
            supports_function_calling: Some(false),
            require_max_tokens: Some(false),
            supports_thinking: Some(false),
            optimal_thinking_budget: Some(1024),
        }
    }
}

impl ModelInfoData {
    /// Create a new ModelInfoData with the minimum required fields
    #[inline]
    pub fn new(provider_name: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            provider_name: provider_name.into(),
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set the maximum input tokens for this model
    #[inline]
    pub fn with_max_input_tokens(mut self, tokens: u64) -> Self {
        self.max_input_tokens = Some(tokens);
        self
    }

    /// Set the maximum output tokens for this model
    #[inline]
    pub fn with_max_output_tokens(mut self, tokens: u64) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Set the pricing information for this model
    #[inline]
    pub fn with_pricing(mut self, input_price: f64, output_price: f64) -> Self {
        self.input_price = Some(input_price);
        self.output_price = Some(output_price);
        self
    }

    /// Enable or disable vision support for this model
    #[inline]
    pub fn with_vision_support(mut self, supports: bool) -> Self {
        self.supports_vision = Some(supports);
        self
    }

    /// Enable or disable function calling support for this model
    #[inline]
    pub fn with_function_calling(mut self, supports: bool) -> Self {
        self.supports_function_calling = Some(supports);
        self
    }

    /// Set whether max_tokens is required for this model
    #[inline]
    pub fn require_max_tokens(mut self, required: bool) -> Self {
        self.require_max_tokens = Some(required);
        self
    }

    /// Enable or disable thinking capabilities for this model
    #[inline]
    pub fn with_thinking_support(mut self, supports: bool, optimal_budget: Option<u32>) -> Self {
        self.supports_thinking = Some(supports);
        if let Some(budget) = optimal_budget {
            self.optimal_thinking_budget = Some(budget);
        }
        self
    }
}

/// A registry of model information for quick lookup
#[derive(Debug, Default)]
pub struct ModelRegistry {
    models: HashMap<String, ModelInfoData>,
}

impl ModelRegistry {
    /// Create a new empty model registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Add a model to the registry
    pub fn register(&mut self, model: ModelInfoData) -> &mut Self {
        let key = format!("{}:{}", model.provider_name, model.name);
        self.models.insert(key, model);
        self
    }

    /// Look up a model by provider and name
    pub fn get(&self, provider: &str, name: &str) -> Option<&ModelInfoData> {
        let key = format!("{}:{}", provider, name);
        self.models.get(&key)
    }

    /// Get all models from a specific provider
    pub fn get_by_provider(&self, provider: &str) -> Vec<&ModelInfoData> {
        self.models
            .iter()
            .filter(|(k, _)| k.starts_with(&format!("{}:", provider)))
            .map(|(_, v)| v)
            .collect()
    }
}

/// Global model registry instance
static MODEL_REGISTRY: OnceLock<ModelRegistry> = OnceLock::new();

/// Initialize the global model registry with the default models
pub fn init_model_registry() -> &'static ModelRegistry {
    MODEL_REGISTRY.get_or_init(|| {
        let mut registry = ModelRegistry::new();
        // Add default models here if needed
        registry
    })
}

/// Get model information by provider and name
/// 
/// Returns a reference to the model info if found, or None if not found
pub fn get_model_info(provider: &str, name: &str) -> Option<&'static ModelInfoData> {
    let registry = MODEL_REGISTRY.get_or_init(ModelRegistry::new);
    registry.get(provider, name)
}


