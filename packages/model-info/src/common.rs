use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};

pub trait Model:
    Send + Sync + std::fmt::Debug + Clone + Copy + PartialEq + Eq + std::hash::Hash + 'static
{
    // Basic model information
    fn name(&self) -> &'static str;
    fn provider_name(&self) -> &'static str;

    // Token limits (split max_context_length into input/output)
    fn max_input_tokens(&self) -> Option<u32>;
    fn max_output_tokens(&self) -> Option<u32>;

    // Convenience method: total token capacity (input + output)
    fn max_context_length(&self) -> u64 {
        let input = self.max_input_tokens().unwrap_or(4096) as u64;
        let output = self.max_output_tokens().unwrap_or(2048) as u64;
        input + output
    }

    // Pricing (made optional to handle unknown pricing)
    fn pricing_input(&self) -> Option<f64>;
    fn pricing_output(&self) -> Option<f64>;

    // Capability methods
    fn supports_vision(&self) -> bool;
    fn supports_function_calling(&self) -> bool;
    fn supports_embeddings(&self) -> bool;
    fn requires_max_tokens(&self) -> bool;
    fn supports_thinking(&self) -> bool;
    fn required_temperature(&self) -> Option<f64>;

    // Advanced features
    fn optimal_thinking_budget(&self) -> Option<u32> {
        None
    }

    fn system_prompt_prefix(&self) -> Option<&'static str> {
        None
    }

    fn real_name(&self) -> Option<&'static str> {
        None
    }

    fn model_type(&self) -> Option<&'static str> {
        None
    }

    // Helper method to convert to ModelInfo
    fn to_model_info(&self) -> ModelInfo {
        ModelInfo {
            provider_name: self.provider_name(),
            name: self.name(),
            max_input_tokens: self.max_input_tokens().and_then(NonZeroU32::new),
            max_output_tokens: self.max_output_tokens().and_then(NonZeroU32::new),
            input_price: self.pricing_input(),
            output_price: self.pricing_output(),
            supports_vision: self.supports_vision(),
            supports_function_calling: self.supports_function_calling(),
            supports_embeddings: self.supports_embeddings(),
            requires_max_tokens: self.requires_max_tokens(),
            supports_thinking: self.supports_thinking(),
            optimal_thinking_budget: self.optimal_thinking_budget(),
            system_prompt_prefix: self.system_prompt_prefix().map(|s| s.to_string()),
            real_name: self.real_name().map(|s| s.to_string()),
            model_type: self.model_type().map(|s| s.to_string()),
            patch: None,
            required_temperature: self.required_temperature(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[must_use = "ModelInfo should be used to make informed decisions about model selection"]
pub struct ModelInfo {
    // Core identification
    pub provider_name: &'static str,
    pub name: &'static str,

    // Token limits
    pub max_input_tokens: Option<NonZeroU32>,
    pub max_output_tokens: Option<NonZeroU32>,

    // Pricing (optional to handle unknown pricing)
    pub input_price: Option<f64>,
    pub output_price: Option<f64>,

    // Capability flags
    #[serde(default)]
    pub supports_vision: bool,
    #[serde(default)]
    pub supports_function_calling: bool,
    #[serde(default)]
    pub supports_embeddings: bool,
    #[serde(default)]
    pub requires_max_tokens: bool,
    #[serde(default)]
    pub supports_thinking: bool,

    // Advanced features
    pub optimal_thinking_budget: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub model_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch: Option<serde_json::Value>,
    pub required_temperature: Option<f64>,
}

/// Error types for model operations
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model not found: {provider}:{name}")]
    ModelNotFound { provider: String, name: String },

    #[error("Model already exists: {provider}:{name}")]
    ModelAlreadyExists { provider: String, name: String },

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Provider not found: {0}")]
    ProviderNotFound(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}

/// Result type for model operations
pub type Result<T> = std::result::Result<T, ModelError>;

/// Model capabilities for filtering and querying
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelCapabilities {
    pub supports_vision: bool,
    pub supports_function_calling: bool,
    pub supports_fine_tuning: bool,
    pub supports_batch_processing: bool,
    pub supports_realtime: bool,
    pub supports_multimodal: bool,
    pub supports_thinking: bool,
    pub supports_embedding: bool,
    pub supports_code_completion: bool,
    pub supports_chat: bool,
    pub supports_instruction_following: bool,
    pub supports_few_shot_learning: bool,
    pub supports_zero_shot_learning: bool,
    pub has_long_context: bool,
    pub is_low_latency: bool,
    pub is_high_throughput: bool,
    pub supports_quantization: bool,
    pub supports_distillation: bool,
    pub supports_pruning: bool,
}

impl ModelInfo {
    /// Get the full model identifier as "provider:name"
    #[inline]
    pub fn id(&self) -> &'static str {
        self.name
    }

    /// Get the provider name
    #[inline]
    pub fn provider(&self) -> &'static str {
        self.provider_name
    }

    /// Get the model name
    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Check if the model supports vision
    #[inline]
    pub fn has_vision(&self) -> bool {
        self.supports_vision
    }

    /// Check if the model supports function calling
    #[inline]
    pub fn has_function_calling(&self) -> bool {
        self.supports_function_calling
    }

    /// Check if the model supports embeddings
    #[inline]
    pub fn has_embeddings(&self) -> bool {
        self.supports_embeddings
    }

    /// Check if the model requires max_tokens to be specified
    #[inline]
    pub fn requires_max_tokens(&self) -> bool {
        self.requires_max_tokens
    }

    /// Check if the model supports thinking/reasoning
    #[inline]
    pub fn has_thinking(&self) -> bool {
        self.supports_thinking
    }

    /// Get the optimal thinking budget if supported
    #[inline]
    pub fn thinking_budget(&self) -> Option<u32> {
        self.optimal_thinking_budget
    }

    /// Get the price for a given number of input tokens
    #[inline]
    pub fn price_for_input(&self, tokens: u32) -> Option<f64> {
        self.input_price
            .map(|price| (price * tokens as f64) / 1_000_000.0)
    }

    /// Get the price for a given number of output tokens
    #[inline]
    pub fn price_for_output(&self, tokens: u32) -> Option<f64> {
        self.output_price
            .map(|price| (price * tokens as f64) / 1_000_000.0)
    }

    /// Convert to ModelCapabilities for filtering and querying
    pub fn to_capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_vision: self.supports_vision,
            supports_function_calling: self.supports_function_calling,
            supports_fine_tuning: false,      // Not in ModelInfo yet
            supports_batch_processing: false, // Not in ModelInfo yet
            supports_realtime: false,         // Not in ModelInfo yet
            supports_multimodal: self.supports_vision, // Map vision to multimodal
            supports_thinking: self.supports_thinking,
            supports_embedding: self.supports_embeddings,
            supports_code_completion: false, // Not in ModelInfo yet
            supports_chat: true,             // Assume all models support chat
            supports_instruction_following: true, // Assume all models support instructions
            supports_few_shot_learning: true, // Assume all models support few-shot
            supports_zero_shot_learning: true, // Assume all models support zero-shot
            has_long_context: self
                .max_input_tokens
                .is_some_and(|tokens| tokens.get() > 32000),
            is_low_latency: false,        // Not in ModelInfo yet
            is_high_throughput: false,    // Not in ModelInfo yet
            supports_quantization: false, // Not in ModelInfo yet
            supports_distillation: false, // Not in ModelInfo yet
            supports_pruning: false,      // Not in ModelInfo yet
        }
    }

    /// Validate the model configuration
    pub fn validate(&self) -> Result<()> {
        if self.provider_name.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "provider_name cannot be empty".into(),
            ));
        }

        if self.name.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "name cannot be empty".into(),
            ));
        }

        if let Some(max_input) = self.max_input_tokens
            && max_input.get() == 0
        {
            return Err(ModelError::InvalidConfiguration(
                "max_input_tokens cannot be zero".into(),
            ));
        }

        if let Some(max_output) = self.max_output_tokens
            && max_output.get() == 0
        {
            return Err(ModelError::InvalidConfiguration(
                "max_output_tokens cannot be zero".into(),
            ));
        }

        if self.supports_thinking && self.optimal_thinking_budget.is_none() {
            return Err(ModelError::InvalidConfiguration(
                "optimal_thinking_budget must be set when supports_thinking is true".into(),
            ));
        }

        Ok(())
    }

    /// Create a new builder
    pub fn builder() -> ModelInfoBuilder {
        ModelInfoBuilder::new()
    }
}

impl Hash for ModelInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.provider_name.hash(state);
        self.name.hash(state);
    }
}

/// Builder for creating ModelInfo instances
#[derive(Debug, Clone, Default)]
pub struct ModelInfoBuilder {
    provider_name: Option<&'static str>,
    name: Option<&'static str>,
    max_input_tokens: Option<NonZeroU32>,
    max_output_tokens: Option<NonZeroU32>,
    input_price: Option<f64>,
    output_price: Option<f64>,
    supports_vision: bool,
    supports_function_calling: bool,
    supports_embeddings: bool,
    requires_max_tokens: bool,
    supports_thinking: bool,
    optimal_thinking_budget: Option<u32>,
    system_prompt_prefix: Option<String>,
    real_name: Option<String>,
    model_type: Option<String>,
    patch: Option<serde_json::Value>,
    required_temperature: Option<f64>,
}

impl ModelInfoBuilder {
    /// Create a new ModelInfoBuilder
    pub fn new() -> Self {
        Self { ..Self::default() }
    }

    /// Set the provider name
    pub fn provider_name(mut self, provider_name: &'static str) -> Self {
        self.provider_name = Some(provider_name);
        self
    }

    /// Set the model name
    pub fn name(mut self, name: &'static str) -> Self {
        self.name = Some(name);
        self
    }

    /// Set maximum input tokens
    pub fn max_input_tokens(mut self, tokens: u32) -> Self {
        self.max_input_tokens = NonZeroU32::new(tokens);
        self
    }

    /// Set maximum output tokens
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = NonZeroU32::new(tokens);
        self
    }

    /// Set pricing per 1M tokens
    pub fn pricing(mut self, input: f64, output: f64) -> Self {
        self.input_price = Some(input);
        self.output_price = Some(output);
        self
    }

    /// Enable vision support
    pub fn with_vision(mut self, enabled: bool) -> Self {
        self.supports_vision = enabled;
        self
    }

    /// Enable function calling support
    pub fn with_function_calling(mut self, enabled: bool) -> Self {
        self.supports_function_calling = enabled;
        self
    }

    /// Enable embeddings support
    pub fn with_embeddings(mut self, enabled: bool) -> Self {
        self.supports_embeddings = enabled;
        self
    }

    /// Set requires max tokens
    pub fn requires_max_tokens(mut self, required: bool) -> Self {
        self.requires_max_tokens = required;
        self
    }

    /// Enable thinking support with budget
    pub fn with_thinking(mut self, budget: u32) -> Self {
        self.supports_thinking = true;
        self.optimal_thinking_budget = Some(budget);
        self
    }

    /// Set system prompt prefix
    pub fn system_prompt_prefix(mut self, prefix: String) -> Self {
        self.system_prompt_prefix = Some(prefix);
        self
    }

    /// Set real name
    pub fn real_name(mut self, name: String) -> Self {
        self.real_name = Some(name);
        self
    }

    /// Set model type
    pub fn model_type(mut self, model_type: String) -> Self {
        self.model_type = Some(model_type);
        self
    }

    /// Set patch configuration
    pub fn patch(mut self, patch: serde_json::Value) -> Self {
        self.patch = Some(patch);
        self
    }

    /// Set required temperature
    pub fn required_temperature(mut self, temp: f64) -> Self {
        self.required_temperature = Some(temp);
        self
    }

    /// Build the ModelInfo instance
    pub fn build(self) -> Result<ModelInfo> {
        let provider_name = self
            .provider_name
            .ok_or_else(|| ModelError::InvalidConfiguration("provider_name is required".into()))?;

        let name = self
            .name
            .ok_or_else(|| ModelError::InvalidConfiguration("name is required".into()))?;

        if self.supports_thinking && self.optimal_thinking_budget.is_none() {
            return Err(ModelError::InvalidConfiguration(
                "optimal_thinking_budget must be set when supports_thinking is true".into(),
            ));
        }

        let model_info = ModelInfo {
            provider_name,
            name,
            max_input_tokens: self.max_input_tokens,
            max_output_tokens: self.max_output_tokens,
            input_price: self.input_price,
            output_price: self.output_price,
            supports_vision: self.supports_vision,
            supports_function_calling: self.supports_function_calling,
            supports_embeddings: self.supports_embeddings,
            requires_max_tokens: self.requires_max_tokens,
            supports_thinking: self.supports_thinking,
            optimal_thinking_budget: self.optimal_thinking_budget,
            system_prompt_prefix: self.system_prompt_prefix,
            real_name: self.real_name,
            model_type: self.model_type,
            patch: self.patch,
            required_temperature: self.required_temperature,
        };

        model_info.validate()?;
        Ok(model_info)
    }
}

/// A collection of model information for a specific provider
#[derive(Debug, Clone, Default)]
pub struct ProviderModels {
    provider_name: &'static str,
    models: Vec<ModelInfo>,
}

impl ProviderModels {
    /// Create a new provider model collection
    #[inline]
    pub fn new(provider_name: &'static str) -> Self {
        Self {
            provider_name,
            models: Vec::new(),
        }
    }

    /// Add a model to the collection
    pub fn add_model(&mut self, model: ModelInfo) -> Result<()> {
        if model.provider_name != self.provider_name {
            return Err(ModelError::InvalidConfiguration(
                "model provider does not match collection provider".into(),
            ));
        }

        if self.models.iter().any(|m| m.name == model.name) {
            return Err(ModelError::ModelAlreadyExists {
                provider: self.provider_name.into(),
                name: model.name.into(),
            });
        }

        self.models.push(model);
        Ok(())
    }

    /// Get a model by name
    #[inline]
    pub fn get(&self, name: &str) -> Option<&ModelInfo> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Get all models
    #[inline]
    pub fn all(&self) -> &[ModelInfo] {
        &self.models
    }

    /// Get the provider name
    #[inline]
    pub fn provider_name(&self) -> &'static str {
        self.provider_name
    }
}

pub trait ProviderTrait {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo>;
    fn list_models(&self) -> AsyncStream<ModelInfo>;
    fn provider_name(&self) -> &'static str;
}
