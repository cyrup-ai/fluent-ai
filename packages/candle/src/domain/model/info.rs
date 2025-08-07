//! Model information types and utilities

// Removed unused import: std::borrow::Cow
// Removed unused import: std::fmt
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};

// Removed unused import: smallvec::SmallVec
use crate::domain::model::error::{CandleModelError, CandleResult};

/// Core metadata and capabilities for a Candle AI model
///
/// This struct provides a standardized way to represent Candle model capabilities,
/// limitations, and metadata across different providers.
///
/// **IMPORTANT**: This struct deserializes directly from the external models.yaml
/// file curated by sigoden on GitHub. The field names and structure must match
/// that YAML format exactly. CandleModelInfo is the single source of truth for model data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[must_use = "CandleModelInfo should be used to make informed decisions about model selection"]
pub struct CandleModelInfo {
    /// The name of the provider (e.g., "openai", "anthropic", "google")
    pub provider_name: &'static str,

    /// The name of the model (e.g., "gpt-4.1", "claude-opus-4-20250514")
    pub name: &'static str,

    /// Maximum number of input tokens supported by the model
    pub max_input_tokens: Option<NonZeroU32>,

    /// Maximum number of output tokens that can be generated
    pub max_output_tokens: Option<NonZeroU32>,

    /// Price per 1M input tokens in USD (if known)
    pub input_price: Option<f64>,

    /// Price per 1M output tokens in USD (if known)
    pub output_price: Option<f64>,

    /// Whether the model supports image/video input (multimodal)
    #[serde(default)]
    pub supports_vision: bool,

    /// Whether the model supports function calling/tool use
    #[serde(default)]
    pub supports_function_calling: bool,

    /// Whether the model supports streaming responses
    #[serde(default)]
    pub supports_streaming: bool,

    /// Whether the model supports embeddings
    #[serde(default)]
    pub supports_embeddings: bool,

    /// Whether the model requires max_tokens to be specified
    #[serde(default)]
    pub requires_max_tokens: bool,

    /// Whether the model supports thinking/reasoning capabilities
    #[serde(default)]
    pub supports_thinking: bool,

    /// Optimal thinking budget for this model in tokens (if applicable)
    pub optimal_thinking_budget: Option<u32>,

    /// System prompt prefix for this model (if any)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt_prefix: Option<String>,

    /// Real name of the model (if different from name)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_name: Option<String>,

    /// Model type (e.g., "embedding" for embedding models)
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub model_type: Option<String>,

    /// Short CLI identifier for model selection (e.g., "kimi-k2", "qwen-coder")
    pub model_id: &'static str,

    /// HuggingFace repository URL for automatic model downloads
    pub hf_repo_url: &'static str,

    /// Model quantization format (e.g., "Q4_0", "Q5_0", "F16")
    pub quantization: &'static str,

    /// Patch configuration for API requests
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch: Option<serde_json::Value>}

impl CandleModelInfo {

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

    /// Check if the model supports streaming
    #[inline]
    pub fn has_streaming(&self) -> bool {
        self.supports_streaming
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

    /// Get the model's short CLI identifier
    #[inline]
    pub fn model_id(&self) -> &'static str {
        self.model_id
    }

    /// Get the HuggingFace repository URL for automatic downloads
    #[inline]
    pub fn hf_repo_url(&self) -> &'static str {
        self.hf_repo_url
    }

    /// Get the model's quantization format
    #[inline]
    pub fn quantization(&self) -> &'static str {
        self.quantization
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

    /// Convert to CandleModelCapabilities for filtering and querying
    ///
    /// This creates a CandleModelCapabilities struct from this CandleModelInfo instance.
    /// CandleModelInfo remains the single source of truth from YAML deserialization.
    pub fn to_capabilities(&self) -> crate::domain::model::capabilities::CandleModelCapabilities {
        crate::domain::model::capabilities::CandleModelCapabilities {
            supports_vision: self.supports_vision,
            supports_function_calling: self.supports_function_calling,
            supports_streaming: self.supports_streaming,
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
                .map_or(false, |tokens| tokens.get() > 32000),
            is_low_latency: false,        // Not in ModelInfo yet
            is_high_throughput: false,    // Not in ModelInfo yet
            supports_quantization: false, // Not in ModelInfo yet
            supports_distillation: false, // Not in ModelInfo yet
            supports_pruning: false,      // Not in ModelInfo yet
        }
    }

    /// Validate the model configuration
    pub fn validate(&self) -> CandleResult<()> {
        if self.provider_name.is_empty() {
            return Err(CandleModelError::InvalidConfiguration(
                "provider_name cannot be empty".into(),
            ));
        }

        if self.name.is_empty() {
            return Err(CandleModelError::InvalidConfiguration(
                "name cannot be empty".into(),
            ));
        }

        if let Some(max_input) = self.max_input_tokens {
            if max_input.get() == 0 {
                return Err(CandleModelError::InvalidConfiguration(
                    "max_input_tokens cannot be zero".into(),
                ));
            }
        }

        if let Some(max_output) = self.max_output_tokens {
            if max_output.get() == 0 {
                return Err(CandleModelError::InvalidConfiguration(
                    "max_output_tokens cannot be zero".into(),
                ));
            }
        }

        if self.supports_thinking && self.optimal_thinking_budget.is_none() {
            return Err(CandleModelError::InvalidConfiguration(
                "optimal_thinking_budget must be set when supports_thinking is true".into(),
            ));
        }

        Ok(())
    }
}

impl Hash for CandleModelInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.provider_name.hash(state);
        self.name.hash(state);
    }
}

/// A collection of model information for a specific provider
#[derive(Debug, Clone, Default)]
pub struct CandleProviderModels {
    provider_name: &'static str,
    models: Vec<CandleModelInfo>}

impl CandleProviderModels {
    /// Create a new provider model collection
    #[inline]
    pub fn new(provider_name: &'static str) -> Self {
        Self {
            provider_name,
            models: Vec::new()}
    }

    /// Add a model to the collection
    pub fn add_model(&mut self, model: CandleModelInfo) -> CandleResult<()> {
        if model.provider_name != self.provider_name {
            return Err(CandleModelError::InvalidConfiguration(
                "model provider does not match collection provider".into(),
            ));
        }

        if self.models.iter().any(|m| m.name == model.name) {
            return Err(CandleModelError::ModelAlreadyExists {
                provider: self.provider_name.into(),
                name: model.name.into()});
        }

        self.models.push(model);
        Ok(())
    }

    /// Get a model by name
    #[inline]
    pub fn get(&self, name: &str) -> Option<&CandleModelInfo> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Get all models
    #[inline]
    pub fn all(&self) -> &[CandleModelInfo] {
        &self.models
    }

    /// Get the provider name
    #[inline]
    pub fn provider_name(&self) -> &'static str {
        self.provider_name
    }
}


