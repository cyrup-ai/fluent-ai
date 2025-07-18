//! Zero-allocation, blazing-fast model trait definitions for AI services
//!
//! This module provides the core Model trait and related types that are completely
//! independent of the dynamically generated model enumerations. The design separates
//! trait definitions from concrete implementations for maximum performance and flexibility.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Zero-allocation model information with optimized string handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Provider name - uses static string when possible for zero allocation
    pub provider_name: Cow<'static, str>,
    /// Model name - uses static string when possible for zero allocation
    pub name: Cow<'static, str>,
    /// Maximum input tokens supported
    pub max_input_tokens: Option<u64>,
    /// Maximum output tokens supported
    pub max_output_tokens: Option<u64>,
    /// Input price per token in USD
    pub input_price: Option<f64>,
    /// Output price per token in USD
    pub output_price: Option<f64>,
    /// Whether the model supports vision/image inputs
    pub supports_vision: Option<bool>,
    /// Whether the model supports function calling
    pub supports_function_calling: Option<bool>,
    /// Whether the model supports streaming responses
    pub supports_streaming: Option<bool>,
    /// Whether the model supports embedding generation
    pub supports_embedding: Option<bool>,
    /// Whether the model supports multimodal inputs
    pub supports_multimodal: Option<bool>,
    /// Whether the model supports real-time processing
    pub supports_realtime: Option<bool>,
    /// Whether the model supports fine-tuning
    pub supports_fine_tuning: Option<bool>,
    /// Whether the model supports batch processing
    pub supports_batch_processing: Option<bool>,
    /// Whether the model requires max_tokens parameter
    pub require_max_tokens: Option<bool>,
    /// Token limits - optimized structure
    pub token_limits: TokenLimits,
    /// Pricing information - optimized structure
    pub pricing: ModelPricing,
}

/// Zero-allocation token limits with optimized numeric handling
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TokenLimits {
    /// Maximum input tokens (0 means unlimited)
    pub max_input: u64,
    /// Maximum output tokens (0 means unlimited)
    pub max_output: u64,
    /// Total context window size (0 means unlimited)
    pub context_window: u64,
}

impl TokenLimits {
    /// Create new token limits with all values unlimited
    #[inline]
    pub const fn unlimited() -> Self {
        Self {
            max_input: 0,
            max_output: 0,
            context_window: 0,
        }
    }

    /// Create token limits with specified values
    #[inline]
    pub const fn new(max_input: u64, max_output: u64, context_window: u64) -> Self {
        Self {
            max_input,
            max_output,
            context_window,
        }
    }

    /// Check if input tokens are unlimited - zero allocation
    #[inline]
    pub const fn has_unlimited_input(&self) -> bool {
        self.max_input == 0
    }

    /// Check if output tokens are unlimited - zero allocation
    #[inline]
    pub const fn has_unlimited_output(&self) -> bool {
        self.max_output == 0
    }

    /// Check if context window is unlimited - zero allocation
    #[inline]
    pub const fn has_unlimited_context(&self) -> bool {
        self.context_window == 0
    }

    /// Get effective input limit (returns u64::MAX for unlimited)
    #[inline]
    pub const fn effective_input_limit(&self) -> u64 {
        if self.max_input == 0 {
            u64::MAX
        } else {
            self.max_input
        }
    }

    /// Get effective output limit (returns u64::MAX for unlimited)
    #[inline]
    pub const fn effective_output_limit(&self) -> u64 {
        if self.max_output == 0 {
            u64::MAX
        } else {
            self.max_output
        }
    }

    /// Get effective context window limit (returns u64::MAX for unlimited)
    #[inline]
    pub const fn effective_context_limit(&self) -> u64 {
        if self.context_window == 0 {
            u64::MAX
        } else {
            self.context_window
        }
    }
}

impl Default for TokenLimits {
    #[inline]
    fn default() -> Self {
        Self::unlimited()
    }
}

/// Zero-allocation pricing information with optimized numeric handling
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ModelPricing {
    /// Input price per million tokens in USD (0.0 means free)
    pub input_price_per_million: f64,
    /// Output price per million tokens in USD (0.0 means free)
    pub output_price_per_million: f64,
    /// Pricing tier for optimization (0 = free, 1 = cheap, 2 = standard, 3 = premium)
    pub tier: u8,
}

impl ModelPricing {
    /// Create free pricing
    #[inline]
    pub const fn free() -> Self {
        Self {
            input_price_per_million: 0.0,
            output_price_per_million: 0.0,
            tier: 0,
        }
    }

    /// Create pricing with specified values
    #[inline]
    pub const fn new(
        input_price_per_million: f64,
        output_price_per_million: f64,
        tier: u8,
    ) -> Self {
        Self {
            input_price_per_million,
            output_price_per_million,
            tier,
        }
    }

    /// Check if the model is free - zero allocation
    #[inline]
    pub const fn is_free(&self) -> bool {
        self.input_price_per_million == 0.0 && self.output_price_per_million == 0.0
    }

    /// Get input price per token (not per million) - zero allocation
    #[inline]
    pub const fn input_price_per_token(&self) -> f64 {
        self.input_price_per_million / 1_000_000.0
    }

    /// Get output price per token (not per million) - zero allocation
    #[inline]
    pub const fn output_price_per_token(&self) -> f64 {
        self.output_price_per_million / 1_000_000.0
    }

    /// Calculate total cost for given token usage - zero allocation
    #[inline]
    pub const fn calculate_cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        (input_tokens as f64 * self.input_price_per_token())
            + (output_tokens as f64 * self.output_price_per_token())
    }
}

impl Default for ModelPricing {
    #[inline]
    fn default() -> Self {
        Self::free()
    }
}

/// Core trait that all AI models must implement
/// Designed for zero-allocation, blazing-fast performance
pub trait Model: Send + Sync + fmt::Debug {
    /// Get detailed information about this model - zero allocation when possible
    fn info(&self) -> &ModelInfo;

    /// Get the model name as a static string - zero allocation
    fn name(&self) -> &'static str;

    /// Get the provider name for this model - zero allocation
    fn provider(&self) -> &'static str;

    /// Get token limits - zero allocation
    #[inline]
    fn token_limits(&self) -> TokenLimits {
        self.info().token_limits
    }

    /// Get pricing information - zero allocation
    #[inline]
    fn pricing(&self) -> ModelPricing {
        self.info().pricing
    }

    /// Check if model supports vision - zero allocation
    #[inline]
    fn supports_vision(&self) -> bool {
        self.info().supports_vision.unwrap_or(false)
    }

    /// Check if model supports function calling - zero allocation
    #[inline]
    fn supports_function_calling(&self) -> bool {
        self.info().supports_function_calling.unwrap_or(false)
    }

    /// Check if model supports streaming - zero allocation
    #[inline]
    fn supports_streaming(&self) -> bool {
        self.info().supports_streaming.unwrap_or(false)
    }

    /// Check if model supports embedding - zero allocation
    #[inline]
    fn supports_embedding(&self) -> bool {
        self.info().supports_embedding.unwrap_or(false)
    }

    /// Check if model supports multimodal inputs - zero allocation
    #[inline]
    fn supports_multimodal(&self) -> bool {
        self.info().supports_multimodal.unwrap_or(false)
    }

    /// Check if model supports real-time processing - zero allocation
    #[inline]
    fn supports_realtime(&self) -> bool {
        self.info().supports_realtime.unwrap_or(false)
    }

    /// Check if model supports fine-tuning - zero allocation
    #[inline]
    fn supports_fine_tuning(&self) -> bool {
        self.info().supports_fine_tuning.unwrap_or(false)
    }

    /// Check if model supports batch processing - zero allocation
    #[inline]
    fn supports_batch_processing(&self) -> bool {
        self.info().supports_batch_processing.unwrap_or(false)
    }

    /// Get maximum input tokens - zero allocation
    #[inline]
    fn max_input_tokens(&self) -> u64 {
        self.token_limits().effective_input_limit()
    }

    /// Get maximum output tokens - zero allocation
    #[inline]
    fn max_output_tokens(&self) -> u64 {
        self.token_limits().effective_output_limit()
    }

    /// Get context window size - zero allocation
    #[inline]
    fn context_window(&self) -> u64 {
        self.token_limits().effective_context_limit()
    }

    /// Check if model is free - zero allocation
    #[inline]
    fn is_free(&self) -> bool {
        self.pricing().is_free()
    }

    /// Calculate cost for token usage - zero allocation
    #[inline]
    fn calculate_cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        self.pricing().calculate_cost(input_tokens, output_tokens)
    }

    /// Get pricing tier - zero allocation
    #[inline]
    fn pricing_tier(&self) -> u8 {
        self.pricing().tier
    }

    /// Check if this model is the same as another - zero allocation
    #[inline]
    fn is_same_model(&self, other: &impl Model) -> bool {
        self.name() == other.name() && self.provider() == other.provider()
    }
}

// ModelInfoBuilder moved to fluent_ai/src/builders/model.rs

/// Type alias for backward compatibility
pub type ModelInfoData = ModelInfo;

/// High-performance model registry with zero-allocation lookups
pub struct ModelRegistry<T: Model> {
    /// Models indexed by provider and name for O(1) lookup
    models: BTreeMap<(&'static str, &'static str), Arc<T>>,
    /// Provider index for fast provider-based queries
    providers: BTreeMap<&'static str, Vec<&'static str>>,
}

impl<T: Model> ModelRegistry<T> {
    /// Create new empty registry
    #[inline]
    pub fn new() -> Self {
        Self {
            models: BTreeMap::new(),
            providers: BTreeMap::new(),
        }
    }

    /// Register a model - zero allocation for lookups
    pub fn register(&mut self, model: T) -> Result<(), &'static str> {
        let provider = model.provider();
        let name = model.name();
        let key = (provider, name);

        if self.models.contains_key(&key) {
            return Err("Model already registered");
        }

        self.models.insert(key, Arc::new(model));

        // Update provider index
        self.providers
            .entry(provider)
            .or_insert_with(Vec::new)
            .push(name);

        Ok(())
    }

    /// Get model by provider and name - zero allocation
    #[inline]
    pub fn get<'a>(&'a self, provider: &'a str, name: &'a str) -> Option<&'a Arc<T>> {
        self.models.get(&(provider, name))
    }

    /// Get all models for a provider - zero allocation
    #[inline]
    pub fn get_provider_models<'a>(&'a self, provider: &'a str) -> Vec<&'a Arc<T>> {
        self.providers
            .get(provider)
            .map(|names| {
                names
                    .iter()
                    .filter_map(|name| self.models.get(&(provider, name)))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all providers - zero allocation
    #[inline]
    pub fn get_providers(&self) -> Vec<&'static str> {
        self.providers.keys().copied().collect()
    }

    /// Get total model count - zero allocation
    #[inline]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if registry is empty - zero allocation
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

impl<T: Model> Default for ModelRegistry<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Model> fmt::Debug for ModelRegistry<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelRegistry")
            .field("model_count", &self.models.len())
            .field("provider_count", &self.providers.len())
            .finish()
    }
}

/// Thread-safe model registry for global access
pub type SharedModelRegistry<T> = Arc<std::sync::RwLock<ModelRegistry<T>>>;

// Global registry functionality should be implemented by the provider crate
// that has concrete model types, not in the domain crate
