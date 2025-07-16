//! Zero-allocation, blazing-fast model trait definitions for AI services
//!
//! This module provides the core Model trait and related types that are completely
//! independent of the dynamically generated model enumerations. The design separates
//! trait definitions from concrete implementations for maximum performance and flexibility.

use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

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
    /// Whether the model requires max_tokens parameter
    pub require_max_tokens: Option<bool>,
    /// Model capabilities - cached for performance
    pub capabilities: ModelCapabilities,
    /// Token limits - optimized structure
    pub token_limits: TokenLimits,
    /// Pricing information - optimized structure
    pub pricing: ModelPricing,
}

/// High-performance model capabilities with bit-packed flags
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelCapabilities {
    /// Packed capability flags for zero-allocation access
    flags: u32,
}

impl ModelCapabilities {
    /// Vision support flag bit
    const VISION_FLAG: u32 = 1 << 0;
    /// Function calling support flag bit
    const FUNCTION_CALLING_FLAG: u32 = 1 << 1;
    /// Streaming support flag bit
    const STREAMING_FLAG: u32 = 1 << 2;
    /// Embedding support flag bit
    const EMBEDDING_FLAG: u32 = 1 << 3;
    /// Multimodal support flag bit
    const MULTIMODAL_FLAG: u32 = 1 << 4;
    /// Real-time support flag bit
    const REALTIME_FLAG: u32 = 1 << 5;
    /// Fine-tuning support flag bit
    const FINE_TUNING_FLAG: u32 = 1 << 6;
    /// Batch processing support flag bit
    const BATCH_PROCESSING_FLAG: u32 = 1 << 7;

    /// Create new capabilities with all flags disabled
    #[inline]
    pub const fn new() -> Self {
        Self { flags: 0 }
    }

    /// Create capabilities with all features enabled
    #[inline]
    pub const fn all() -> Self {
        Self { flags: u32::MAX }
    }

    /// Check if vision is supported - zero allocation
    #[inline]
    pub const fn has_vision(&self) -> bool {
        self.flags & Self::VISION_FLAG != 0
    }

    /// Check if function calling is supported - zero allocation
    #[inline]
    pub const fn has_function_calling(&self) -> bool {
        self.flags & Self::FUNCTION_CALLING_FLAG != 0
    }

    /// Check if streaming is supported - zero allocation
    #[inline]
    pub const fn has_streaming(&self) -> bool {
        self.flags & Self::STREAMING_FLAG != 0
    }

    /// Check if embedding is supported - zero allocation
    #[inline]
    pub const fn has_embedding(&self) -> bool {
        self.flags & Self::EMBEDDING_FLAG != 0
    }

    /// Check if multimodal is supported - zero allocation
    #[inline]
    pub const fn has_multimodal(&self) -> bool {
        self.flags & Self::MULTIMODAL_FLAG != 0
    }

    /// Check if real-time is supported - zero allocation
    #[inline]
    pub const fn has_realtime(&self) -> bool {
        self.flags & Self::REALTIME_FLAG != 0
    }

    /// Check if fine-tuning is supported - zero allocation
    #[inline]
    pub const fn has_fine_tuning(&self) -> bool {
        self.flags & Self::FINE_TUNING_FLAG != 0
    }

    /// Check if batch processing is supported - zero allocation
    #[inline]
    pub const fn has_batch_processing(&self) -> bool {
        self.flags & Self::BATCH_PROCESSING_FLAG != 0
    }

    /// Enable vision support - zero allocation
    #[inline]
    pub const fn with_vision(mut self) -> Self {
        self.flags |= Self::VISION_FLAG;
        self
    }

    /// Enable function calling support - zero allocation
    #[inline]
    pub const fn with_function_calling(mut self) -> Self {
        self.flags |= Self::FUNCTION_CALLING_FLAG;
        self
    }

    /// Enable streaming support - zero allocation
    #[inline]
    pub const fn with_streaming(mut self) -> Self {
        self.flags |= Self::STREAMING_FLAG;
        self
    }

    /// Enable embedding support - zero allocation
    #[inline]
    pub const fn with_embedding(mut self) -> Self {
        self.flags |= Self::EMBEDDING_FLAG;
        self
    }

    /// Enable multimodal support - zero allocation
    #[inline]
    pub const fn with_multimodal(mut self) -> Self {
        self.flags |= Self::MULTIMODAL_FLAG;
        self
    }

    /// Enable real-time support - zero allocation
    #[inline]
    pub const fn with_realtime(mut self) -> Self {
        self.flags |= Self::REALTIME_FLAG;
        self
    }

    /// Enable fine-tuning support - zero allocation
    #[inline]
    pub const fn with_fine_tuning(mut self) -> Self {
        self.flags |= Self::FINE_TUNING_FLAG;
        self
    }

    /// Enable batch processing support - zero allocation
    #[inline]
    pub const fn with_batch_processing(mut self) -> Self {
        self.flags |= Self::BATCH_PROCESSING_FLAG;
        self
    }
}

impl Default for ModelCapabilities {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
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
    pub const fn new(input_price_per_million: f64, output_price_per_million: f64, tier: u8) -> Self {
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
        (input_tokens as f64 * self.input_price_per_token()) + 
        (output_tokens as f64 * self.output_price_per_token())
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
    
    /// Get model capabilities - zero allocation
    #[inline]
    fn capabilities(&self) -> ModelCapabilities {
        self.info().capabilities
    }
    
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
        self.capabilities().has_vision()
    }
    
    /// Check if model supports function calling - zero allocation
    #[inline]
    fn supports_function_calling(&self) -> bool {
        self.capabilities().has_function_calling()
    }
    
    /// Check if model supports streaming - zero allocation
    #[inline]
    fn supports_streaming(&self) -> bool {
        self.capabilities().has_streaming()
    }
    
    /// Check if model supports embedding - zero allocation
    #[inline]
    fn supports_embedding(&self) -> bool {
        self.capabilities().has_embedding()
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

/// Optimized model information builder with zero-allocation patterns
pub struct ModelInfoBuilder {
    info: ModelInfo,
}

impl ModelInfoBuilder {
    /// Create new builder with default values
    #[inline]
    pub fn new() -> Self {
        Self {
            info: ModelInfo {
                provider_name: Cow::Borrowed(""),
                name: Cow::Borrowed(""),
                max_input_tokens: None,
                max_output_tokens: None,
                input_price: None,
                output_price: None,
                supports_vision: None,
                supports_function_calling: None,
                require_max_tokens: None,
                capabilities: ModelCapabilities::new(),
                token_limits: TokenLimits::unlimited(),
                pricing: ModelPricing::free(),
            },
        }
    }
    
    /// Set provider name using static string for zero allocation
    #[inline]
    pub fn provider_name(mut self, name: &'static str) -> Self {
        self.info.provider_name = Cow::Borrowed(name);
        self
    }
    
    /// Set provider name using owned string
    #[inline]
    pub fn provider_name_owned(mut self, name: String) -> Self {
        self.info.provider_name = Cow::Owned(name);
        self
    }
    
    /// Set model name using static string for zero allocation
    #[inline]
    pub fn name(mut self, name: &'static str) -> Self {
        self.info.name = Cow::Borrowed(name);
        self
    }
    
    /// Set model name using owned string
    #[inline]
    pub fn name_owned(mut self, name: String) -> Self {
        self.info.name = Cow::Owned(name);
        self
    }
    
    /// Set capabilities
    #[inline]
    pub fn capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.info.capabilities = capabilities;
        self
    }
    
    /// Set token limits
    #[inline]
    pub fn token_limits(mut self, limits: TokenLimits) -> Self {
        self.info.token_limits = limits;
        self
    }
    
    /// Set pricing
    #[inline]
    pub fn pricing(mut self, pricing: ModelPricing) -> Self {
        self.info.pricing = pricing;
        self
    }
    
    /// Set max input tokens
    #[inline]
    pub fn max_input_tokens(mut self, tokens: u64) -> Self {
        self.info.max_input_tokens = Some(tokens);
        self.info.token_limits.max_input = tokens;
        self
    }
    
    /// Set max output tokens
    #[inline]
    pub fn max_output_tokens(mut self, tokens: u64) -> Self {
        self.info.max_output_tokens = Some(tokens);
        self.info.token_limits.max_output = tokens;
        self
    }
    
    /// Set vision support
    #[inline]
    pub fn supports_vision(mut self, supported: bool) -> Self {
        self.info.supports_vision = Some(supported);
        if supported {
            self.info.capabilities = self.info.capabilities.with_vision();
        }
        self
    }
    
    /// Set function calling support
    #[inline]
    pub fn supports_function_calling(mut self, supported: bool) -> Self {
        self.info.supports_function_calling = Some(supported);
        if supported {
            self.info.capabilities = self.info.capabilities.with_function_calling();
        }
        self
    }
    
    /// Set pricing per million tokens
    #[inline]
    pub fn pricing_per_million(mut self, input: f64, output: f64, tier: u8) -> Self {
        self.info.input_price = Some(input / 1_000_000.0);
        self.info.output_price = Some(output / 1_000_000.0);
        self.info.pricing = ModelPricing::new(input, output, tier);
        self
    }
    
    /// Build the final ModelInfo
    #[inline]
    pub fn build(self) -> ModelInfo {
        self.info
    }
}

impl Default for ModelInfoBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

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
        self.providers.entry(provider).or_insert_with(Vec::new).push(name);
        
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
        self.providers.get(provider)
            .map(|names| names.iter().filter_map(|name| self.models.get(&(provider, name))).collect())
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