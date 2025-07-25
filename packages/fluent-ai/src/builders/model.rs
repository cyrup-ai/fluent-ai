//! Model builder implementations with zero-allocation, lock-free design
//!
//! All model construction logic and builder patterns.

use std::borrow::Cow;

use fluent_ai_domain::model::{ModelInfo, ModelPricing, TokenLimits};

/// Optimized model information builder with zero-allocation patterns
pub struct ModelInfoBuilder {
    info: ModelInfo}

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
                supports_streaming: None,
                supports_embedding: None,
                supports_multimodal: None,
                supports_realtime: None,
                supports_fine_tuning: None,
                supports_batch_processing: None,
                require_max_tokens: None,
                token_limits: TokenLimits::unlimited(),
                pricing: ModelPricing::free()}}
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
        self
    }

    /// Set function calling support
    #[inline]
    pub fn supports_function_calling(mut self, supported: bool) -> Self {
        self.info.supports_function_calling = Some(supported);
        self
    }

    /// Set streaming support
    #[inline]
    pub fn supports_streaming(mut self, supported: bool) -> Self {
        self.info.supports_streaming = Some(supported);
        self
    }

    /// Set embedding support
    #[inline]
    pub fn supports_embedding(mut self, supported: bool) -> Self {
        self.info.supports_embedding = Some(supported);
        self
    }

    /// Set multimodal support
    #[inline]
    pub fn supports_multimodal(mut self, supported: bool) -> Self {
        self.info.supports_multimodal = Some(supported);
        self
    }

    /// Set real-time support
    #[inline]
    pub fn supports_realtime(mut self, supported: bool) -> Self {
        self.info.supports_realtime = Some(supported);
        self
    }

    /// Set fine-tuning support
    #[inline]
    pub fn supports_fine_tuning(mut self, supported: bool) -> Self {
        self.info.supports_fine_tuning = Some(supported);
        self
    }

    /// Set batch processing support
    #[inline]
    pub fn supports_batch_processing(mut self, supported: bool) -> Self {
        self.info.supports_batch_processing = Some(supported);
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

/// Builder function for convenient model info construction
#[inline]
pub fn model_info() -> ModelInfoBuilder {
    ModelInfoBuilder::new()
}
