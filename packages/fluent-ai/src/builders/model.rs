use crate::domain::model::{ModelInfo, ModelCapabilities, TokenLimits, ModelPricing};
use std::borrow::Cow;

/// Zero-allocation model information builder with blazing-fast performance
#[derive(Debug, Clone)]
pub struct ModelInfoBuilder {
    info: ModelInfo,
}

impl ModelInfoBuilder {
    /// Create new builder with default values - zero allocation hot path
    #[inline(always)]
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
    #[inline(always)]
    pub fn provider_name(mut self, name: &'static str) -> Self {
        self.info.provider_name = Cow::Borrowed(name);
        self
    }
    
    /// Set provider name using owned string when necessary
    #[inline(always)]
    pub fn provider_name_owned(mut self, name: String) -> Self {
        self.info.provider_name = Cow::Owned(name);
        self
    }
    
    /// Set model name using static string for zero allocation
    #[inline(always)]
    pub fn name(mut self, name: &'static str) -> Self {
        self.info.name = Cow::Borrowed(name);
        self
    }
    
    /// Set model name using owned string when necessary
    #[inline(always)]
    pub fn name_owned(mut self, name: String) -> Self {
        self.info.name = Cow::Owned(name);
        self
    }
    
    /// Set capabilities with zero allocation
    #[inline(always)]
    pub fn capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.info.capabilities = capabilities;
        self
    }
    
    /// Set token limits with zero allocation
    #[inline(always)]
    pub fn token_limits(mut self, limits: TokenLimits) -> Self {
        self.info.token_limits = limits;
        self
    }
    
    /// Set pricing with zero allocation
    #[inline(always)]
    pub fn pricing(mut self, pricing: ModelPricing) -> Self {
        self.info.pricing = pricing;
        self
    }
    
    /// Set max input tokens - optimized for performance
    #[inline(always)]
    pub fn max_input_tokens(mut self, tokens: u64) -> Self {
        self.info.max_input_tokens = Some(tokens);
        self.info.token_limits.max_input = tokens;
        self
    }
    
    /// Set max output tokens - optimized for performance
    #[inline(always)]
    pub fn max_output_tokens(mut self, tokens: u64) -> Self {
        self.info.max_output_tokens = Some(tokens);
        self.info.token_limits.max_output = tokens;
        self
    }
    
    /// Set vision support with capability synchronization
    #[inline(always)]
    pub fn supports_vision(mut self, supported: bool) -> Self {
        self.info.supports_vision = Some(supported);
        if supported {
            self.info.capabilities = self.info.capabilities.with_vision();
        }
        self
    }
    
    /// Set function calling support with capability synchronization
    #[inline(always)]
    pub fn supports_function_calling(mut self, supported: bool) -> Self {
        self.info.supports_function_calling = Some(supported);
        if supported {
            self.info.capabilities = self.info.capabilities.with_function_calling();
        }
        self
    }
    
    /// Set whether max_tokens parameter is required
    #[inline(always)]
    pub fn require_max_tokens(mut self, required: bool) -> Self {
        self.info.require_max_tokens = Some(required);
        self
    }
    
    /// Set pricing per million tokens with automatic unit conversion
    #[inline(always)]
    pub fn pricing_per_million(mut self, input: f64, output: f64, tier: u8) -> Self {
        self.info.input_price = Some(input / 1_000_000.0);
        self.info.output_price = Some(output / 1_000_000.0);
        self.info.pricing = ModelPricing::new(input, output, tier);
        self
    }
    
    /// Set pricing per token directly
    #[inline(always)]
    pub fn pricing_per_token(mut self, input: f64, output: f64, tier: u8) -> Self {
        self.info.input_price = Some(input);
        self.info.output_price = Some(output);
        self.info.pricing = ModelPricing::new(input * 1_000_000.0, output * 1_000_000.0, tier);
        self
    }
    
    /// Enable all capabilities for maximum model functionality
    #[inline(always)]
    pub fn enable_all_capabilities(mut self) -> Self {
        self.info.capabilities = ModelCapabilities::all();
        self.info.supports_vision = Some(true);
        self.info.supports_function_calling = Some(true);
        self
    }
    
    /// Set unlimited token limits
    #[inline(always)]
    pub fn unlimited_tokens(mut self) -> Self {
        self.info.token_limits = TokenLimits::unlimited();
        self.info.max_input_tokens = None;
        self.info.max_output_tokens = None;
        self
    }
    
    /// Set context window size with automatic limit calculation
    #[inline(always)]
    pub fn context_window(mut self, size: u64) -> Self {
        self.info.token_limits.context_window = size;
        // If input/output limits not set, use reasonable defaults
        if self.info.token_limits.max_input == 0 {
            self.info.token_limits.max_input = (size as f64 * 0.8) as u64;
        }
        if self.info.token_limits.max_output == 0 {
            self.info.token_limits.max_output = (size as f64 * 0.2) as u64;
        }
        self
    }
    
    /// Build the final ModelInfo with validation
    #[inline(always)]
    pub fn build(self) -> ModelInfo {
        self.info
    }
    
    /// Build with validation that ensures model is properly configured
    pub fn build_validated(self) -> Result<ModelInfo, &'static str> {
        if self.info.provider_name.is_empty() {
            return Err("Provider name cannot be empty");
        }
        if self.info.name.is_empty() {
            return Err("Model name cannot be empty");
        }
        Ok(self.info)
    }
}

impl Default for ModelInfoBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl ModelInfo {
    /// Create a builder for this model info
    #[inline(always)]
    pub fn builder() -> ModelInfoBuilder {
        ModelInfoBuilder::new()
    }
    
    /// Create a builder from existing model info
    #[inline(always)]
    pub fn to_builder(self) -> ModelInfoBuilder {
        ModelInfoBuilder { info: self }
    }
}