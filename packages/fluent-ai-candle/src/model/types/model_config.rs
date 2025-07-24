//! Model configuration structure with performance optimization
//!
//! Defines the core ModelConfig structure with zero-allocation access patterns
//! and blazing-fast configuration for different model architectures.

use super::{ModelType, QuantizationType};

/// Model configuration with zero-allocation design and inline optimizations
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model type
    pub model_type: ModelType,
    /// Context length
    pub context_length: u32,
    /// Vocabulary size
    pub vocab_size: u32,
    /// Number of layers
    pub num_layers: u32,
    /// Hidden dimension
    pub hidden_size: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// RoPE theta parameter
    pub rope_theta: f32,
    /// RoPE frequency base
    pub rope_freq_base: f32,
    /// Use flash attention
    pub use_flash_attn: bool,
    /// Quantization type
    pub quantization: QuantizationType,
}

impl Default for ModelConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            model_type: ModelType::KimiK2,
            context_length: 4096,
            vocab_size: 32000,
            num_layers: 32,
            hidden_size: 4096,
            num_heads: 32,
            rope_theta: 10000.0,
            rope_freq_base: 1.0,
            use_flash_attn: true,
            quantization: QuantizationType::None,
        }
    }
}

impl ModelConfig {
    /// Create configuration for specific model type with blazing-fast setup
    pub fn for_model_type(model_type: ModelType) -> Self {
        let mut config = Self::default();
        config.model_type = model_type;
        config.context_length = model_type.default_context_length();

        // Apply model-specific defaults with zero-allocation patterns
        match model_type {
            ModelType::KimiK2 => {
                config.vocab_size = 163840;
                config.hidden_size = 7168;
                config.num_heads = 64;
                config.num_layers = 61;
                config.rope_theta = 50000.0;
                config.context_length = 131072; // 131k context length
            }
        }

        config
    }
}
