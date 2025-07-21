//! Model configuration types and enumerations
//!
//! This module provides core type definitions for:
//! - Model architecture types 
//! - Quantization options
//! - Model configuration structures
//! - Default values for different architectures

/// Supported model types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// LLaMA family models (LLaMA, LLaMA2, Code Llama)
    Llama = 0,
    /// Mistral family models (Mistral 7B, Mixtral)
    Mistral = 1,
    /// Gemma family models
    Gemma = 2,
    /// Phi family models
    Phi = 3,
    /// Qwen family models
    Qwen = 4,
    /// Custom/unknown model
    Custom = 255,
}

impl ModelType {
    /// Get model type from string
    #[inline(always)]
    pub fn from_str(s: &str) -> Self {
        let s = s.to_lowercase();
        if s.contains("llama") {
            Self::Llama
        } else if s.contains("mistral") || s.contains("mixtral") {
            Self::Mistral
        } else if s.contains("gemma") {
            Self::Gemma
        } else if s.contains("phi") {
            Self::Phi
        } else if s.contains("qwen") {
            Self::Qwen
        } else {
            Self::Custom
        }
    }

    /// Get default context length for model type
    #[inline(always)]
    pub fn default_context_length(&self) -> u32 {
        match self {
            Self::Llama => 4096,
            Self::Mistral => 8192,
            Self::Gemma => 8192,
            Self::Phi => 2048,
            Self::Qwen => 8192,
            Self::Custom => 2048,
        }
    }
}

/// Quantization types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// No quantization
    None = 0,
    /// 4-bit quantization (Q4_0)
    Q4_0 = 1,
    /// 4-bit quantization (Q4_1)
    Q4_1 = 2,
    /// 8-bit quantization
    Q8_0 = 3,
}

impl Default for QuantizationType {
    fn default() -> Self {
        Self::None
    }
}

impl QuantizationType {
    /// Get memory reduction factor for quantization type
    #[inline(always)]
    pub fn memory_factor(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Q4_0 | Self::Q4_1 => 0.25, // ~4x reduction
            Self::Q8_0 => 0.5, // ~2x reduction
        }
    }

    /// Get performance impact factor (higher = slower)
    #[inline(always)]
    pub fn performance_impact(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Q4_0 | Self::Q4_1 => 1.2, // Slight overhead
            Self::Q8_0 => 1.1, // Minimal overhead
        }
    }
}

/// Model configuration
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
            model_type: ModelType::Llama,
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
    /// Create configuration for specific model type
    pub fn for_model_type(model_type: ModelType) -> Self {
        let mut config = Self::default();
        config.model_type = model_type;
        config.context_length = model_type.default_context_length();
        
        // Adjust defaults based on model type
        match model_type {
            ModelType::Mistral => {
                config.vocab_size = 32000;
                config.hidden_size = 4096;
                config.num_heads = 32;
            }
            ModelType::Gemma => {
                config.vocab_size = 256000;
                config.hidden_size = 3072;
                config.num_heads = 24;
            }
            ModelType::Phi => {
                config.vocab_size = 51200;
                config.hidden_size = 2560;
                config.num_heads = 32;
                config.num_layers = 32;
            }
            ModelType::Qwen => {
                config.vocab_size = 151936;
                config.hidden_size = 4096;
                config.num_heads = 32;
                config.rope_theta = 1000000.0;
            }
            ModelType::Llama | ModelType::Custom => {
                // Use defaults
            }
        }
        
        config
    }

    /// Estimate memory requirements in bytes
    pub fn estimate_memory_bytes(&self) -> u64 {
        let param_count = self.estimate_parameter_count();
        let bytes_per_param = match self.quantization.memory_factor() {
            f if f <= 0.25 => 1, // Q4
            f if f <= 0.5 => 1,  // Q8 
            _ => 4, // F32
        };
        
        // Model weights + KV cache + intermediate activations
        let model_size = param_count * bytes_per_param as u64;
        let kv_cache_size = (self.num_layers as u64 * self.hidden_size as u64 * self.context_length as u64 * 2) / 1024; // Rough estimate
        let activation_size = self.hidden_size as u64 * 1024; // Working memory
        
        model_size + kv_cache_size + activation_size
    }

    /// Estimate parameter count
    pub fn estimate_parameter_count(&self) -> u64 {
        let embedding_params = self.vocab_size as u64 * self.hidden_size as u64;
        let layer_params = self.num_layers as u64 * self.hidden_size as u64 * self.hidden_size as u64 * 4; // Rough estimate
        let output_params = self.hidden_size as u64 * self.vocab_size as u64;
        
        embedding_params + layer_params + output_params
    }
}