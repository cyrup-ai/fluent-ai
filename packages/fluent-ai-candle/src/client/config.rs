//! Client configuration types and implementations
//! 
//! This module contains all configuration-related types for the CandleCompletionClient,
//! including device selection, model configuration, and quantization options.

use crate::generator::GenerationConfig;
use crate::hub::HubConfig;
use crate::kv_cache::KVCacheConfig;
use crate::sampling::{Sampling, SamplingConfig};
use crate::streaming::StreamingConfig;
use crate::tokenizer::TokenizerConfig;
use crate::var_builder::VarBuilderConfig;

/// Configuration for CandleCompletionClient with sophisticated features
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CandleClientConfig {
    /// Model path or identifier
    pub model_path: String,
    /// Tokenizer path or identifier
    pub tokenizer_path: Option<String>,
    /// Device to use for computation
    pub device_type: DeviceType,
    /// Model configuration
    pub model_config: ModelConfig,
    /// Tokenizer configuration
    pub tokenizer_config: TokenizerConfig,
    /// Generation configuration
    pub generation_config: GenerationConfig,
    /// Sampling configuration for sophisticated sampling
    pub sampling_config: Sampling,
    /// Streaming configuration for real-time output
    pub streaming_config: StreamingConfig,
    /// VarBuilder configuration for weight loading
    pub var_builder_config: VarBuilderConfig,
    /// KV cache configuration for efficient generation
    pub kv_cache_config: KVCacheConfig,
    /// Hub configuration for model downloading
    pub hub_config: HubConfig,
    /// Enable model quantization
    pub enable_quantization: bool,
    /// Quantization type
    pub quantization_type: QuantizationType,
    /// Maximum concurrent requests
    pub max_concurrent_requests: u32,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Enable caching
    pub enable_caching: bool,
    /// Cache size in MB
    pub cache_size_mb: u32,
    /// Enable sophisticated sampling
    pub enable_sophisticated_sampling: bool,
    /// Enable real-time streaming
    pub enable_streaming_optimization: bool,
    /// Enable KV caching
    pub enable_kv_cache: bool,
    /// Enable Hub integration
    pub enable_hub_integration: bool}

impl Default for CandleClientConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: None,
            device_type: DeviceType::Auto,
            model_config: ModelConfig::default(),
            tokenizer_config: TokenizerConfig::default(),
            generation_config: GenerationConfig::default(),
            sampling_config: SamplingConfig::default().build_sampling(),
            streaming_config: StreamingConfig::default(),
            var_builder_config: VarBuilderConfig::default(),
            kv_cache_config: KVCacheConfig::default(),
            hub_config: HubConfig::default(),
            enable_quantization: false,
            quantization_type: QuantizationType::Q8_0,
            max_concurrent_requests: 4,
            request_timeout_seconds: 300,
            enable_caching: true,
            cache_size_mb: 512,
            enable_sophisticated_sampling: true,
            enable_streaming_optimization: true,
            enable_kv_cache: true,
            enable_hub_integration: true}
    }
}

/// Device type selection
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// Automatically select best available device
    Auto = 0,
    /// Use CPU only
    Cpu = 1,
    /// Use CUDA GPU
    Cuda = 2,
    /// Use Metal GPU (macOS)
    Metal = 3}

/// Model configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture
    pub architecture: ModelArchitecture,
    /// Context length
    pub context_length: u32,
    /// Hidden size
    pub hidden_size: u32,
    /// Number of attention heads
    pub num_attention_heads: u32,
    /// Number of layers
    pub num_layers: u32,
    /// Vocabulary size
    pub vocab_size: u32,
    /// Use flash attention
    pub use_flash_attention: bool,
    /// RoPE scaling factor
    pub rope_scaling: f32}

impl Default for ModelConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::Llama,
            context_length: 2048,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_layers: 32,
            vocab_size: 32000,
            use_flash_attention: true,
            rope_scaling: 1.0}
    }
}

/// Supported model architectures
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// LLaMA family models
    Llama = 0,
    /// Mistral family models
    Mistral = 1,
    /// Mixtral MoE models
    Mixtral = 2,
    /// Gemma models
    Gemma = 3,
    /// Phi models
    Phi = 4,
    /// Custom architecture
    Custom = 255}

/// Quantization types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// 4-bit quantization (variant 0)
    Q4_0 = 0,
    /// 4-bit quantization (variant 1)
    Q4_1 = 1,
    /// 8-bit quantization
    Q8_0 = 2,
    /// No quantization
    None = 255}

/// Maximum messages per completion request (compile-time bounded)
pub const MAX_MESSAGES: usize = 128;
/// Maximum tools per request (compile-time bounded)  
pub const MAX_TOOLS: usize = 32;
/// Maximum documents per request (compile-time bounded)
pub const MAX_DOCUMENTS: usize = 64;