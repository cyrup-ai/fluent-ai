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

/// Device type selection for model computation
///
/// Specifies which hardware device should be used for model inference.
/// The selection affects performance, memory usage, and availability.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// Automatically select the best available device
    ///
    /// The system will choose the optimal device based on availability:
    /// 1. CUDA GPU (if available and CUDA support is compiled)
    /// 2. Metal GPU (if on macOS with Metal support)
    /// 3. CPU (fallback)
    Auto = 0,
    
    /// Use CPU only for computation
    ///
    /// Forces CPU-only inference, which is:
    /// - Always available on all platforms
    /// - Lower memory requirements
    /// - Slower than GPU inference
    /// - Better for small models or when GPU memory is limited
    Cpu = 1,
    
    /// Use CUDA GPU for computation
    ///
    /// Requires:
    /// - NVIDIA GPU with CUDA support
    /// - CUDA runtime libraries installed
    /// - Sufficient GPU memory for the model
    /// 
    /// Benefits:
    /// - Significantly faster inference
    /// - Better parallelization for large models
    /// - Higher memory bandwidth
    Cuda = 2,
    
    /// Use Metal GPU for computation (macOS only)
    ///
    /// Requires:
    /// - macOS with Metal support
    /// - Apple Silicon or compatible Metal GPU
    /// - Sufficient GPU memory for the model
    ///
    /// Benefits:
    /// - Optimized for Apple hardware
    /// - Unified memory architecture on Apple Silicon
    /// - Lower power consumption than discrete GPUs
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

/// Supported model architectures for inference
///
/// Defines the neural network architecture type, which determines
/// the specific implementation, attention mechanisms, and layer structures
/// used during model inference.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// LLaMA (Large Language Model Meta AI) family models
    ///
    /// Includes:
    /// - LLaMA 1 and LLaMA 2 variants
    /// - Code Llama models
    /// - Vicuna, Alpaca, and other LLaMA derivatives
    ///
    /// Features:
    /// - RMSNorm normalization
    /// - SwiGLU activation function
    /// - Rotary Position Embedding (RoPE)
    /// - Group Query Attention (in LLaMA 2)
    Llama = 0,
    
    /// Mistral family models
    ///
    /// Includes:
    /// - Mistral 7B and variants
    /// - OpenOrca-Mistral models
    /// - Fine-tuned Mistral derivatives
    ///
    /// Features:
    /// - Sliding window attention
    /// - Group Query Attention
    /// - Efficient inference optimizations
    /// - Strong performance on reasoning tasks
    Mistral = 1,
    
    /// Mixtral Mixture of Experts (MoE) models
    ///
    /// Includes:
    /// - Mixtral 8x7B and 8x22B models
    /// - Sparse MoE architectures
    ///
    /// Features:
    /// - 8 expert networks per layer
    /// - Router network for expert selection
    /// - Conditional computation for efficiency
    /// - High parameter count with sparse activation
    Mixtral = 2,
    
    /// Gemma models from Google DeepMind
    ///
    /// Includes:
    /// - Gemma 2B and 7B models
    /// - Instruction-tuned variants
    ///
    /// Features:
    /// - RMSNorm normalization
    /// - Rotary Position Embedding
    /// - GeGLU activation function
    /// - Efficient small-scale architectures
    Gemma = 3,
    
    /// Phi models from Microsoft Research
    ///
    /// Includes:
    /// - Phi-1, Phi-1.5, Phi-2, and Phi-3 models
    /// - Code-focused variants
    ///
    /// Features:
    /// - Compact architectures (1.3B-14B parameters)
    /// - High performance despite smaller size
    /// - Optimized for reasoning and code generation
    /// - Efficient inference characteristics
    Phi = 4,
    
    /// Custom or experimental architecture
    ///
    /// Used for:
    /// - Research models with novel architectures
    /// - Custom implementations not covered by standard types
    /// - Experimental or proprietary model designs
    ///
    /// Note: Requires manual configuration of architecture-specific parameters
    Custom = 255}

/// Model weight quantization types for memory and performance optimization
///
/// Quantization reduces model memory usage and can improve inference speed
/// at the cost of some accuracy. Different quantization schemes offer
/// different trade-offs between compression ratio and quality preservation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// 4-bit quantization with block-wise scaling (GGML Q4_0 format)
    ///
    /// Characteristics:
    /// - 4 bits per weight (16 possible values)
    /// - Block size of 32 weights with shared scale factor
    /// - ~4x memory reduction compared to FP16
    /// - Fastest quantized inference
    /// - Moderate quality loss, suitable for most applications
    ///
    /// Memory usage: ~25% of original model size
    /// Quality: Good for most tasks, some degradation on complex reasoning
    Q4_0 = 0,
    
    /// 4-bit quantization with block-wise scaling and bias (GGML Q4_1 format)
    ///
    /// Characteristics:
    /// - 4 bits per weight with additional bias term
    /// - Block size of 32 weights with scale and bias
    /// - Better quality than Q4_0 at slight memory cost
    /// - Slightly slower inference than Q4_0
    /// - Improved handling of weight distributions
    ///
    /// Memory usage: ~28% of original model size
    /// Quality: Better than Q4_0, good balance of size and performance
    Q4_1 = 1,
    
    /// 8-bit quantization with block-wise scaling (GGML Q8_0 format)
    ///
    /// Characteristics:
    /// - 8 bits per weight (256 possible values)
    /// - Block size of 32 weights with shared scale factor
    /// - ~2x memory reduction compared to FP16
    /// - Slower than 4-bit but much better quality
    /// - Minimal quality loss for most models
    ///
    /// Memory usage: ~50% of original model size
    /// Quality: Excellent, nearly indistinguishable from full precision
    Q8_0 = 2,
    
    /// No quantization - full precision weights
    ///
    /// Characteristics:
    /// - Uses original model precision (typically FP16 or FP32)
    /// - Maximum quality and accuracy
    /// - Highest memory usage and potentially slower inference
    /// - Recommended for quality-critical applications
    ///
    /// Memory usage: 100% of original model size
    /// Quality: Perfect preservation of original model accuracy
    None = 255}

/// Maximum messages per completion request (compile-time bounded)
///
/// This constant defines the upper limit for the number of messages that can be
/// included in a single completion request. The limit enables stack-based allocation
/// and prevents excessive memory usage.
///
/// Value: 128 messages
/// 
/// Usage: Chat conversations, multi-turn dialogues, context messages
pub const MAX_MESSAGES: usize = 128;

/// Maximum tools per request (compile-time bounded)
///
/// This constant defines the upper limit for the number of function calling tools
/// that can be registered for a single completion request. This enables efficient
/// tool management with stack-based data structures.
///
/// Value: 32 tools
///
/// Usage: Function calling, tool-augmented generation, API integrations
pub const MAX_TOOLS: usize = 32;

/// Maximum documents per request (compile-time bounded)
///
/// This constant defines the upper limit for the number of documents that can be
/// included as context in a single completion request. This enables efficient
/// document processing with pre-allocated storage.
///
/// Value: 64 documents
///
/// Usage: RAG (Retrieval-Augmented Generation), document QA, context injection
pub const MAX_DOCUMENTS: usize = 64;