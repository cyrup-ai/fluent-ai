//! Candle model definitions that comply with domain traits

use std::borrow::Cow;
use std::num::NonZeroU32;

use fluent_ai_domain::model::{ModelInfo, traits::Model};
use serde::{Deserialize, Serialize};

/// Candle-supported models for local inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CandleModel {
    /// DeepSeek-V3 Devstral 22B parameter model (DEFAULT)
    Devstral22B,
    /// LLaMA 2 7B parameter model
    Llama27B,
    /// LLaMA 2 13B parameter model
    Llama213B,
    /// Mistral 7B parameter model
    Mistral7B,
    /// Code Llama 7B parameter model
    CodeLlama7B,
    /// Phi-3 Mini parameter model
    Phi3Mini,
    /// Gemma 2B parameter model
    Gemma2B,
    /// Gemma 7B parameter model
    Gemma7B,
    /// Kimi-K2 FP16 parameter model
    KimiK2Fp16,
    /// Kimi-K2 FP8 parameter model
    KimiK2Fp8,
}

const DEVSTRAL_22B_INFO: ModelInfo = ModelInfo {
    provider_name: "candle",
    name: "devstral-22b",
    max_input_tokens: NonZeroU32::new(32768),
    max_output_tokens: NonZeroU32::new(32768),
    input_price: None,
    output_price: None,
    supports_vision: false,
    supports_function_calling: true,
    supports_streaming: true,
    supports_embeddings: false,
    requires_max_tokens: false,
    supports_thinking: true,
    optimal_thinking_budget: Some(4096),
    system_prompt_prefix: None,
    real_name: None,
    model_type: None,
    patch: None,
};

const LLAMA2_7B_INFO: ModelInfo = ModelInfo {
    provider_name: "candle",
    name: "llama2-7b",
    max_input_tokens: NonZeroU32::new(4096),
    max_output_tokens: NonZeroU32::new(4096),
    input_price: None,
    output_price: None,
    supports_vision: false,
    supports_function_calling: false,
    supports_streaming: true,
    supports_embeddings: false,
    requires_max_tokens: false,
    supports_thinking: false,
    optimal_thinking_budget: None,
    system_prompt_prefix: None,
    real_name: None,
    model_type: None,
    patch: None,
};

const LLAMA2_13B_INFO: ModelInfo = ModelInfo {
    provider_name: "candle",
    name: "llama2-13b",
    max_input_tokens: NonZeroU32::new(4096),
    max_output_tokens: NonZeroU32::new(4096),
    input_price: None,
    output_price: None,
    supports_vision: false,
    supports_function_calling: false,
    supports_streaming: true,
    supports_embeddings: false,
    requires_max_tokens: false,
    supports_thinking: false,
    optimal_thinking_budget: None,
    system_prompt_prefix: None,
    real_name: None,
    model_type: None,
    patch: None,
};

const MISTRAL_7B_INFO: ModelInfo = ModelInfo {
    provider_name: "candle",
    name: "mistral-7b",
    max_input_tokens: NonZeroU32::new(8192),
    max_output_tokens: NonZeroU32::new(8192),
    input_price: None,
    output_price: None,
    supports_vision: false,
    supports_function_calling: true,
    supports_streaming: true,
    supports_embeddings: false,
    requires_max_tokens: false,
    supports_thinking: false,
    optimal_thinking_budget: None,
    system_prompt_prefix: None,
    real_name: None,
    model_type: None,
    patch: None,
};

const CODELLAMA_7B_INFO: ModelInfo = ModelInfo {
    provider_name: "candle",
    name: "codellama-7b",
    max_input_tokens: NonZeroU32::new(16384),
    max_output_tokens: NonZeroU32::new(16384),
    input_price: None,
    output_price: None,
    supports_vision: false,
    supports_function_calling: false,
    supports_streaming: true,
    supports_embeddings: false,
    requires_max_tokens: false,
    supports_thinking: false,
    optimal_thinking_budget: None,
    system_prompt_prefix: None,
    real_name: None,
    model_type: None,
    patch: None,
};

const PHI3_MINI_INFO: ModelInfo = ModelInfo {
    provider_name: "candle",
    name: "phi3-mini",
    max_input_tokens: NonZeroU32::new(4096),
    max_output_tokens: NonZeroU32::new(4096),
    input_price: None,
    output_price: None,
    supports_vision: false,
    supports_function_calling: true,
    supports_streaming: true,
    supports_embeddings: false,
    requires_max_tokens: false,
    supports_thinking: false,
    optimal_thinking_budget: None,
    system_prompt_prefix: None,
    real_name: None,
    model_type: None,
    patch: None,
};

const GEMMA_2B_INFO: ModelInfo = ModelInfo {
    provider_name: "candle",
    name: "gemma-2b",
    max_input_tokens: NonZeroU32::new(8192),
    max_output_tokens: NonZeroU32::new(8192),
    input_price: None,
    output_price: None,
    supports_vision: false,
    supports_function_calling: false,
    supports_streaming: true,
    supports_embeddings: false,
    requires_max_tokens: false,
    supports_thinking: false,
    optimal_thinking_budget: None,
    system_prompt_prefix: None,
    real_name: None,
    model_type: None,
    patch: None,
};

const GEMMA_7B_INFO: ModelInfo = ModelInfo {
    provider_name: "candle",
    name: "gemma-7b",
    max_input_tokens: NonZeroU32::new(8192),
    max_output_tokens: NonZeroU32::new(8192),
    input_price: None,
    output_price: None,
    supports_vision: false,
    supports_function_calling: false,
    supports_streaming: true,
    supports_embeddings: false,
    requires_max_tokens: false,
    supports_thinking: false,
    optimal_thinking_budget: None,
    system_prompt_prefix: None,
    real_name: None,
    model_type: None,
    patch: None,
};

const KIMI_K2_FP16_INFO: ModelInfo = ModelInfo {
    provider_name: "candle",
    name: "kimi-k2-fp16",
    max_input_tokens: NonZeroU32::new(32768),
    max_output_tokens: NonZeroU32::new(32768),
    input_price: None,
    output_price: None,
    supports_vision: false,
    supports_function_calling: true,
    supports_streaming: true,
    supports_embeddings: false,
    requires_max_tokens: false,
    supports_thinking: true,
    optimal_thinking_budget: Some(4096),
    system_prompt_prefix: None,
    real_name: None,
    model_type: None,
    patch: None,
};

const KIMI_K2_FP8_INFO: ModelInfo = ModelInfo {
    provider_name: "candle",
    name: "kimi-k2-fp8",
    max_input_tokens: NonZeroU32::new(32768),
    max_output_tokens: NonZeroU32::new(32768),
    input_price: None,
    output_price: None,
    supports_vision: false,
    supports_function_calling: true,
    supports_streaming: true,
    supports_embeddings: false,
    requires_max_tokens: false,
    supports_thinking: true,
    optimal_thinking_budget: Some(4096),
    system_prompt_prefix: None,
    real_name: None,
    model_type: None,
    patch: None,
};

impl Model for CandleModel {
    fn info(&self) -> &'static ModelInfo {
        match self {
            CandleModel::Devstral22B => &DEVSTRAL_22B_INFO,
            CandleModel::Llama27B => &LLAMA2_7B_INFO,
            CandleModel::Llama213B => &LLAMA2_13B_INFO,
            CandleModel::Mistral7B => &MISTRAL_7B_INFO,
            CandleModel::CodeLlama7B => &CODELLAMA_7B_INFO,
            CandleModel::Phi3Mini => &PHI3_MINI_INFO,
            CandleModel::Gemma2B => &GEMMA_2B_INFO,
            CandleModel::Gemma7B => &GEMMA_7B_INFO,
            CandleModel::KimiK2Fp16 => &KIMI_K2_FP16_INFO,
            CandleModel::KimiK2Fp8 => &KIMI_K2_FP8_INFO,
        }
    }
}

impl std::fmt::Display for CandleModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            CandleModel::Devstral22B => "DeepSeek-V3 Devstral 22B",
            CandleModel::Llama27B => "LLaMA 2 7B",
            CandleModel::Llama213B => "LLaMA 2 13B",
            CandleModel::Mistral7B => "Mistral 7B",
            CandleModel::CodeLlama7B => "Code Llama 7B",
            CandleModel::Phi3Mini => "Phi-3 Mini",
            CandleModel::Gemma2B => "Gemma 2B",
            CandleModel::Gemma7B => "Gemma 7B",
            CandleModel::KimiK2Fp16 => "Kimi-K2 FP16",
            CandleModel::KimiK2Fp8 => "Kimi-K2 FP8",
        };
        write!(f, "{}", name)
    }
}

impl Default for CandleModel {
    fn default() -> Self {
        CandleModel::Devstral22B
    }
}

/// Device configuration for Candle inference
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CandleDevice {
    /// Use CPU for inference
    Cpu,
    /// Use CUDA GPU for inference
    Cuda(u32),
    /// Use Metal GPU for inference (Apple Silicon)
    Metal,
    /// Auto-detect best available device
    Auto,
}

impl Default for CandleDevice {
    fn default() -> Self {
        CandleDevice::Auto
    }
}

impl std::fmt::Display for CandleDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CandleDevice::Cpu => write!(f, "CPU"),
            CandleDevice::Cuda(id) => write!(f, "CUDA:{}", id),
            CandleDevice::Metal => write!(f, "Metal"),
            CandleDevice::Auto => write!(f, "Auto"),
        }
    }
}

/// Extended model information for Candle models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleModelInfo {
    /// Base model information
    pub model: CandleModel,
    /// Device configuration
    pub device: CandleDevice,
    /// Model size in parameters
    pub parameters: u64,
    /// Context length
    pub context_length: u32,
    /// Whether the model supports function calling
    pub supports_tools: bool,
    /// Whether the model supports vision
    pub supports_vision: bool,
    /// Model file path or URL
    pub model_path: Option<String>,
    /// Tokenizer path or URL
    pub tokenizer_path: Option<String>,
}

impl CandleModelInfo {
    /// Create new model info
    pub fn new(model: CandleModel, device: CandleDevice) -> Self {
        let (parameters, context_length, supports_tools, supports_vision) = match model {
            CandleModel::Devstral22B => (22_000_000_000, 32768, true, false),
            CandleModel::Llama27B => (7_000_000_000, 4096, false, false),
            CandleModel::Llama213B => (13_000_000_000, 4096, false, false),
            CandleModel::Mistral7B => (7_000_000_000, 8192, true, false),
            CandleModel::CodeLlama7B => (7_000_000_000, 16384, false, false),
            CandleModel::Phi3Mini => (3_800_000_000, 128000, false, false),
            CandleModel::Gemma2B => (2_000_000_000, 8192, false, false),
            CandleModel::Gemma7B => (7_000_000_000, 8192, false, false),
            CandleModel::KimiK2Fp16 => (2_600_000_000, 128000, true, true),
            CandleModel::KimiK2Fp8 => (2_600_000_000, 128000, true, true),
        };

        Self {
            model,
            device,
            parameters,
            context_length,
            supports_tools,
            supports_vision,
            model_path: None,
            tokenizer_path: None,
        }
    }

    /// Set model file path
    pub fn with_model_path(mut self, path: String) -> Self {
        self.model_path = Some(path);
        self
    }

    /// Set tokenizer file path
    pub fn with_tokenizer_path(mut self, path: String) -> Self {
        self.tokenizer_path = Some(path);
        self
    }
}
