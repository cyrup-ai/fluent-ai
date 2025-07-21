//! Candle model definitions that comply with domain traits

use std::borrow::Cow;
use std::num::NonZeroU32;

use fluent_ai_domain::model::{ModelInfo, traits::Model};
use serde::{Deserialize, Serialize};

/// Candle-supported models for local inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CandleModel {
    /// DeepSeek-V3 Devstral 22B parameter model (DEFAULT)
    Devstral_22B,
    /// LLaMA 2 7B parameter model
    Llama2_7B,
    /// LLaMA 2 13B parameter model
    Llama2_13B,
    /// Mistral 7B parameter model
    Mistral_7B,
    /// Code Llama 7B parameter model
    CodeLlama_7B,
    /// Phi-3 Mini model
    Phi3_Mini,
    /// Gemma 2B model
    Gemma_2B,
    /// Gemma 7B model
    Gemma_7B,
}

// Static model info constants  
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

impl Model for CandleModel {
    fn info(&self) -> &'static ModelInfo {
        match self {
            CandleModel::Devstral_22B => &DEVSTRAL_22B_INFO,
            CandleModel::Llama2_7B => ModelInfoData {
                provider_name: "candle".to_string(),
                name: "llama2-7b".to_string(),
                max_input_tokens: Some(4096),
                max_output_tokens: Some(4096),
                input_price: None, // Local inference - no API costs
                output_price: None,
                supports_vision: Some(false),
                supports_function_calling: Some(false),
                require_max_tokens: Some(false),
                supports_thinking: Some(false),
                optimal_thinking_budget: None,
            },
            CandleModel::Llama2_13B => ModelInfoData {
                provider_name: "candle".to_string(),
                name: "llama2-13b".to_string(),
                max_input_tokens: Some(4096),
                max_output_tokens: Some(4096),
                input_price: None,
                output_price: None,
                supports_vision: Some(false),
                supports_function_calling: Some(false),
                require_max_tokens: Some(false),
                supports_thinking: Some(false),
                optimal_thinking_budget: None,
            },
            CandleModel::Mistral_7B => ModelInfoData {
                provider_name: "candle".to_string(),
                name: "mistral-7b".to_string(),
                max_input_tokens: Some(8192),
                max_output_tokens: Some(8192),
                input_price: None,
                output_price: None,
                supports_vision: Some(false),
                supports_function_calling: Some(true),
                require_max_tokens: Some(false),
                supports_thinking: Some(false),
                optimal_thinking_budget: None,
            },
            CandleModel::CodeLlama_7B => ModelInfoData {
                provider_name: "candle".to_string(),
                name: "codellama-7b".to_string(),
                max_input_tokens: Some(16384),
                max_output_tokens: Some(16384),
                input_price: None,
                output_price: None,
                supports_vision: Some(false),
                supports_function_calling: Some(false),
                require_max_tokens: Some(false),
                supports_thinking: Some(false),
                optimal_thinking_budget: None,
            },
            CandleModel::Phi3_Mini => ModelInfoData {
                provider_name: "candle".to_string(),
                name: "phi3-mini".to_string(),
                max_input_tokens: Some(4096),
                max_output_tokens: Some(4096),
                input_price: None,
                output_price: None,
                supports_vision: Some(false),
                supports_function_calling: Some(true),
                require_max_tokens: Some(false),
                supports_thinking: Some(false),
                optimal_thinking_budget: None,
            },
            CandleModel::Gemma_2B => ModelInfoData {
                provider_name: "candle".to_string(),
                name: "gemma-2b".to_string(),
                max_input_tokens: Some(8192),
                max_output_tokens: Some(8192),
                input_price: None,
                output_price: None,
                supports_vision: Some(false),
                supports_function_calling: Some(false),
                require_max_tokens: Some(false),
                supports_thinking: Some(false),
                optimal_thinking_budget: None,
            },
            CandleModel::Gemma_7B => ModelInfoData {
                provider_name: "candle".to_string(),
                name: "gemma-7b".to_string(),
                max_input_tokens: Some(8192),
                max_output_tokens: Some(8192),
                input_price: None,
                output_price: None,
                supports_vision: Some(false),
                supports_function_calling: Some(false),
                require_max_tokens: Some(false),
                supports_thinking: Some(false),
                optimal_thinking_budget: None,
            },
        };

        ModelInfo::from_data(data)
    }
}

impl std::fmt::Display for CandleModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            CandleModel::Devstral_22B => "DeepSeek-V3 Devstral 22B",
            CandleModel::Llama2_7B => "LLaMA 2 7B",
            CandleModel::Llama2_13B => "LLaMA 2 13B",
            CandleModel::Mistral_7B => "Mistral 7B",
            CandleModel::CodeLlama_7B => "Code Llama 7B",
            CandleModel::Phi3_Mini => "Phi-3 Mini",
            CandleModel::Gemma_2B => "Gemma 2B",
            CandleModel::Gemma_7B => "Gemma 7B",
        };
        write!(f, "{}", name)
    }
}

impl Default for CandleModel {
    fn default() -> Self {
        CandleModel::Devstral_22B
    }
}

/// Candle model information wrapper
// CandleModelInfo is now imported from fluent_ai_domain::model
// Removed duplicated CandleModelInfo struct - use canonical domain type
#[derive(Debug, Clone)]
pub struct CandleModelInfo {
    /// Model path
    pub model_path: String,
    /// Tokenizer path
    pub tokenizer_path: Option<String>,
    /// Device to use for inference
    pub device: CandleDevice,
}

/// Candle device options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandleDevice {
    /// CPU inference
    Cpu,
    /// CUDA GPU inference
    Cuda(u32),
    /// Metal GPU inference (macOS)
    Metal(u32),
}

impl Default for CandleDevice {
    fn default() -> Self {
        // Default to Metal on macOS, CPU otherwise
        #[cfg(target_os = "macos")]
        return CandleDevice::Metal(0);

        #[cfg(not(target_os = "macos"))]
        CandleDevice::Cpu
    }
}

impl std::fmt::Display for CandleDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CandleDevice::Cpu => write!(f, "CPU"),
            CandleDevice::Cuda(id) => write!(f, "CUDA:{}", id),
            CandleDevice::Metal(id) => write!(f, "Metal:{}", id),
        }
    }
}
