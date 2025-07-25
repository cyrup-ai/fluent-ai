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
    /// Phi-3 Mini parameter model
    Phi3_Mini,
    /// Gemma 2B parameter model
    Gemma_2B,
    /// Gemma 7B parameter model
    Gemma_7B,
    /// Kimi-K2 FP16 parameter model
    KimiK2_FP16,
    /// Kimi-K2 FP8 parameter model
    KimiK2_FP8}

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
    patch: None};

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
    patch: None};

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
    patch: None};

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
    patch: None};

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
    patch: None};

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
    patch: None};

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
    patch: None};

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
    patch: None};

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
    patch: None};

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
    patch: None};

impl Model for CandleModel {
    fn info(&self) -> &'static ModelInfo {
        match self {
            CandleModel::Devstral_22B => &DEVSTRAL_22B_INFO,
            CandleModel::Llama2_7B => &LLAMA2_7B_INFO,
            CandleModel::Llama2_13B => &LLAMA2_13B_INFO,
            CandleModel::Mistral_7B => &MISTRAL_7B_INFO,
            CandleModel::CodeLlama_7B => &CODELLAMA_7B_INFO,
            CandleModel::Phi3_Mini => &PHI3_MINI_INFO,
            CandleModel::Gemma_2B => &GEMMA_2B_INFO,
            CandleModel::Gemma_7B => &GEMMA_7B_INFO,
            CandleModel::KimiK2_FP16 => &KIMI_K2_FP16_INFO,
            CandleModel::KimiK2_FP8 => &KIMI_K2_FP8_INFO}
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
            CandleModel::KimiK2_FP16 => "Kimi-K2 FP16",
            CandleModel::KimiK2_FP8 => "Kimi-K2 FP8"};
        write!(f, "{}", name)
    }
}

impl Default for CandleModel {
    fn default() -> Self {
        CandleModel::Devstral_22B
    }
}
