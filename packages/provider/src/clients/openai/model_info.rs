//! OpenAI model configurations and ModelInfo implementations
//!
//! Compile-time model configurations for blazing-fast zero-allocation initialization
//! All configurations are const and embedded in the binary for optimal performance

use crate::completion_provider::{ModelConfig, ModelInfo, ModelPrompt, CompletionProvider};
use crate::clients::openai::completion::OpenAICompletionBuilder;
use crate::AsyncStream;
use fluent_ai_domain::chunk::CompletionChunk;

/// GPT-4o model information
#[derive(Debug, Clone, Copy)]
pub struct Gpt4o;

impl ModelInfo for Gpt4o {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 4096,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 128000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: true,
        provider: "openai",
        model_name: "gpt-4o",
    };
}

impl ModelPrompt for Gpt4o {
    type Provider = OpenAICompletionBuilder;
}

/// GPT-4o Mini model information
#[derive(Debug, Clone, Copy)]
pub struct Gpt4oMini;

impl ModelInfo for Gpt4oMini {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 16384,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 128000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        provider: "openai",
        model_name: "gpt-4o-mini",
    };
}

impl ModelPrompt for Gpt4oMini {
    type Provider = OpenAICompletionBuilder;
}

/// GPT-4 Turbo model information
#[derive(Debug, Clone, Copy)]
pub struct Gpt4Turbo;

impl ModelInfo for Gpt4Turbo {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 4096,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 128000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        provider: "openai",
        model_name: "gpt-4-turbo",
    };
}

impl ModelPrompt for Gpt4Turbo {
    type Provider = OpenAICompletionBuilder;
}

/// GPT-3.5 Turbo model information
#[derive(Debug, Clone, Copy)]
pub struct Gpt35Turbo;

impl ModelInfo for Gpt35Turbo {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 4096,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 16385,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: false,
        supports_audio: false,
        provider: "openai",
        model_name: "gpt-3.5-turbo",
    };
}

impl ModelPrompt for Gpt35Turbo {
    type Provider = OpenAICompletionBuilder;
}

/// Get model configuration by name (zero allocation lookup)
#[inline(always)]
pub fn get_model_config(model_name: &'static str) -> &'static ModelConfig {
    // Use central model config system from model_info.rs
    crate::model_info::get_model_config(model_name)
}

/// Get model name from enum variant (zero allocation)
#[inline(always)]
pub const fn model_name_from_variant(variant: &str) -> &'static str {
    match variant {
        "OpenaiGpt4o" => "gpt-4o",
        "OpenaiGpt4oMini" => "gpt-4o-mini", 
        "OpenaiGpt4Turbo" => "gpt-4-turbo",
        "OpenaiGpt35Turbo" => "gpt-3.5-turbo",
        _ => "gpt-4o", // Default fallback
    }
}

/// Check if model supports tools at compile time
#[inline(always)]
pub const fn model_supports_tools(model_name: &str) -> bool {
    get_model_config(model_name).supports_tools
}

/// Check if model supports vision at compile time
#[inline(always)]
pub const fn model_supports_vision(model_name: &str) -> bool {
    get_model_config(model_name).supports_vision
}

/// Check if model supports audio at compile time
#[inline(always)]
pub const fn model_supports_audio(model_name: &str) -> bool {
    get_model_config(model_name).supports_audio
}

/// Get model context length at compile time
#[inline(always)]
pub const fn get_model_context_length(model_name: &str) -> u32 {
    get_model_config(model_name).context_length
}

/// Get model max output tokens at compile time
#[inline(always)]
pub const fn get_model_max_output_tokens(model_name: &str) -> u32 {
    get_model_config(model_name).max_tokens
}