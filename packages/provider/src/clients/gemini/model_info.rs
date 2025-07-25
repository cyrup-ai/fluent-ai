//! Gemini model configurations and ModelInfo implementations
//!
//! Compile-time model configurations for blazing-fast zero-allocation initialization
//! All configurations are const and embedded in the binary for optimal performance

use fluent_ai_domain::chunk::CompletionChunk;

use crate::AsyncStream;
use crate::clients::gemini::completion::GeminiCompletionBuilder;
use crate::completion_provider::{CompletionProvider, ModelConfig, ModelInfo, ModelPrompt};

/// Gemini 2.5 Pro model information
#[derive(Debug, Clone, Copy)]
pub struct Gemini25Pro;

impl ModelInfo for Gemini25Pro {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 2000000, // 2M context
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: true,
        provider: "gemini",
        model_name: "gemini-2.5-pro-preview-06-05"};
}

impl ModelPrompt for Gemini25Pro {
    type Provider = GeminiCompletionBuilder;
}

/// Gemini 2.5 Flash model information
#[derive(Debug, Clone, Copy)]
pub struct Gemini25Flash;

impl ModelInfo for Gemini25Flash {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 1000000, // 1M context
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: true,
        provider: "gemini",
        model_name: "gemini-2.5-flash-preview-05-20"};
}

impl ModelPrompt for Gemini25Flash {
    type Provider = GeminiCompletionBuilder;
}

/// Gemini 2.0 Flash model information
#[derive(Debug, Clone, Copy)]
pub struct Gemini20Flash;

impl ModelInfo for Gemini20Flash {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 1000000, // 1M context
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: true,
        provider: "gemini",
        model_name: "gemini-2.0-flash"};
}

impl ModelPrompt for Gemini20Flash {
    type Provider = GeminiCompletionBuilder;
}

/// Gemini 1.5 Pro model information
#[derive(Debug, Clone, Copy)]
pub struct Gemini15Pro;

impl ModelInfo for Gemini15Pro {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 2000000, // 2M context
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: true,
        provider: "gemini",
        model_name: "gemini-1.5-pro"};
}

impl ModelPrompt for Gemini15Pro {
    type Provider = GeminiCompletionBuilder;
}

/// Gemini 1.5 Flash model information
#[derive(Debug, Clone, Copy)]
pub struct Gemini15Flash;

impl ModelInfo for Gemini15Flash {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 1000000, // 1M context
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        provider: "gemini",
        model_name: "gemini-1.5-flash"};
}

impl ModelPrompt for Gemini15Flash {
    type Provider = GeminiCompletionBuilder;
}

/// Gemini 1.5 Pro 8B model information
#[derive(Debug, Clone, Copy)]
pub struct Gemini15Pro8B;

impl ModelInfo for Gemini15Pro8B {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 1000000, // 1M context
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        provider: "gemini",
        model_name: "gemini-1.5-pro-8b"};
}

impl ModelPrompt for Gemini15Pro8B {
    type Provider = GeminiCompletionBuilder;
}

/// Gemini 1.0 Pro model information
#[derive(Debug, Clone, Copy)]
pub struct Gemini10Pro;

impl ModelInfo for Gemini10Pro {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 2048,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 32768, // 32K context
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: false,
        supports_audio: false,
        provider: "gemini",
        model_name: "gemini-1.0-pro"};
}

impl ModelPrompt for Gemini10Pro {
    type Provider = GeminiCompletionBuilder;
}

/// Get model configuration by name (zero allocation lookup)
#[inline(always)]
pub const fn get_model_config(model_name: &str) -> &'static ModelConfig {
    match model_name {
        "gemini-2.5-pro-preview-06-05"
        | "gemini-2.5-pro-preview-05-06"
        | "gemini-2.5-pro-preview-03-25"
        | "gemini-2.5-pro-exp-03-25" => &Gemini25Pro::CONFIG,
        "gemini-2.5-flash-preview-05-20" | "gemini-2.5-flash-preview-04-17" => {
            &Gemini25Flash::CONFIG
        }
        "gemini-2.0-flash" | "gemini-2.0-flash-lite" => &Gemini20Flash::CONFIG,
        "gemini-1.5-pro" => &Gemini15Pro::CONFIG,
        "gemini-1.5-flash" => &Gemini15Flash::CONFIG,
        "gemini-1.5-pro-8b" => &Gemini15Pro8B::CONFIG,
        "gemini-1.0-pro" => &Gemini10Pro::CONFIG,
        _ => &Gemini15Pro::CONFIG, // Default fallback to latest stable
    }
}

/// Get model name from enum variant (zero allocation)
#[inline(always)]
pub const fn model_name_from_variant(variant: &str) -> &'static str {
    match variant {
        "GeminiGemini25Pro" => "gemini-2.5-pro-preview-06-05",
        "GeminiGemini25Flash" => "gemini-2.5-flash-preview-05-20",
        "GeminiGemini20Flash" => "gemini-2.0-flash",
        "GeminiGemini15Pro" => "gemini-1.5-pro",
        "GeminiGemini15Flash" => "gemini-1.5-flash",
        "GeminiGemini15Pro8B" => "gemini-1.5-pro-8b",
        "GeminiGemini10Pro" => "gemini-1.0-pro",
        _ => "gemini-1.5-pro", // Default fallback
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
