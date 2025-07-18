//! VertexAI model configurations and ModelInfo implementations
//!
//! Compile-time model configurations for blazing-fast zero-allocation initialization
//! All configurations are const and embedded in the binary for optimal performance

use crate::completion_provider::{ModelConfig, ModelInfo, ModelPrompt, CompletionProvider};
use crate::clients::vertexai::completion::VertexAICompletionBuilder;
use crate::AsyncStream;
use fluent_ai_domain::chunk::CompletionChunk;

/// Gemini 2.5 Flash model information
#[derive(Debug, Clone, Copy)]
pub struct Gemini25Flash;

impl ModelInfo for Gemini25Flash {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 0.95,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 1000000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        provider: "vertexai",
        model_name: "gemini-2.5-flash",
    };
}

impl ModelPrompt for Gemini25Flash {
    type Provider = VertexAICompletionBuilder;
}

/// Gemini 2.5 Pro model information
#[derive(Debug, Clone, Copy)]
pub struct Gemini25Pro;

impl ModelInfo for Gemini25Pro {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 0.95,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 2000000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        provider: "vertexai",
        model_name: "gemini-2.5-pro",
    };
}

impl ModelPrompt for Gemini25Pro {
    type Provider = VertexAICompletionBuilder;
}

/// Gemini 1.5 Flash model information
#[derive(Debug, Clone, Copy)]
pub struct Gemini15Flash;

impl ModelInfo for Gemini15Flash {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 0.95,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 1000000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        provider: "vertexai",
        model_name: "gemini-1.5-flash",
    };
}

impl ModelPrompt for Gemini15Flash {
    type Provider = VertexAICompletionBuilder;
}

/// Gemini 1.5 Pro model information
#[derive(Debug, Clone, Copy)]
pub struct Gemini15Pro;

impl ModelInfo for Gemini15Pro {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 0.95,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 2000000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        provider: "vertexai",
        model_name: "gemini-1.5-pro",
    };
}

impl ModelPrompt for Gemini15Pro {
    type Provider = VertexAICompletionBuilder;
}

/// Claude 3.5 Sonnet via VertexAI model information
#[derive(Debug, Clone, Copy)]
pub struct Claude35Sonnet;

impl ModelInfo for Claude35Sonnet {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 0.95,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 200000,
        system_prompt: "You are Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        provider: "vertexai",
        model_name: "claude-3-5-sonnet",
    };
}

impl ModelPrompt for Claude35Sonnet {
    type Provider = VertexAICompletionBuilder;
}

/// Claude 3.5 Haiku via VertexAI model information
#[derive(Debug, Clone, Copy)]
pub struct Claude35Haiku;

impl ModelInfo for Claude35Haiku {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 0.95,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 200000,
        system_prompt: "You are Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        provider: "vertexai",
        model_name: "claude-3-5-haiku",
    };
}

impl ModelPrompt for Claude35Haiku {
    type Provider = VertexAICompletionBuilder;
}

/// Claude 3 Opus via VertexAI model information
#[derive(Debug, Clone, Copy)]
pub struct Claude3Opus;

impl ModelInfo for Claude3Opus {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 4096,
        temperature: 0.7,
        top_p: 0.95,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 200000,
        system_prompt: "You are Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        provider: "vertexai",
        model_name: "claude-3-opus",
    };
}

impl ModelPrompt for Claude3Opus {
    type Provider = VertexAICompletionBuilder;
}

/// Mistral Large via VertexAI model information
#[derive(Debug, Clone, Copy)]
pub struct MistralLarge;

impl ModelInfo for MistralLarge {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 0.95,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 128000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: false,
        supports_audio: false,
        provider: "vertexai",
        model_name: "mistral-large",
    };
}

impl ModelPrompt for MistralLarge {
    type Provider = VertexAICompletionBuilder;
}

/// Model configuration lookup function
pub fn get_model_config(model_name: &str) -> &'static ModelConfig {
    match model_name {
        "gemini-2.5-flash" => &Gemini25Flash::CONFIG,
        "gemini-2.5-pro" => &Gemini25Pro::CONFIG,
        "gemini-1.5-flash" => &Gemini15Flash::CONFIG,
        "gemini-1.5-pro" => &Gemini15Pro::CONFIG,
        "claude-3-5-sonnet" => &Claude35Sonnet::CONFIG,
        "claude-3-5-haiku" => &Claude35Haiku::CONFIG,
        "claude-3-opus" => &Claude3Opus::CONFIG,
        "mistral-large" => &MistralLarge::CONFIG,
        _ => &Gemini25Flash::CONFIG, // Default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_configs() {
        let config = get_model_config("gemini-2.5-flash");
        assert_eq!(config.model_name, "gemini-2.5-flash");
        assert_eq!(config.provider, "vertexai");
        assert!(config.supports_vision);
        
        let config = get_model_config("claude-3-5-sonnet");
        assert_eq!(config.model_name, "claude-3-5-sonnet");
        assert_eq!(config.provider, "vertexai");
        assert!(config.supports_tools);
    }
    
    #[test]
    fn test_fallback_config() {
        let config = get_model_config("unknown-model");
        assert_eq!(config.model_name, "gemini-2.5-flash");
    }
}