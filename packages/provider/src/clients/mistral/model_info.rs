//! Mistral model configurations and ModelInfo implementations
//!
//! Compile-time model configurations for blazing-fast zero-allocation initialization
//! All configurations are const and embedded in the binary for optimal performance
//!
//! Enables elegant ergonomic syntax: `Model::MistralLarge.prompt("Hello world")`

use fluent_ai_domain::chunk::CompletionChunk;

use crate::AsyncStream;
use crate::clients::mistral::completion::MistralCompletionBuilder;
use crate::completion_provider::{CompletionProvider, ModelConfig, ModelInfo, ModelPrompt};

// ================================================================================================
// Premium Paid Models
// ================================================================================================

/// Mistral Large - Premium flagship model with 128k context
#[derive(Debug, Clone, Copy)]
pub struct MistralLarge;

impl ModelInfo for MistralLarge {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 128000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        supports_thinking: false,
        optimal_thinking_budget: 0,
        provider: "mistral",
        model_name: "mistral-large-latest"};
}

impl ModelPrompt for MistralLarge {
    type Provider = MistralCompletionBuilder;
}

/// Codestral - Code-specialized model with 32k context
#[derive(Debug, Clone, Copy)]
pub struct Codestral;

impl ModelInfo for Codestral {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 32000,
        system_prompt: "You are a helpful AI assistant specialized in code generation and programming tasks.",
        supports_tools: true,
        supports_vision: false,
        supports_audio: false,
        supports_thinking: false,
        optimal_thinking_budget: 0,
        provider: "mistral",
        model_name: "codestral-latest"};
}

impl ModelPrompt for Codestral {
    type Provider = MistralCompletionBuilder;
}

/// Pixtral Large - Premium vision model with 128k context
#[derive(Debug, Clone, Copy)]
pub struct PixtralLarge;

impl ModelInfo for PixtralLarge {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 128000,
        system_prompt: "You are a helpful AI assistant with advanced vision capabilities.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        supports_thinking: false,
        optimal_thinking_budget: 0,
        provider: "mistral",
        model_name: "pixtral-large-latest"};
}

impl ModelPrompt for PixtralLarge {
    type Provider = MistralCompletionBuilder;
}

/// Mistral Saba - Advanced reasoning model with thinking capabilities
#[derive(Debug, Clone, Copy)]
pub struct MistralSaba;

impl ModelInfo for MistralSaba {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 128000,
        system_prompt: "You are a helpful AI assistant with advanced reasoning capabilities.",
        supports_tools: true,
        supports_vision: false,
        supports_audio: false,
        supports_thinking: true,
        optimal_thinking_budget: 5000,
        provider: "mistral",
        model_name: "mistral-saba-latest"};
}

impl ModelPrompt for MistralSaba {
    type Provider = MistralCompletionBuilder;
}

/// Ministral 3B - Compact efficient model with 128k context
#[derive(Debug, Clone, Copy)]
pub struct Ministral3B;

impl ModelInfo for Ministral3B {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 128000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: false,
        supports_audio: false,
        supports_thinking: false,
        optimal_thinking_budget: 0,
        provider: "mistral",
        model_name: "ministral-3b-latest"};
}

impl ModelPrompt for Ministral3B {
    type Provider = MistralCompletionBuilder;
}

/// Ministral 8B - Medium efficient model with 128k context
#[derive(Debug, Clone, Copy)]
pub struct Ministral8B;

impl ModelInfo for Ministral8B {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 128000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: false,
        supports_audio: false,
        supports_thinking: false,
        optimal_thinking_budget: 0,
        provider: "mistral",
        model_name: "ministral-8b-latest"};
}

impl ModelPrompt for Ministral8B {
    type Provider = MistralCompletionBuilder;
}

// ================================================================================================
// Free Open Models
// ================================================================================================

/// Mistral Small - Free tier model with 32k context
#[derive(Debug, Clone, Copy)]
pub struct MistralSmall;

impl ModelInfo for MistralSmall {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 32000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: false,
        supports_audio: false,
        supports_thinking: false,
        optimal_thinking_budget: 0,
        provider: "mistral",
        model_name: "mistral-small-latest"};
}

impl ModelPrompt for MistralSmall {
    type Provider = MistralCompletionBuilder;
}

/// Pixtral Small - Free vision model with 128k context
#[derive(Debug, Clone, Copy)]
pub struct PixtralSmall;

impl ModelInfo for PixtralSmall {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 128000,
        system_prompt: "You are a helpful AI assistant with vision capabilities.",
        supports_tools: true,
        supports_vision: true,
        supports_audio: false,
        supports_thinking: false,
        optimal_thinking_budget: 0,
        provider: "mistral",
        model_name: "pixtral-12b-2409"};
}

impl ModelPrompt for PixtralSmall {
    type Provider = MistralCompletionBuilder;
}

/// Mistral Nemo - Open source model with 128k context
#[derive(Debug, Clone, Copy)]
pub struct MistralNemo;

impl ModelInfo for MistralNemo {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 128000,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: true,
        supports_vision: false,
        supports_audio: false,
        supports_thinking: false,
        optimal_thinking_budget: 0,
        provider: "mistral",
        model_name: "open-mistral-nemo"};
}

impl ModelPrompt for MistralNemo {
    type Provider = MistralCompletionBuilder;
}

/// Codestral Mamba - Open source code model with 256k context
#[derive(Debug, Clone, Copy)]
pub struct CodestralMamba;

impl ModelInfo for CodestralMamba {
    const CONFIG: ModelConfig = ModelConfig {
        max_tokens: 8192,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: 256000,
        system_prompt: "You are a helpful AI assistant specialized in code generation with large context capabilities.",
        supports_tools: true,
        supports_vision: false,
        supports_audio: false,
        supports_thinking: false,
        optimal_thinking_budget: 0,
        provider: "mistral",
        model_name: "open-codestral-mamba"};
}

impl ModelPrompt for CodestralMamba {
    type Provider = MistralCompletionBuilder;
}

// ================================================================================================
// Zero-allocation helper functions for blazing-fast lookups
// ================================================================================================

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
        "MistralMistralLarge" => "mistral-large-latest",
        "MistralCodestral" => "codestral-latest",
        "MistralPixtralLarge" => "pixtral-large-latest",
        "MistralMistralSaba" => "mistral-saba-latest",
        "MistralMinistral3B" => "ministral-3b-latest",
        "MistralMinistral8B" => "ministral-8b-latest",
        "MistralMistralSmall" => "mistral-small-latest",
        "MistralPixtralSmall" => "pixtral-12b-2409",
        "MistralMistralNemo" => "open-mistral-nemo",
        "MistralCodestralMamba" => "open-codestral-mamba",
        _ => "mistral-large-latest", // Default fallback to flagship model
    }
}

/// Check if model supports tools at compile time
#[inline(always)]
pub const fn model_supports_tools(model_name: &str) -> bool {
    // All Mistral models support tools
    true
}

/// Check if model supports vision at compile time  
#[inline(always)]
pub const fn model_supports_vision(model_name: &str) -> bool {
    match model_name {
        "mistral-large-latest" => true,
        "pixtral-large-latest" => true,
        "pixtral-12b-2409" => true,
        _ => false}
}

/// Check if model supports thinking at compile time
#[inline(always)]
pub const fn model_supports_thinking(model_name: &str) -> bool {
    match model_name {
        "mistral-saba-latest" => true,
        _ => false}
}

/// Check if model supports audio at compile time
#[inline(always)]
pub const fn model_supports_audio(model_name: &str) -> bool {
    // No Mistral models currently support audio
    false
}

/// Get model context length at compile time
#[inline(always)]
pub const fn get_model_context_length(model_name: &str) -> u32 {
    match model_name {
        "codestral-latest" => 32000,
        "mistral-small-latest" => 32000,
        "open-codestral-mamba" => 256000,
        _ => 128000, // Default for most models
    }
}

/// Get model max output tokens at compile time
#[inline(always)]
pub const fn get_model_max_output_tokens(model_name: &str) -> u32 {
    // All Mistral models support up to 8192 output tokens
    8192
}

/// Get optimal thinking budget for reasoning models
#[inline(always)]
pub const fn get_model_thinking_budget(model_name: &str) -> u32 {
    match model_name {
        "mistral-saba-latest" => 5000,
        _ => 0}
}

/// Get model by compile-time lookup (zero allocation)
#[inline(always)]
pub const fn is_free_model(model_name: &str) -> bool {
    match model_name {
        "mistral-small-latest" => true,
        "pixtral-12b-2409" => true,
        "open-mistral-nemo" => true,
        "open-codestral-mamba" => true,
        _ => false}
}

/// Get model pricing tier (zero allocation)
#[inline(always)]
pub const fn get_model_pricing_tier(model_name: &str) -> u8 {
    match model_name {
        // Free models (tier 0)
        "mistral-small-latest" => 0,
        "pixtral-12b-2409" => 0,
        "open-mistral-nemo" => 0,
        "open-codestral-mamba" => 0,
        // Cheap models (tier 1)
        "ministral-3b-latest" => 1,
        "ministral-8b-latest" => 1,
        // Standard models (tier 2)
        "codestral-latest" => 2,
        // Premium models (tier 3)
        "mistral-large-latest" => 3,
        "pixtral-large-latest" => 3,
        "mistral-saba-latest" => 3,
        _ => 2, // Default to standard tier
    }
}

/// Validate model name exists (zero allocation)
#[inline(always)]
pub const fn is_valid_model(model_name: &str) -> bool {
    match model_name {
        "mistral-large-latest"
        | "codestral-latest"
        | "pixtral-large-latest"
        | "mistral-saba-latest"
        | "ministral-3b-latest"
        | "ministral-8b-latest"
        | "mistral-small-latest"
        | "pixtral-12b-2409"
        | "open-mistral-nemo"
        | "open-codestral-mamba" => true,
        _ => false}
}
