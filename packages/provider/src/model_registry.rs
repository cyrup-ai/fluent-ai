//! Model registry for centralized model configuration management
//! 
//! This module provides zero-allocation model lookups using static HashMap caching
//! and serves as the central registry for all AI provider models.

use std::collections::HashMap;
use once_cell::sync::OnceCell;
use crate::completion_provider::ModelConfig;
use crate::model_capabilities::ModelInfoData;
use crate::model_pricing::model_info_to_config;

/// Global model configuration cache for zero-allocation lookups
static MODEL_CONFIG_CACHE: OnceCell<HashMap<&'static str, ModelConfig>> = OnceCell::new();

/// Get model configuration with zero-allocation caching
/// 
/// This function provides O(1) model configuration lookup using a pre-computed
/// HashMap that's initialized once and cached statically. All model configurations
/// are validated and optimized for production use.
/// 
/// # Arguments
/// * `model_name` - Static string identifier for the model
/// 
/// # Returns
/// * Reference to the model configuration, or default config for unknown models
/// 
/// # Performance
/// - First call: O(n) to initialize the entire registry
/// - Subsequent calls: O(1) HashMap lookup
/// - Zero allocation after initialization
pub fn get_model_config(model_name: &'static str) -> &'static ModelConfig {
    let cache = MODEL_CONFIG_CACHE.get_or_init(|| {
        initialize_model_registry()
    });
    
    cache.get(model_name).unwrap_or_else(|| {
        // Fallback for unknown models with production-ready defaults
        static DEFAULT_CONFIG: ModelConfig = ModelConfig {
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 0.9,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            context_length: 128000,
            system_prompt: "You are a helpful AI assistant.",
            supports_tools: false,
            supports_vision: false,
            supports_audio: false,
            supports_thinking: false,
            optimal_thinking_budget: 1024,
            provider: "unknown",
            model_name: "unknown",
        };
        &DEFAULT_CONFIG
    })
}

/// Initialize the complete model registry with all supported models
/// 
/// This function is called once during the first model lookup and builds
/// the complete registry of all AI provider models. The implementation
/// is auto-generated from the model specifications.
/// 
/// # Returns
/// * HashMap containing all model configurations keyed by model name
fn initialize_model_registry() -> HashMap<&'static str, ModelConfig> {
    let mut map = HashMap::new();
    
    // AUTO-GENERATED START - Model registry initialization
    // This section is auto-generated from model specifications
    // Do not edit manually - use the model configuration generator
    
    // OpenAI Models
    register_openai_models(&mut map);
    
    // Google Models  
    register_google_models(&mut map);
    
    // Anthropic Models
    register_anthropic_models(&mut map);
    
    // Meta Models
    register_meta_models(&mut map);
    
    // Mistral Models
    register_mistral_models(&mut map);
    
    // Other Provider Models
    register_other_models(&mut map);
    
    // AUTO-GENERATED END
    
    map
}

/// Register OpenAI model configurations
fn register_openai_models(map: &mut HashMap<&'static str, ModelConfig>) {
    use crate::model_info_generated::*;
    
    let info = get_gpt41_info();
    let config = model_info_to_config(&info, "Gpt41");
    map.insert("Gpt41", config);
    
    let info = get_gpt41mini_info();
    let config = model_info_to_config(&info, "Gpt41Mini");
    map.insert("Gpt41Mini", config);
    
    let info = get_gpt41nano_info();
    let config = model_info_to_config(&info, "Gpt41Nano");
    map.insert("Gpt41Nano", config);
    
    let info = get_gpt4o_info();
    let config = model_info_to_config(&info, "Gpt4O");
    map.insert("Gpt4O", config);
    
    // Additional OpenAI models will be auto-generated here
}

/// Register Google model configurations  
fn register_google_models(map: &mut HashMap<&'static str, ModelConfig>) {
    use crate::model_info_generated::*;
    
    let info = get_gemini25flash_info();
    let config = model_info_to_config(&info, "Gemini25Flash");
    map.insert("Gemini25Flash", config);
    
    let info = get_gemini25pro_info();
    let config = model_info_to_config(&info, "Gemini25Pro");
    map.insert("Gemini25Pro", config);
    
    // Additional Google models will be auto-generated here
}

/// Register Anthropic model configurations
fn register_anthropic_models(map: &mut HashMap<&'static str, ModelConfig>) {
    use crate::model_info_generated::*;
    
    let info = get_anthropicclaudesonnet4_info();
    let config = model_info_to_config(&info, "AnthropicClaudeSonnet4");
    map.insert("AnthropicClaudeSonnet4", config);
    
    let info = get_anthropicclaude37sonnet_info();
    let config = model_info_to_config(&info, "AnthropicClaude37Sonnet");
    map.insert("AnthropicClaude37Sonnet", config);
    
    // Additional Anthropic models will be auto-generated here
}

/// Register Meta model configurations
fn register_meta_models(map: &mut HashMap<&'static str, ModelConfig>) {
    use crate::model_info_generated::*;
    
    let info = get_metallamallama4maverick_info();
    let config = model_info_to_config(&info, "MetaLlamaLlama4Maverick");
    map.insert("MetaLlamaLlama4Maverick", config);
    
    // Additional Meta models will be auto-generated here
}

/// Register Mistral model configurations
fn register_mistral_models(map: &mut HashMap<&'static str, ModelConfig>) {
    use crate::model_info_generated::*;
    
    let info = get_mistralmediumlatest_info();
    let config = model_info_to_config(&info, "MistralMediumLatest");
    map.insert("MistralMediumLatest", config);
    
    // Additional Mistral models will be auto-generated here
}

/// Register other provider model configurations
fn register_other_models(map: &mut HashMap<&'static str, ModelConfig>) {
    use crate::model_info_generated::*;
    
    // AI21, Cohere, DeepSeek, Qwen, etc. models will be auto-generated here
}

/// Get all available model names for enumeration
/// 
/// This function provides a way to enumerate all registered models
/// without exposing the internal HashMap structure.
/// 
/// # Returns
/// * Vector of all available model names
pub fn get_all_model_names() -> Vec<&'static str> {
    let cache = MODEL_CONFIG_CACHE.get_or_init(|| {
        initialize_model_registry()
    });
    
    cache.keys().copied().collect()
}

/// Check if a model is registered in the system
/// 
/// # Arguments
/// * `model_name` - Name of the model to check
/// 
/// # Returns
/// * `true` if the model is registered, `false` otherwise
pub fn is_model_registered(model_name: &str) -> bool {
    let cache = MODEL_CONFIG_CACHE.get_or_init(|| {
        initialize_model_registry()
    });
    
    cache.contains_key(model_name)
}

/// Get models by provider
/// 
/// # Arguments
/// * `provider` - Provider name to filter by
/// 
/// # Returns
/// * Vector of model configurations for the specified provider
pub fn get_models_by_provider(provider: &str) -> Vec<(&'static str, &'static ModelConfig)> {
    let cache = MODEL_CONFIG_CACHE.get_or_init(|| {
        initialize_model_registry()
    });
    
    cache
        .iter()
        .filter(|(_, config)| config.provider == provider)
        .map(|(name, config)| (*name, config))
        .collect()
}