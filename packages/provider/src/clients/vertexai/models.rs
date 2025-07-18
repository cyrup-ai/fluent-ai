//! Zero-allocation dynamic model registry for VertexAI
//!
//! Provides lock-free model metadata lookup, capability detection, and parameter validation
//! for all dynamically generated VertexAI models from the build system.

use crate::clients::vertexai::{VertexAIError, VertexAIResult};
use crate::models::Models;
use crate::providers::Providers;
use arrayvec::{ArrayString, ArrayVec};
use crossbeam_skiplist::SkipMap;
use cyrup_sugars::ZeroOneOrMany;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Maximum number of stop sequences per model
const MAX_STOP_SEQUENCES: usize = 8;

/// Maximum model name length
const MAX_MODEL_NAME_LEN: usize = 128;

/// Global model registry with lock-free concurrent access
static MODEL_REGISTRY: LazyLock<SkipMap<&'static str, ModelConfig>> = LazyLock::new(|| {
    let registry = SkipMap::new();
    initialize_model_registry(&registry);
    registry
});

/// Model capabilities flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Supports function/tool calling
    pub supports_tools: bool,
    
    /// Supports vision/image input
    pub supports_vision: bool,
    
    /// Supports audio input
    pub supports_audio: bool,
    
    /// Supports video input  
    pub supports_video: bool,
    
    /// Supports system messages
    pub supports_system: bool,
    
    /// Supports streaming responses
    pub supports_streaming: bool,
    
    /// Supports JSON mode
    pub supports_json_mode: bool,
    
    /// Supports reasoning/thinking
    pub supports_thinking: bool,
}

/// Model configuration with zero allocation storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name (zero allocation)
    pub name: &'static str,
    
    /// Model display name
    pub display_name: ArrayString<MAX_MODEL_NAME_LEN>,
    
    /// Model family (gemini, claude, mistral, etc.)
    pub family: &'static str,
    
    /// Model version
    pub version: ArrayString<32>,
    
    /// Model capabilities
    pub capabilities: ModelCapabilities,
    
    /// Maximum input tokens
    pub max_input_tokens: u32,
    
    /// Maximum output tokens
    pub max_output_tokens: u32,
    
    /// Context window size
    pub context_window: u32,
    
    /// Temperature range
    pub temperature_range: (f32, f32),
    
    /// Top-p range
    pub top_p_range: (f32, f32),
    
    /// Top-k range
    pub top_k_range: (u32, u32),
    
    /// Default stop sequences
    pub default_stop_sequences: ArrayVec<ArrayString<32>, MAX_STOP_SEQUENCES>,
    
    /// Cost per 1M input tokens (USD)
    pub cost_per_million_input_tokens: f64,
    
    /// Cost per 1M output tokens (USD)
    pub cost_per_million_output_tokens: f64,
    
    /// Rate limits (requests per minute)
    pub rate_limit_rpm: u32,
    
    /// Rate limits (tokens per minute)
    pub rate_limit_tpm: u32,
    
    /// Model endpoint path template
    pub endpoint_template: &'static str,
    
    /// Whether model requires max_tokens parameter
    pub requires_max_tokens: bool,
    
    /// Whether model is deprecated
    pub deprecated: bool,
    
    /// Recommended thinking budget for reasoning models
    pub optimal_thinking_budget: Option<u32>,
}

/// VertexAI models registry and utilities
pub struct VertexAIModels;

impl VertexAIModels {
    /// Get all VertexAI models dynamically from the auto-generated enum
    pub fn all_models() -> ZeroOneOrMany<Models> {
        Providers::Vertexai.models()
    }
    
    /// Get model configuration by name with zero allocation lookup
    pub fn get_config(model_name: &str) -> Option<&'static ModelConfig> {
        MODEL_REGISTRY.get(model_name).map(|entry| entry.value())
    }
    
    /// Validate model name exists in VertexAI
    pub fn validate_model(model_name: &str) -> VertexAIResult<()> {
        if Self::get_config(model_name).is_none() {
            return Err(VertexAIError::ModelNotFound {
                model: model_name.to_string(),
            });
        }
        Ok(())
    }
    
    /// Check if model supports specific capability
    pub fn supports_capability(model_name: &str, capability: &str) -> VertexAIResult<bool> {
        let config = Self::get_config(model_name).ok_or_else(|| VertexAIError::ModelNotFound {
            model: model_name.to_string(),
        })?;
        
        let supports = match capability {
            "tools" | "function_calling" => config.capabilities.supports_tools,
            "vision" | "images" => config.capabilities.supports_vision,
            "audio" => config.capabilities.supports_audio,
            "video" => config.capabilities.supports_video,
            "system" | "system_messages" => config.capabilities.supports_system,
            "streaming" => config.capabilities.supports_streaming,
            "json_mode" => config.capabilities.supports_json_mode,
            "thinking" | "reasoning" => config.capabilities.supports_thinking,
            _ => false,
        };
        
        Ok(supports)
    }
    
    /// Validate generation parameters for model
    pub fn validate_parameters(
        model_name: &str,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        max_tokens: Option<u32>,
    ) -> VertexAIResult<()> {
        let config = Self::get_config(model_name).ok_or_else(|| VertexAIError::ModelNotFound {
            model: model_name.to_string(),
        })?;
        
        // Validate temperature
        if let Some(temp) = temperature {
            if temp < config.temperature_range.0 || temp > config.temperature_range.1 {
                return Err(VertexAIError::RequestValidation {
                    field: "temperature".to_string(),
                    reason: format!(
                        "Temperature {} out of range [{}, {}]",
                        temp, config.temperature_range.0, config.temperature_range.1
                    ),
                });
            }
        }
        
        // Validate top_p
        if let Some(p) = top_p {
            if p < config.top_p_range.0 || p > config.top_p_range.1 {
                return Err(VertexAIError::RequestValidation {
                    field: "top_p".to_string(),
                    reason: format!(
                        "Top-p {} out of range [{}, {}]",
                        p, config.top_p_range.0, config.top_p_range.1
                    ),
                });
            }
        }
        
        // Validate top_k
        if let Some(k) = top_k {
            if k < config.top_k_range.0 || k > config.top_k_range.1 {
                return Err(VertexAIError::RequestValidation {
                    field: "top_k".to_string(),
                    reason: format!(
                        "Top-k {} out of range [{}, {}]",
                        k, config.top_k_range.0, config.top_k_range.1
                    ),
                });
            }
        }
        
        // Validate max_tokens
        if let Some(tokens) = max_tokens {
            if tokens > config.max_output_tokens {
                return Err(VertexAIError::RequestValidation {
                    field: "max_tokens".to_string(),
                    reason: format!(
                        "Max tokens {} exceeds model limit {}",
                        tokens, config.max_output_tokens
                    ),
                });
            }
        } else if config.requires_max_tokens {
            return Err(VertexAIError::RequestValidation {
                field: "max_tokens".to_string(),
                reason: "Max tokens parameter is required for this model".to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Get endpoint URL for model
    pub fn get_endpoint_url(model_name: &str, project_id: &str, region: &str) -> VertexAIResult<String> {
        let config = Self::get_config(model_name).ok_or_else(|| VertexAIError::ModelNotFound {
            model: model_name.to_string(),
        })?;
        
        let url = config.endpoint_template
            .replace("{project}", project_id)
            .replace("{region}", region)
            .replace("{model}", model_name);
        
        Ok(url)
    }
    
    /// Estimate cost for request
    pub fn estimate_cost(
        model_name: &str,
        input_tokens: u32,
        output_tokens: u32,
    ) -> VertexAIResult<f64> {
        let config = Self::get_config(model_name).ok_or_else(|| VertexAIError::ModelNotFound {
            model: model_name.to_string(),
        })?;
        
        let input_cost = (input_tokens as f64 / 1_000_000.0) * config.cost_per_million_input_tokens;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * config.cost_per_million_output_tokens;
        
        Ok(input_cost + output_cost)
    }
    
    /// Get models by family
    pub fn models_by_family(family: &str) -> Vec<&'static str> {
        MODEL_REGISTRY
            .iter()
            .filter(|entry| entry.value().family == family)
            .map(|entry| *entry.key())
            .collect()
    }
    
    /// Get registry statistics
    pub fn registry_stats() -> (usize, usize) {
        let total_models = MODEL_REGISTRY.len();
        let active_models = MODEL_REGISTRY
            .iter()
            .filter(|entry| !entry.value().deprecated)
            .count();
        
        (total_models, active_models)
    }
}

/// Initialize model registry with dynamic model detection
fn initialize_model_registry(registry: &SkipMap<&'static str, ModelConfig>) {
    // Get all VertexAI models from the auto-generated enum
    let models = Providers::Vertexai.models();
    
    match models {
        ZeroOneOrMany::Zero => {
            // No models available
        }
        ZeroOneOrMany::One(model) => {
            add_model_config(registry, model);
        }
        ZeroOneOrMany::Many(model_list) => {
            for model in model_list {
                add_model_config(registry, model);
            }
        }
    }
}

/// Add model configuration to registry
fn add_model_config(registry: &SkipMap<&'static str, ModelConfig>, model: Models) {
    let model_name = model.name();
    let model_info = model.info();
    
    // Determine model family from name
    let family = determine_model_family(model_name);
    
    // Create model configuration based on detected family and capabilities
    let config = match family {
        "gemini" => create_gemini_config(model_name, &model_info),
        "claude" => create_claude_config(model_name, &model_info),
        "mistral" => create_mistral_config(model_name, &model_info),
        "embedding" => create_embedding_config(model_name, &model_info),
        _ => create_default_config(model_name, &model_info),
    };
    
    registry.insert(model_name, config);
}

/// Determine model family from name
fn determine_model_family(model_name: &str) -> &'static str {
    let name_lower = model_name.to_lowercase();
    
    if name_lower.contains("gemini") || name_lower.contains("gemma") {
        "gemini"
    } else if name_lower.contains("claude") {
        "claude"
    } else if name_lower.contains("mistral") || name_lower.contains("codestral") {
        "mistral"
    } else if name_lower.contains("embedding") {
        "embedding"
    } else {
        "unknown"
    }
}

/// Create Gemini model configuration
fn create_gemini_config(model_name: &str, model_info: &crate::model_info::ModelInfoData) -> ModelConfig {
    let capabilities = ModelCapabilities {
        supports_tools: model_info.supports_function_calling.unwrap_or(true),
        supports_vision: model_info.supports_vision.unwrap_or(true),
        supports_audio: false,
        supports_video: true,
        supports_system: true,
        supports_streaming: true,
        supports_json_mode: true,
        supports_thinking: model_info.supports_thinking.unwrap_or(false),
    };
    
    ModelConfig {
        name: model_name,
        display_name: ArrayString::from(model_name).unwrap_or_default(),
        family: "gemini",
        version: ArrayString::from("1.0").unwrap_or_default(),
        capabilities,
        max_input_tokens: model_info.max_input_tokens.unwrap_or(1_000_000) as u32,
        max_output_tokens: model_info.max_output_tokens.unwrap_or(8192) as u32,
        context_window: model_info.max_input_tokens.unwrap_or(1_000_000) as u32,
        temperature_range: (0.0, 2.0),
        top_p_range: (0.0, 1.0),
        top_k_range: (1, 40),
        default_stop_sequences: ArrayVec::new(),
        cost_per_million_input_tokens: model_info.input_price.unwrap_or(0.00125),
        cost_per_million_output_tokens: model_info.output_price.unwrap_or(0.005),
        rate_limit_rpm: 300,
        rate_limit_tpm: 300_000,
        endpoint_template: "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:generateContent",
        requires_max_tokens: false,
        deprecated: false,
        optimal_thinking_budget: model_info.optimal_thinking_budget,
    }
}

/// Create Claude model configuration
fn create_claude_config(model_name: &str, model_info: &crate::model_info::ModelInfoData) -> ModelConfig {
    let capabilities = ModelCapabilities {
        supports_tools: model_info.supports_function_calling.unwrap_or(true),
        supports_vision: model_info.supports_vision.unwrap_or(true),
        supports_audio: false,
        supports_video: false,
        supports_system: true,
        supports_streaming: true,
        supports_json_mode: false,
        supports_thinking: model_info.supports_thinking.unwrap_or(false),
    };
    
    ModelConfig {
        name: model_name,
        display_name: ArrayString::from(model_name).unwrap_or_default(),
        family: "claude",
        version: ArrayString::from("3.0").unwrap_or_default(),
        capabilities,
        max_input_tokens: model_info.max_input_tokens.unwrap_or(200_000) as u32,
        max_output_tokens: model_info.max_output_tokens.unwrap_or(8192) as u32,
        context_window: model_info.max_input_tokens.unwrap_or(200_000) as u32,
        temperature_range: (0.0, 1.0),
        top_p_range: (0.0, 1.0),
        top_k_range: (1, 500),
        default_stop_sequences: ArrayVec::new(),
        cost_per_million_input_tokens: model_info.input_price.unwrap_or(0.003),
        cost_per_million_output_tokens: model_info.output_price.unwrap_or(0.015),
        rate_limit_rpm: 60,
        rate_limit_tpm: 40_000,
        endpoint_template: "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:generateContent",
        requires_max_tokens: true,
        deprecated: false,
        optimal_thinking_budget: model_info.optimal_thinking_budget,
    }
}

/// Create Mistral model configuration
fn create_mistral_config(model_name: &str, model_info: &crate::model_info::ModelInfoData) -> ModelConfig {
    let capabilities = ModelCapabilities {
        supports_tools: model_info.supports_function_calling.unwrap_or(true),
        supports_vision: false,
        supports_audio: false,
        supports_video: false,
        supports_system: true,
        supports_streaming: true,
        supports_json_mode: true,
        supports_thinking: false,
    };
    
    ModelConfig {
        name: model_name,
        display_name: ArrayString::from(model_name).unwrap_or_default(),
        family: "mistral",
        version: ArrayString::from("1.0").unwrap_or_default(),
        capabilities,
        max_input_tokens: model_info.max_input_tokens.unwrap_or(128_000) as u32,
        max_output_tokens: model_info.max_output_tokens.unwrap_or(8192) as u32,
        context_window: model_info.max_input_tokens.unwrap_or(128_000) as u32,
        temperature_range: (0.0, 1.5),
        top_p_range: (0.0, 1.0),
        top_k_range: (1, 50),
        default_stop_sequences: ArrayVec::new(),
        cost_per_million_input_tokens: model_info.input_price.unwrap_or(0.0025),
        cost_per_million_output_tokens: model_info.output_price.unwrap_or(0.0075),
        rate_limit_rpm: 120,
        rate_limit_tpm: 120_000,
        endpoint_template: "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/mistralai/models/{model}:generateContent",
        requires_max_tokens: false,
        deprecated: false,
        optimal_thinking_budget: None,
    }
}

/// Create embedding model configuration
fn create_embedding_config(model_name: &str, model_info: &crate::model_info::ModelInfoData) -> ModelConfig {
    let capabilities = ModelCapabilities {
        supports_tools: false,
        supports_vision: false,
        supports_audio: false,
        supports_video: false,
        supports_system: false,
        supports_streaming: false,
        supports_json_mode: false,
        supports_thinking: false,
    };
    
    ModelConfig {
        name: model_name,
        display_name: ArrayString::from(model_name).unwrap_or_default(),
        family: "embedding",
        version: ArrayString::from("1.0").unwrap_or_default(),
        capabilities,
        max_input_tokens: model_info.max_input_tokens.unwrap_or(3072) as u32,
        max_output_tokens: 0,
        context_window: model_info.max_input_tokens.unwrap_or(3072) as u32,
        temperature_range: (0.0, 0.0),
        top_p_range: (0.0, 0.0),
        top_k_range: (0, 0),
        default_stop_sequences: ArrayVec::new(),
        cost_per_million_input_tokens: model_info.input_price.unwrap_or(0.00002),
        cost_per_million_output_tokens: 0.0,
        rate_limit_rpm: 300,
        rate_limit_tpm: 1_000_000,
        endpoint_template: "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:predict",
        requires_max_tokens: false,
        deprecated: false,
        optimal_thinking_budget: None,
    }
}

/// Create default model configuration
fn create_default_config(model_name: &str, model_info: &crate::model_info::ModelInfoData) -> ModelConfig {
    let capabilities = ModelCapabilities {
        supports_tools: model_info.supports_function_calling.unwrap_or(false),
        supports_vision: model_info.supports_vision.unwrap_or(false),
        supports_audio: false,
        supports_video: false,
        supports_system: true,
        supports_streaming: true,
        supports_json_mode: false,
        supports_thinking: model_info.supports_thinking.unwrap_or(false),
    };
    
    ModelConfig {
        name: model_name,
        display_name: ArrayString::from(model_name).unwrap_or_default(),
        family: "unknown",
        version: ArrayString::from("1.0").unwrap_or_default(),
        capabilities,
        max_input_tokens: model_info.max_input_tokens.unwrap_or(4096) as u32,
        max_output_tokens: model_info.max_output_tokens.unwrap_or(4096) as u32,
        context_window: model_info.max_input_tokens.unwrap_or(4096) as u32,
        temperature_range: (0.0, 1.0),
        top_p_range: (0.0, 1.0),
        top_k_range: (1, 40),
        default_stop_sequences: ArrayVec::new(),
        cost_per_million_input_tokens: model_info.input_price.unwrap_or(0.001),
        cost_per_million_output_tokens: model_info.output_price.unwrap_or(0.002),
        rate_limit_rpm: 60,
        rate_limit_tpm: 60_000,
        endpoint_template: "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{model}:predict",
        requires_max_tokens: false,
        deprecated: false,
        optimal_thinking_budget: model_info.optimal_thinking_budget,
    }
}