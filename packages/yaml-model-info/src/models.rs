//! Pure YAML model information structs
//!
//! This module provides simple data structures that mirror the YAML structure exactly.
//! Zero transformations, zero domain dependencies - just raw YAML data containers.
//!
//! Performance characteristics:
//! - Blazing-fast deserialization with yyaml
//! - Zero allocation constructors where possible
//! - Elegant ergonomic field access

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
/// Pure YAML provider data - mirrors YAML sequence structure exactly
///
/// Each provider contains a name and list of models.
/// This struct deserializes directly from yyaml with zero transformations.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct YamlProvider {
    /// Provider name identifier (e.g., "openai", "anthropic", "gemini")
    pub provider: String,
    /// List of models available from this provider
    pub models: Vec<YamlModel>,
}

impl YamlProvider {
    /// Get provider identifier - zero allocation when possible
    #[inline(always)]
    pub fn identifier(&self) -> &str {
        &self.provider
    }

    /// Get model count for this provider
    #[inline(always)]
    pub const fn model_count(&self) -> usize {
        self.models.len()
    }
}

/// Pure YAML model structure - mirrors YAML model structure exactly
///
/// All fields match the YAML structure with proper Option types for optional fields.
/// Optimized for blazing-fast deserialization and zero allocation access.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct YamlModel {
    /// Model identifier name as specified by the provider
    pub name: String,
    /// Maximum number of input tokens this model can process
    #[serde(default)]
    pub max_input_tokens: Option<u64>,
    /// Maximum number of tokens this model can generate in response
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    /// Cost per input token in USD (for pricing calculations)
    #[serde(default)]
    pub input_price: Option<f64>,
    /// Cost per output token in USD (for pricing calculations)
    #[serde(default)]
    pub output_price: Option<f64>,
    /// Whether this model supports vision/image understanding capabilities
    #[serde(default)]
    pub supports_vision: Option<bool>,
    /// Whether this model supports function/tool calling features
    #[serde(default)]
    pub supports_function_calling: Option<bool>,
    /// Whether this model requires max_tokens parameter to be specified
    #[serde(default)]
    pub require_max_tokens: Option<bool>,
    /// Alternative or canonical name for the model (if different from name)
    #[serde(default)]
    pub real_name: Option<String>,
    /// System prompt prefix that should be prepended to all conversations
    #[serde(default)]
    pub system_prompt_prefix: Option<String>,
    /// Model type classification (e.g., "chat", "completion", "embedding")
    #[serde(default, rename = "type")]
    pub model_type: Option<String>,
    /// Token budget limit for cost-controlled operations
    #[serde(default)]
    pub budget_tokens: Option<u64>,
    /// Maximum tokens per processing chunk for streaming operations
    #[serde(default)]
    pub max_tokens_per_chunk: Option<u64>,
    /// Default chunk size for batch processing operations
    #[serde(default)]
    pub default_chunk_size: Option<u64>,
    /// Maximum batch size for concurrent request processing
    #[serde(default)]
    pub max_batch_size: Option<u64>,
    /// Whether to include reasoning steps in model responses
    #[serde(default)]
    pub include_reasoning: Option<bool>,
    /// Custom patch data for model-specific configuration overrides
    #[serde(default)]
    pub patch: Option<serde_json::Value>,
    /// Additional fields not explicitly defined in the schema
    #[serde(flatten)]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl YamlModel {
    /// Create full identifier for this model - zero allocation when possible
    #[inline(always)]
    pub fn identifier(&self, provider: &str) -> String {
        format!("{}:{}", provider, self.name)
    }

    /// Check if model supports specific capability - zero allocation
    #[inline(always)]
    pub fn supports_capability(&self, capability: &str) -> bool {
        match capability {
            "vision" => self.supports_vision.unwrap_or(false),
            "function_calling" => self.supports_function_calling.unwrap_or(false),
            "chat" => true, // All models support basic chat
            _ => {
                // Check extra fields for custom capabilities
                self.extra_fields.contains_key(capability)
            }
        }
    }

    /// Get input cost per token - const fn for compile-time optimization
    #[inline(always)]
    pub const fn input_cost(&self) -> Option<f64> {
        self.input_price
    }

    /// Get output cost per token - const fn for compile-time optimization
    #[inline(always)]
    pub const fn output_cost(&self) -> Option<f64> {
        self.output_price
    }
}
