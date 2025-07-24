//! Pure YAML model information structs
//!
//! This module provides simple data structures that mirror the YAML structure exactly.
//! Zero transformations, zero domain dependencies - just raw YAML data containers.
//!
//! Performance characteristics:
//! - Blazing-fast deserialization with yyaml
//! - Zero allocation constructors where possible
//! - Elegant ergonomic field access

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pure YAML provider data - mirrors YAML sequence structure exactly
/// 
/// Each provider contains a name and list of models.
/// This struct deserializes directly from yyaml with zero transformations.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct YamlProvider {
    pub provider: String,
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
    pub name: String,
    #[serde(default)]
    pub max_input_tokens: Option<u64>,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub input_price: Option<f64>,
    #[serde(default)]
    pub output_price: Option<f64>,
    #[serde(default)]
    pub supports_vision: Option<bool>,
    #[serde(default)]
    pub supports_function_calling: Option<bool>,
    #[serde(default)]
    pub require_max_tokens: Option<bool>,
    #[serde(default)]
    pub real_name: Option<String>,
    #[serde(default)]
    pub system_prompt_prefix: Option<String>,
    #[serde(default, rename = "type")]
    pub model_type: Option<String>,
    #[serde(default)]
    pub budget_tokens: Option<u64>,
    #[serde(default)]
    pub max_tokens_per_chunk: Option<u64>,
    #[serde(default)]
    pub default_chunk_size: Option<u64>,
    #[serde(default)]
    pub max_batch_size: Option<u64>,
    #[serde(default)]
    pub include_reasoning: Option<bool>,
    #[serde(default)]
    pub patch: Option<serde_json::Value>,
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