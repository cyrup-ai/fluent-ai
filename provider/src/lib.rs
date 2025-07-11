//! Fluent AI Provider Library
//!
//! This crate provides provider and model traits and definitions for AI services.
//! The enum variants are auto-generated from the AiChat models.yaml file.

use serde::{Deserialize, Serialize};

/// Information about a specific model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfoData {
    pub provider_name: String,
    pub name: String,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub input_price: Option<f64>,
    pub output_price: Option<f64>,
    pub supports_vision: Option<bool>,
    pub supports_function_calling: Option<bool>,
    pub require_max_tokens: Option<bool>,
}

/// Trait for AI providers
pub trait Provider: std::fmt::Debug + Send + Sync {
    /// Get all models supported by this provider
    fn models(&self) -> Vec<Box<dyn Model>>;

    /// Get the provider name
    fn name(&self) -> &str;
}

/// Trait for AI models
pub trait Model: std::fmt::Debug + Send + Sync {
    /// Get detailed information about this model
    fn info(&self) -> ModelInfoData;

    /// Get the original model name
    fn name(&self) -> &str;
}

/// Trait for model information
pub trait ModelInfo: std::fmt::Debug + Send + Sync {
    /// Get the model information data
    fn data(&self) -> ModelInfoData;
}

pub mod model;
pub mod provider;

// Re-export generated implementations for convenience
pub use model::Models;
pub use provider::Providers;
