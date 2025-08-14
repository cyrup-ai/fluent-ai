//! Model information re-exports from model-info crate
//!
//! This module provides a bridge to the model-info crate, re-exporting
//! the core types for backward compatibility.

pub use model_info::{
    Model, ModelCapabilities, ModelError, ModelInfo, ModelInfoBuilder, ProviderModels,
    ProviderTrait, Result,
};
