//! Model configuration types and enumerations
//!
//! This module provides core type definitions organized by concept:
//! - Model architecture types
//! - Quantization options  
//! - Model configuration structures
//! - Default values for different architectures

pub mod model_config;
pub mod model_config_methods;
pub mod model_type;
pub mod quantization_type;

// Re-export all types for ergonomic access
pub use model_config::ModelConfig;
pub use model_type::ModelType;
pub use quantization_type::QuantizationType;