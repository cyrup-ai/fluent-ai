//! Embedding System Module
//!
//! This module provides a consolidated embedding system with sparse file separation
//! for singular concerns. Each submodule handles a specific aspect of embedding functionality.

/// Core embedding types, traits, and domain objects
pub mod core;

/// Embedding configuration types and builder patterns
pub mod config;

/// Usage tracking and token counting for embedding operations
pub mod usage;

/// Embedding service implementations and caching mechanisms
pub mod service;

// Re-export core types from core module
// Legacy compatibility re-exports
pub use core::EmbeddingModel as EmbeddingModelTrait;
pub use core::{Embedding, EmbeddingData, EmbeddingModel, EmbeddingResponse};

pub use config::EmbeddingConfig as EmbeddingConfiguration;
// Re-export configuration types
pub use config::{EmbeddingConfig, IntoEmbeddingConfig};
// Re-export service types
pub use service::{EmbeddingPool, EmbeddingService, InMemoryEmbeddingCache};
// Re-export usage tracking types
pub use usage::{EmbeddingUsage, TokenUsage};
