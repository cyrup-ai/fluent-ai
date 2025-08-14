/// Cache configuration and management
pub mod cache;
/// Database connection and storage configuration
pub mod database;
/// Large Language Model provider configuration
pub mod llm;
/// Memory system configuration and settings
pub mod memory;
/// Shared configuration types used across modules
pub mod shared;
/// Vector store configuration for embeddings
pub mod vector;

// Re-export specific types to avoid ambiguous glob re-exports
pub use cache::{
    get_cached_config, get_pool_stats, get_pooled_memory, return_pooled_memory, update_config_cache,
};
pub use database::DatabaseConfig;
pub use llm::{LLMConfig, LLMConfigError, LLMProvider};
pub use memory::MemoryConfig;
pub use shared::{EmbeddingConfig, RetryConfig};
pub use vector::VectorStoreConfig;
