//! Embedding providers module - zero-allocation, lock-free implementations
//! for various embedding services.

#![deny(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications,
    clippy::unwrap_used,
    clippy::expect_used
)]

mod cache;
mod circuit_breaker;
mod cohere;
mod cognitive;
mod config;
mod error;
mod metrics;
mod openai;
mod storage;
mod traits;

// Re-export public API
pub use {
    cache::MultiLayerCache,
    circuit_breaker::CognitiveCircuitBreaker,
    cohere::CohereEmbeddingProvider,
    cognitive::CognitiveEmbeddingProvider,
    config::{CacheConfig, CircuitBreakerConfig, EmbeddingConfig},
    error::EmbeddingError,
    metrics::{CacheMetrics, CognitiveMetrics},
    openai::OpenAIEmbeddingProvider,
    storage::{HNSWIndexTrait, SurrealDBStorageTrait},
    traits::{CognitiveMemoryManagerTrait, EnhancedEmbeddingModel, QueryIntent},
};

/// Re-export common types for convenience
pub mod prelude {
    pub use super::{
        cache::MultiLayerCache,
        circuit_breaker::CognitiveCircuitBreaker,
        cohere::CohereEmbeddingProvider,
        cognitive::CognitiveEmbeddingProvider,
        config::{CacheConfig, CircuitBreakerConfig, EmbeddingConfig},
        error::EmbeddingError,
        metrics::{CacheMetrics, CognitiveMetrics},
        openai::OpenAIEmbeddingProvider,
        storage::{HNSWIndexTrait, SurrealDBStorageTrait},
        traits::{CognitiveMemoryManagerTrait, EnhancedEmbeddingModel, QueryIntent},
    };
}

#[cfg(test)]
mod tests {
    // Test modules will be in the tests/ directory
}