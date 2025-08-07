//! Memory System Module
//!
//! Unified high-performance memory system with zero-allocation patterns,
//! quantum-inspired cognitive computing, and SIMD-optimized operations.

/// Memory primitives with zero-allocation design
pub mod primitives;

/// Quantum-inspired cognitive computing types
mod cognitive;

/// High-performance configuration system
pub mod config;

// Re-export configuration types
pub use config::llm::{LLMConfig, LLMProvider};
// Re-export manager types
pub use manager::{MemoryManagerTrait as MemoryManager, SurrealDBMemoryManager};

/// Core memory management and configuration
pub mod manager;

/// SIMD-optimized vector operations for high-performance memory processing
mod ops;

/// Memory tool implementation for MCP integration
mod tool;

/// Cache implementation
mod cache;

/// Memory pool implementation
mod pool;

/// Memory serialization utilities
mod serialization;

/// Memory workflow management - removed fake implementation

/// Memory trait definitions for trait-backed architecture
pub mod traits;

// Re-export all new domain types
// Type aliases for migration compatibility

/// Compatibility mode for memory systems
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
pub enum CompatibilityMode {
    /// Strict mode: Only allow exact matches
    Strict,
    /// Flexible mode: Allow best-effort conversions  
    #[default]
    Flexible,
    /// Hybrid mode: Support both modern and transitional types simultaneously
    Hybrid}

// Re-export specific types to avoid ambiguous glob re-exports
pub use cognitive::{CognitiveMemory, CognitiveProcessor};
pub use config::database::{DatabaseType, PoolConfig};
pub use config::shared::RetryConfig;
pub use config::shared::{EmbeddingConfig, EmbeddingModelType};
pub use config::vector::{
    DistanceMetric, IndexConfig, IndexType, PerformanceConfig, SimdConfig, VectorStoreType};
pub use config::{DatabaseConfig, MemoryConfig, VectorStoreConfig};
// Conditional re-exports for cognitive features
// Removed unexpected cfg condition "cognitive" - feature does not exist
// Re-export fluent_ai_memory types for convenience
// Removed unexpected cfg condition "fluent-ai-memory" - feature does not exist
// Re-export core types for backward compatibility
pub use manager::Memory;
pub use ops::{
    CpuArchitecture, CpuFeatures, EMBEDDING_DIMENSION, Op, SIMD_WIDTH, SMALL_EMBEDDING_DIMENSION};
pub use primitives::*;
// Re-export commonly used primitives types
pub use primitives::{MemoryContent, MemoryTypeEnum};
pub use tool::{MemoryOperation, MemoryResult, MemoryTool, MemoryToolError, MemoryToolResult};
// Re-export trait types for trait-backed architecture
pub use traits::{CandleMemory, CandleMemoryStats, CandleMemoryImpl, MockCandleMemory};

// BoxFuture replaced with AsyncStream - use .collect() for Future-like behavior

/// Fallback trait definition (removed unexpected cfg condition "fluent-ai-memory")
pub trait MemoryManagerTrait: Send + Sync {
    type Error;
    type MemoryNode;

    fn store_memory(&self, memory: Self::MemoryNode) -> Result<(), Self::Error>;
}

// Trait is already exported as public above

// Primary error type is now the new MemoryError from primitives
pub type Error = primitives::MemoryError;

// Compatibility aliases
pub type VectorStoreError = Error;
// MemoryError alias removed to avoid conflict with fluent_ai_memory::Error

/// Memory system configuration combining all subsystem configurations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemorySystemConfig {
    /// Database configuration for persistent storage
    pub database: DatabaseConfig,
    /// Vector store configuration for embeddings and similarity search
    pub vector_store: VectorStoreConfig,
    /// LLM configuration for AI operations like summarization and querying
    pub llm: LLMConfig,
    /// Enable cognitive features for advanced memory processing
    pub enable_cognitive: bool,
    /// Compatibility mode for transitional systems migration
    pub compatibility_mode: CompatibilityMode}

impl MemorySystemConfig {
    /// Create optimized configuration for production use
    pub fn optimized() -> primitives::MemoryResult<Self> {
        Ok(Self {
            database: DatabaseConfig::default(),
            vector_store: VectorStoreConfig::default(),
            llm: LLMConfig::default(),
            enable_cognitive: true,
            compatibility_mode: CompatibilityMode::Hybrid})
    }

    /// Create minimal configuration for testing
    pub fn minimal() -> primitives::MemoryResult<Self> {
        Ok(Self {
            database: DatabaseConfig::default(),
            vector_store: VectorStoreConfig::new(
                VectorStoreType::Memory,
                EmbeddingConfig::default(),
                768,
            )?,
            llm: LLMConfig::new(LLMProvider::OpenAI, "gpt-4")?,
            enable_cognitive: false,
            compatibility_mode: CompatibilityMode::Strict})
    }

    /// Validate configuration consistency
    pub fn validate(&self) -> primitives::MemoryResult<()> {
        self.vector_store.validate()?;
        self.llm.validate()?;
        Ok(())
    }
}

impl Default for MemorySystemConfig {
    fn default() -> Self {
        Self::optimized().expect("Default memory system configuration should be valid")
    }
}


/// Convenience functions for creating memory system configurations
impl MemorySystemConfig {

    /// Create configuration optimized for semantic search
    pub fn for_semantic_search() -> primitives::MemoryResult<Self> {
        Ok(Self {
            database: DatabaseConfig::default(),
            vector_store: VectorStoreConfig::new(
                VectorStoreType::FAISS,
                EmbeddingConfig::high_performance(),
                3072,
            )?
            .with_distance_metric(DistanceMetric::Cosine)
            .with_simd_config(SimdConfig::optimized()),
            llm: LLMConfig::default(),
            enable_cognitive: true,
            compatibility_mode: CompatibilityMode::Hybrid,
        })
    }

    /// Create configuration optimized for real-time chat
    pub fn for_realtime_chat() -> primitives::MemoryResult<Self> {
        Ok(Self {
            database: DatabaseConfig::new(DatabaseType::Memory, "memory", "chat", "realtime")?
                .with_pool_config(PoolConfig::minimal()),
            vector_store: VectorStoreConfig::new(VectorStoreType::Memory, EmbeddingConfig::default(), 1536)?
                .with_performance_config(PerformanceConfig::minimal()),
            llm: LLMConfig::new(LLMProvider::OpenAI, "gpt-4")?.with_streaming(true),
            enable_cognitive: false,
            compatibility_mode: CompatibilityMode::Hybrid,
        })
    }

    /// Create configuration optimized for large-scale data processing
    pub fn for_large_scale() -> primitives::MemoryResult<Self> {
        Ok(Self {
            database: DatabaseConfig::new(
                DatabaseType::PostgreSQL,
                "postgresql://localhost:5432/fluent_ai",
                "production",
                "memory_large",
            )?
            .with_pool_config(PoolConfig::optimized(DatabaseType::PostgreSQL)),
            vector_store: VectorStoreConfig::new(VectorStoreType::FAISS, EmbeddingConfig::default(), 1536)?
                .with_index_config(IndexConfig::optimized(IndexType::IVFPQ, 1536, 1000000))
                .with_performance_config(PerformanceConfig::optimized(VectorStoreType::FAISS)),
            llm: LLMConfig::default(),
            enable_cognitive: true,
            compatibility_mode: CompatibilityMode::Hybrid,
        })
    }
}
