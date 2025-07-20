//! Memory System Module
//!
//! Unified high-performance memory system with zero-allocation patterns,
//! quantum-inspired cognitive computing, and SIMD-optimized operations.

/// Memory primitives with zero-allocation design
pub mod primitives;

/// Quantum-inspired cognitive computing types
mod cognitive;

/// High-performance configuration system
mod config;

/// Compatibility layer for legacy types
mod compatibility;

/// Core memory management and configuration (legacy)
mod manager;

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

/// Memory workflow management
mod workflow;

/// Legacy types (maintained for backward compatibility)
mod types_legacy;

// Re-export all new domain types
// Type aliases for migration compatibility
use std::future::Future;
use std::pin::Pin;

pub use cognitive::*;
pub use compatibility::*;
pub use config::*;
// Conditional re-exports for cognitive features
#[cfg(feature = "cognitive")]
pub use fluent_ai_memory::{
    CognitiveMemoryManager, CognitiveMemoryNode, CognitiveSettings,
    EvolutionMetadata as LegacyEvolutionMetadata, QuantumSignature as LegacyQuantumSignature,
};
// Re-export fluent_ai_memory types for convenience
#[cfg(feature = "fluent-ai-memory")]
pub use fluent_ai_memory::{
    MemoryConfig, MemoryManager as MemoryManagerTrait, MemoryMetadata as ExternalMemoryMetadata,
    SurrealDBMemoryManager,
    memory::{
        MemoryQuery, MemoryStream, PendingDeletion, PendingMemory, PendingRelationship,
        RelationshipStream, SurrealMemoryQuery,
    },
};
// Re-export core legacy types for backward compatibility
pub use manager::Memory;
pub use ops::{
    CpuArchitecture, CpuFeatures, EMBEDDING_DIMENSION, Op, SIMD_WIDTH, SMALL_EMBEDDING_DIMENSION,
};
pub use primitives::*;
pub use tool::{MemoryOperation, MemoryResult, MemoryTool, MemoryToolError, MemoryToolResult};
// Re-export commonly used primitives types
pub use primitives::{MemoryTypeEnum, MemoryContent};
// Legacy type aliases for backward compatibility
pub use types_legacy::{
    Error as LegacyMemoryError, ImportanceContext, MemoryMetadata as LegacyMemoryMetadata,
    MemoryNode as LegacyMemoryNode, MemoryRelationship as LegacyMemoryRelationship,
    MemoryType as LegacyMemoryType, VectorStoreIndex, VectorStoreIndexDyn,
    calculate_importance, next_memory_id,
};

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Fallback trait definition for when fluent-ai-memory feature is not enabled
#[cfg(not(feature = "fluent-ai-memory"))]
pub trait MemoryManagerTrait: Send + Sync {
    type Error;
    type MemoryNode;
    
    fn store_memory(&self, memory: Self::MemoryNode) -> Result<(), Self::Error>;
}

// Trait is already exported as public above

// Primary error type is now the new MemoryError from primitives
pub type Error = crate::memory::primitives::MemoryError;

// Legacy compatibility aliases
pub type VectorStoreError = Error;
// MemoryError alias removed to avoid conflict with fluent_ai_memory::Error

/// Memory system configuration combining all subsystem configurations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemorySystemConfig {
    /// Database configuration
    pub database: DatabaseConfig,
    /// Vector store configuration  
    pub vector_store: VectorStoreConfig,
    /// LLM configuration for AI operations
    pub llm: LLMConfig,
    /// Enable cognitive features
    pub enable_cognitive: bool,
    /// Compatibility mode for legacy systems
    pub compatibility_mode: CompatibilityMode,
}

impl MemorySystemConfig {
    /// Create optimized configuration for production use
    pub fn optimized() -> crate::memory::primitives::MemoryResult<Self> {
        Ok(Self {
            database: DatabaseConfig::default(),
            vector_store: VectorStoreConfig::default(),
            llm: LLMConfig::default(),
            enable_cognitive: true,
            compatibility_mode: CompatibilityMode::Hybrid,
        })
    }

    /// Create minimal configuration for testing
    pub fn minimal() -> crate::memory::primitives::MemoryResult<Self> {
        Ok(Self {
            database: DatabaseConfig::default(),
            vector_store: VectorStoreConfig::new(
                VectorStoreType::Memory,
                EmbeddingConfig::default(),
                768,
            )?,
            llm: LLMConfig::new(LLMProvider::OpenAI, "gpt-4")?,
            enable_cognitive: false,
            compatibility_mode: CompatibilityMode::Strict,
        })
    }

    /// Validate configuration consistency
    pub fn validate(&self) -> crate::memory::primitives::MemoryResult<()> {
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

/// Memory system builder for ergonomic configuration
#[derive(Debug, Default)]
pub struct MemorySystemBuilder {
    database_config: Option<DatabaseConfig>,
    vector_config: Option<VectorStoreConfig>,
    llm_config: Option<LLMConfig>,
    enable_cognitive: bool,
    compatibility_mode: CompatibilityMode,
}

impl MemorySystemBuilder {
    /// Create new memory system builder
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set database configuration
    #[inline]
    pub fn with_database_config(mut self, config: DatabaseConfig) -> Self {
        self.database_config = Some(config);
        self
    }

    /// Set vector store configuration
    #[inline]
    pub fn with_vector_config(mut self, config: VectorStoreConfig) -> Self {
        self.vector_config = Some(config);
        self
    }

    /// Set LLM configuration
    #[inline]
    pub fn with_llm_config(mut self, config: LLMConfig) -> Self {
        self.llm_config = Some(config);
        self
    }

    /// Enable cognitive features
    #[inline]
    pub fn with_cognitive(mut self, enabled: bool) -> Self {
        self.enable_cognitive = enabled;
        self
    }

    /// Set compatibility mode
    #[inline]
    pub fn with_compatibility_mode(mut self, mode: CompatibilityMode) -> Self {
        self.compatibility_mode = mode;
        self
    }

    /// Build memory system configuration
    pub fn build(self) -> crate::memory::primitives::MemoryResult<MemorySystemConfig> {
        let config = MemorySystemConfig {
            database: self.database_config.unwrap_or_default(),
            vector_store: self.vector_config.unwrap_or_default(),
            llm: self.llm_config.unwrap_or_default(),
            enable_cognitive: self.enable_cognitive,
            compatibility_mode: self.compatibility_mode,
        };

        config.validate()?;
        Ok(config)
    }
}

/// Convenience functions for creating memory system configurations
impl MemorySystemConfig {
    /// Create builder for memory system configuration
    #[inline]
    pub fn builder() -> MemorySystemBuilder {
        MemorySystemBuilder::new()
    }

    /// Create configuration optimized for semantic search
    pub fn for_semantic_search() -> crate::memory::primitives::MemoryResult<Self> {
        Ok(Self::builder()
            .with_vector_config(
                VectorStoreConfig::new(
                    VectorStoreType::FAISS,
                    EmbeddingConfig::new(
                        EmbeddingModelType::OpenAI,
                        "text-embedding-3-large",
                        3072,
                    ),
                    3072,
                )?
                .with_distance_metric(DistanceMetric::Cosine)
                .with_simd_config(SimdConfig::optimized()),
            )
            .with_cognitive(true)
            .build()?)
    }

    /// Create configuration optimized for real-time chat
    pub fn for_realtime_chat() -> crate::memory::primitives::MemoryResult<Self> {
        Ok(Self::builder()
            .with_database_config(
                DatabaseConfig::new(DatabaseType::Memory, "memory", "chat", "realtime")?
                    .with_pool_config(PoolConfig::minimal()),
            )
            .with_vector_config(
                VectorStoreConfig::new(VectorStoreType::Memory, EmbeddingConfig::default(), 1536)?
                    .with_performance_config(PerformanceConfig::minimal()),
            )
            .with_llm_config(LLMConfig::new(LLMProvider::OpenAI, "gpt-4")?.with_streaming(true))
            .with_cognitive(false)
            .build()?)
    }

    /// Create configuration optimized for large-scale data processing
    pub fn for_large_scale() -> crate::memory::primitives::MemoryResult<Self> {
        Ok(Self::builder()
            .with_database_config(
                DatabaseConfig::new(
                    DatabaseType::PostgreSQL,
                    "postgresql://localhost:5432/fluent_ai",
                    "production",
                    "memory_large",
                )?
                .with_pool_config(PoolConfig::optimized(DatabaseType::PostgreSQL)),
            )
            .with_vector_config(
                VectorStoreConfig::new(VectorStoreType::FAISS, EmbeddingConfig::default(), 1536)?
                    .with_index_config(IndexConfig::optimized(IndexType::IVFPQ, 1536, 1000000))
                    .with_performance_config(PerformanceConfig::optimized(VectorStoreType::FAISS)),
            )
            .with_cognitive(true)
            .build()?)
    }
}
