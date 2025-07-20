//! Memory System Module
//!
//! This module provides a consolidated memory system with sparse file separation
//! for singular concerns. Each submodule handles a specific aspect of memory management.

/// Core memory management and configuration
pub mod manager;

/// SIMD-optimized vector operations for high-performance memory processing
pub mod ops;

/// Memory tool implementation for MCP integration
pub mod tool;

/// Memory workflow system for cognitive processing
pub mod workflow;

// Re-export core types from manager
// Legacy compatibility types
use std::future::Future;
use std::pin::Pin;

// Conditional re-exports for cognitive features
#[cfg(feature = "cognitive")]
pub use fluent_ai_memory::{
    CognitiveMemoryManager, CognitiveMemoryNode, CognitiveSettings, CognitiveState,
    EvolutionMetadata, QuantumSignature,
};
// Re-export fluent_ai_memory types for convenience
pub use fluent_ai_memory::{
    Error as MemoryError, MemoryConfig, MemoryManager as MemoryManagerTrait, MemoryMetadata,
    MemoryNode, MemoryType, SurrealDBMemoryManager,
    memory::{
        MemoryQuery, MemoryRelationship, MemoryStream, MemoryTypeEnum, PendingDeletion,
        PendingMemory, PendingRelationship, RelationshipStream, SurrealMemoryQuery,
    },
};
pub use manager::{
    Memory,
};
// Re-export SIMD operations
pub use ops::{
    CpuArchitecture, CpuFeatures, EMBEDDING_DIMENSION, SIMD_WIDTH, SMALL_EMBEDDING_DIMENSION, Op,
};
// Re-export memory tool
pub use tool::{
    MemoryOperation, MemoryResult, MemoryTool, MemoryToolError, MemoryToolResult,
};
// Re-export workflow system
pub use workflow::{
    AdaptiveWorkflow, MemoryEnhancedWorkflow, Prompt, PromptError, WorkflowError, apply_feedback,
    conversation_workflow, rag_workflow,
};

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
pub type Error = MemoryError;
pub type VectorStoreError = MemoryError;
