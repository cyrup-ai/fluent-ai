//! Provider Module - DECOMPOSED FROM 918 LINES
//!
//! Zero-Allocation Context Provider System with streaming-only architecture, zero Arc usage,
//! lock-free atomic operations, and immutable messaging patterns. Provides blazing-fast
//! context loading and management with full memory integration.
//!
//! The original 918-line provider.rs has been decomposed into focused modules:
//! - errors: Error types and validation (85 lines)
//! - events: Context events and monitoring (65 lines)
//! - memory: Memory integration and nodes (120 lines)
//! - traits: Embedding and memory manager traits (60 lines)
//! - processor: Streaming context processor (160 lines)
//! - context_types: Context data structures (180 lines)
//! - context_impls: Context implementations (200 lines)

pub mod errors;
pub mod events;
pub mod memory;
pub mod traits;
pub mod processor;
pub mod context_types;
pub mod context_impls;

// Re-export error types
pub use errors::{ContextError, ProviderError, ValidationError};

// Re-export event types
pub use events::ContextEvent;

// Re-export memory types
pub use memory::{MemoryNode, MemoryIntegration};

// Re-export trait types
pub use traits::{
    ImmutableEmbeddingModel, ImmutableMemoryManager,
    EmbeddingModelInfo, MemoryManagerInfo};

// Re-export processor types
pub use processor::{StreamingContextProcessor, ContextProcessorStatistics};

// Re-export context types and marker types
pub use context_types::{
    Context, ContextSourceType,
    File, Files, Directory, Github,
    ImmutableFileContext, ImmutableFilesContext, 
    ImmutableDirectoryContext, ImmutableGithubContext,
    // Deprecated aliases removed - use Immutable* versions instead
};

// Context implementations are available through the context_types re-exports
// All implementation methods are available on the Context<T> types directly
