//! High-performance vector store implementations with zero-allocation patterns
//!
//! This module provides in-memory vector stores optimized for similarity search,
//! with configurable similarity metrics, thresholds, and indexing strategies.

pub mod in_memory;
pub mod index;
pub mod similarity_search;

pub use in_memory::*;
pub use index::*;
pub use similarity_search::*;

// Re-export domain types for compatibility
pub use crate::domain::memory::{VectorStoreError, VectorStoreIndex, VectorStoreIndexDyn};
