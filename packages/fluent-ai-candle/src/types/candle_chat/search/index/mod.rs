//! Search indexing system with SIMD optimization
//!
//! This module provides efficient inverted index construction and maintenance
//! with SIMD-optimized text processing and TF-IDF scoring.

// Core types and structures
pub mod core;
pub mod indexing;
pub mod query_engine;
pub mod search_ops;
pub mod sorting;
pub mod utils;

// Re-export main types
pub use core::{ChatSearchIndex, IndexEntry};

// Re-export for backward compatibility
pub use core::ChatSearchIndex as SearchIndex;