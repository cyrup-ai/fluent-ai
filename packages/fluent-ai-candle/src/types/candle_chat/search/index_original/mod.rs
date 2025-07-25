//! Search indexing system with SIMD optimization
//!
//! This module provides efficient inverted index construction and maintenance
//! with SIMD-optimized text processing and TF-IDF scoring.
//!
//! The module is decomposed into focused submodules:
//! - [`types`] - Core types and data structures
//! - [`indexing`] - Message indexing operations with streaming support
//! - [`search_operations`] - Core search operations and query processing
//! - [`fuzzy_search`] - Fuzzy search implementation with Levenshtein distance
//! - [`statistics`] - Search statistics and monitoring functionality

pub mod types;
pub mod indexing;
pub mod search_operations;
pub mod fuzzy_search;
pub mod statistics;

// Re-export the main types for public API
pub use types::{ChatSearchIndex, IndexEntry};

// Re-export the handle_error macro for internal use
pub(crate) use types::handle_error;