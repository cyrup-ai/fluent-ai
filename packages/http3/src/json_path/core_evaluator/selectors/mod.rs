//! JSONPath selector application with zero-allocation patterns
//!
//! Decomposed selector handling for individual selector types:
//! child, index, wildcard, filter, slice, union with both owned and reference-based operations.

pub mod core;
pub mod arrays;
pub mod wildcards;
pub mod filters;

// Re-export core functionality
pub use core::*;