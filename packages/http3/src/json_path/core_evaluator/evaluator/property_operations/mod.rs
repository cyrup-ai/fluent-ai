//! Property-based operations for JSONPath evaluation
//!
//! Handles property path evaluation and recursive property finding operations.

mod core;
mod recursive;
mod extensions;
#[cfg(test)]
mod tests;

// Re-export the main struct for public API compatibility
pub use core::PropertyOperations;