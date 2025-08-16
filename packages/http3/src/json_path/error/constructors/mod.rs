//! Error constructor functions module
//!
//! Provides convenient factory functions for creating JSONPath error types
//! with proper context and formatting.

mod core;
mod helpers;
#[cfg(test)]
mod tests;

// Re-export all constructor functions for public API compatibility
pub use core::*;

pub use helpers::*;
