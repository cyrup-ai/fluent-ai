//! Type conversions and compatibility implementations for JSONPath errors
//!
//! Provides From trait implementations for converting external error types
//! into JsonPathError variants, along with helper traits and utilities.

mod helpers;
mod std_conversions;
#[cfg(test)]
mod tests;

// Re-export all conversion functionality for public API compatibility
pub use helpers::*;
pub use std_conversions::*;
