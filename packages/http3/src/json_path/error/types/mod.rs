//! Error types for JSONPath operations
//!
//! This module provides the core error types and result type alias for JSONPath operations.

mod core;
mod display;
mod traits;

#[cfg(test)]
mod tests;

pub use core::{JsonPathError, JsonPathResult};
