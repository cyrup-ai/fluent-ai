//! HTTP request building and execution module.
//!
//! This module provides the core types and functionality for building and executing HTTP requests
//! in a streaming, zero-allocation manner using the fluent_ai_async architecture.

pub mod auth;
pub mod body;
pub mod execution;
pub mod headers;
pub mod types;

#[cfg(test)]
pub mod tests;

// Re-export the main types for backward compatibility
pub use types::{Request, RequestBuilder};
