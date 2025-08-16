//! AsyncStream-native service and layer traits
//!
//! Provides 100% compatible replacements for tower::Service and tower::Layer
//! using pure AsyncStream patterns with zero allocation and zero futures.
//!
//! This module retains all tower functionality while using the streams-first architecture.

mod concurrency;
mod core;
mod identity;
mod timeout;

#[cfg(test)]
mod tests;

// Re-export core types and traits for backward compatibility
pub use core::{AsyncStreamLayer, AsyncStreamService, ConnResult};

pub use concurrency::{AsyncStreamConcurrencyLayer, AsyncStreamConcurrencyService};
pub use identity::AsyncStreamIdentityLayer;
// Re-export layer implementations
pub use timeout::{AsyncStreamTimeoutLayer, AsyncStreamTimeoutService};

// Re-export Conn type from connect module
pub use crate::hyper::connect::Conn;
