//! HTTP error types and utilities
//!
//! This module provides comprehensive error handling for HTTP operations with
//! zero-allocation patterns and production-quality error reporting.

pub mod constructors;
pub mod conversions;
pub mod legacy;
pub mod types;

// Re-export the main types for backward compatibility
#[cfg(any(
    feature = "gzip",
    feature = "zstd",
    feature = "brotli",
    feature = "deflate",
))]
pub(crate) use legacy::BadScheme;
// Re-export legacy functions for backward compatibility
pub use legacy::{TimedOut, body, builder, request};
pub use types::{HttpError, HttpResult};
