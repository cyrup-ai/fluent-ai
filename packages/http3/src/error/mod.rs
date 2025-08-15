//! HTTP error types and utilities
//!
//! This module provides comprehensive error handling for HTTP operations with
//! zero-allocation patterns and production-quality error reporting.

pub mod types;
pub mod constructors;
pub mod conversions;
pub mod legacy;

// Re-export the main types for backward compatibility
pub use types::{HttpError, HttpResult};

// Re-export legacy functions for backward compatibility
pub use legacy::{builder, request, TimedOut, body};

#[cfg(any(
    feature = "gzip",
    feature = "zstd", 
    feature = "brotli",
    feature = "deflate",
))]
pub(crate) use legacy::BadScheme;