//! HTTP3 Builder API modules
//!
//! Provides the complete fluent API for building and executing HTTP requests
//! with zero allocation and elegant method chaining.

pub mod auth;
pub mod body;
pub mod core;
pub mod headers;
pub mod methods;

// Re-export all public types for convenience
pub use auth::*;
pub use body::*;
pub use core::*;
pub use headers::*;
pub use methods::*;