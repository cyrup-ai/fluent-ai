//! HTTP configuration module
//!
//! Provides comprehensive HTTP client configuration including core types,
//! preset configurations, timeout settings, and security options.

// Module declarations
pub mod client;
pub mod core;
pub mod security;
pub mod timeouts;

// Re-export all public types for backward compatibility
pub use core::{ConnectionReuse, HttpConfig, RetryPolicy, RetryableError};

// Re-export preset configuration methods
pub use client::*;
// Re-export security configuration methods
pub use security::*;
// Re-export timeout configuration methods
pub use timeouts::*;
