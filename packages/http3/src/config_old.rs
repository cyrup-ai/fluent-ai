//! HTTP client configuration
//!
//! This module has been decomposed into focused submodules for better maintainability.
//! All original functionality is preserved through re-exports.

// Re-export everything from the config module
pub use self::config::*;

// Import the decomposed config module
mod config;