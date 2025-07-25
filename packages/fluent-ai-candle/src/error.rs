//! Zero-allocation error handling for candle integration
//!
//! This module has been decomposed into focused submodules for better maintainability.
//! All original functionality is preserved through re-exports.

// Re-export everything from the decomposed modules
pub use self::error::*;

// Module declaration
mod error;