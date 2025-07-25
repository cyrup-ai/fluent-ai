//! Zero-allocation error handling for candle integration
//!
//! This module provides comprehensive error handling capabilities with zero-allocation
//! patterns and structured error context for better debugging and error recovery.

pub mod error_types;
pub mod error_helpers;
pub mod error_context;
pub mod conversions;
pub mod macros;

// Re-export core types and functions
pub use error_types::{CandleError, CandleResult};
pub use error_context::{ErrorContext, CandleErrorWithContext};

