//! Internal utility modules for fluent-ai
//! 
//! This module contains high-performance utilities that are used internally
//! by provider implementations and other internal systems. These are not
//! part of the public API but provide zero-allocation, blazing-fast building
//! blocks for the fluent-ai architecture.

pub mod json_util;

// Re-export commonly used utilities when needed
// pub use json_util::{...};