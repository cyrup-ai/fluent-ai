//! Plugin initialization system with comprehensive scaffolding
//!
//! This module provides complete plugin initialization functionality with zero
//! allocation patterns, blazing-fast performance, and production-ready scaffolding.

pub mod core;
pub mod engine;
pub mod templates;

// Re-export core types and functions for ergonomic use
pub use core::*;

pub use engine::*;
pub use templates::*;
