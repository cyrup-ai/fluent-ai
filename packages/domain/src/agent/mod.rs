//! Agent domain types and role implementations
//!
//! This module consolidates agent data structures and role definitions with automatic memory tool injection.
//! Builder implementations are in fluent_ai package.

pub mod builder;
pub mod chat;
pub mod core;
pub mod role;
pub mod types;

// Re-export commonly used types
pub use core::*;

pub use builder::*;
pub use chat::*;
pub use role::*;
// Removed unused import: types::*
