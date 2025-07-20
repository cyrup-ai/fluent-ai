//! Chat syntax features module
//!
//! This module provides comprehensive chat syntax features with zero-allocation patterns
//! and production-ready functionality. All submodules follow blazing-fast, lock-free,
//! and elegant ergonomic design principles.
//!
//! ## Features
//! - Rich message formatting with SIMD-optimized parsing
//! - Command system with slash commands and auto-completion
//! - Template and macro system for reusable patterns
//! - Advanced configuration with nested settings
//! - Real-time features with atomic state management
//! - Enhanced history management with full-text search
//! - External integration system with plugin architecture
//!
//! ## Architecture
//! All modules use zero-allocation patterns with Arc<str> for string sharing,
//! crossbeam-skiplist for lock-free data structures, and atomic operations
//! for thread-safe state management.

pub mod commands;
pub mod config;
pub mod export;
pub mod formatting;
pub mod integrations;
pub mod macros;
pub mod realtime;
pub mod search;
pub mod templates;

// Re-export all public types for convenience
pub use commands::*;
pub use config::*;
pub use export::*;
pub use formatting::*;
pub use integrations::*;
pub use macros::*;
pub use realtime::*;
pub use search::*;
pub use templates::*;
