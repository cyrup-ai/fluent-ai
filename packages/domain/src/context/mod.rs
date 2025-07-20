//! Unified Context Module
//!
//! This module consolidates all context-related functionality including:
//! - Document types and loading (from document.rs)
//! - Chunk types for streaming operations (from chunk.rs)
//! - Production-ready context management with memory integration (from context.rs)
//! - Core file loading interface and implementation (from loader.rs)
//!
//! The module provides a clean, unified interface for all context operations
//! while maintaining high performance and memory integration capabilities.

pub mod chunk;
pub mod document;
pub mod loader;
pub mod provider;

// Re-export all types for easy access
pub use chunk::*;
pub use document::*;
pub use loader::*;
pub use provider::*;
