//! Unified Context Module
//!
//! This module consolidates all context-related functionality including:
//! - Document types and loading (from document.rs)
//! - Chunk types for streaming operations (from chunk.rs)
//! - Production-ready context management with memory integration (from context.rs)
//! - Core file loading interface and implementation (from loader.rs)
//! - Structured data extraction from unstructured text (from extraction/)
//!
//! The module provides a clean, unified interface for all context operations
//! while maintaining high performance and memory integration capabilities.

pub mod chunk;
pub mod document;
pub mod extraction;
pub mod loader;
pub mod provider;
/// Context trait definitions for trait-backed architecture
pub mod traits;

// Re-export all types for easy access
pub use chunk::*;
pub use document::*;
pub use extraction::*;
pub use loader::*;
pub use provider::*;
// Re-export trait types for trait-backed architecture
pub use traits::{
    CandleContext, CandleContextChunk, CandleContextType, CandleContextMetadata,
    CandleContextSource, CandleContextCapabilities, CandleFileContext
};
