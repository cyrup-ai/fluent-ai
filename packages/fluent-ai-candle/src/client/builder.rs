//! Re-export builder functionality from submodules
//!
//! This module serves as the main entry point for builder functionality
//! while maintaining the original API through focused submodules.

pub mod client_builder;
pub mod typestate;

// Re-export public types for backward compatibility
pub use client_builder::CandleClientBuilder;
pub use typestate::{HasPrompt, NeedsPrompt};