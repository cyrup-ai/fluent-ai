//! Composite processor for chaining multiple logits processors
//!
//! This module has been decomposed into focused submodules for better organization
//! while maintaining all original functionality. The decomposition follows the
//! ≤300 lines per module architectural constraint.
//!
//! ## Decomposed Structure
//! - `core`: Main CompositeProcessor implementation with execution chain logic
//! - `builder`: CompositeProcessorBuilder with fluent API and preset configurations  
//! - `parallel`: ParallelCompositeProcessor for independent processor execution
//! - `tests`: Comprehensive test suite for all composite processor functionality
//!
//! ## Migration Guide
//! All public APIs remain the same and are re-exported from this module:
//! ```rust
//! // This still works exactly the same:
//! use crate::sampling::composite::{CompositeProcessor, CompositeProcessorBuilder};
//! 
//! let composite = CompositeProcessorBuilder::new()
//!     .temperature(0.8)?
//!     .top_k(50)?
//!     .build()?;
//! ```

// Submodules with focused responsibilities
pub mod core;
pub mod builder;
pub mod parallel;

// Tests are only included in test builds
#[cfg(test)]
pub mod tests;

// Re-export all public APIs for backward compatibility
pub use core::CompositeProcessor;
pub use builder::{CompositeProcessorBuilder, presets};
pub use parallel::{ParallelCompositeProcessor, MergeStrategy};

// Legacy alias for utils module (now presets)
pub mod utils {
    pub use super::builder::presets::*;
}