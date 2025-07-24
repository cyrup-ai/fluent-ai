//! Model state for atomic swapping
//!
//! Defines the ModelState structure used for hot-swapping model implementations
//! with zero-allocation patterns and blazing-fast atomic operations.

use candle_core::Module;
use memmap2::Mmap;
use crate::model::types::ModelConfig;

/// Model state for atomic swapping with zero-allocation design
pub struct ModelState {
    /// The actual model implementation
    pub model: Box<dyn Module + Send + Sync>,
    /// Model configuration
    pub config: ModelConfig,
    /// Model file memory mapping for efficient resource management
    pub _mmap: Option<Mmap>,
}