//! Model information types and utilities
//!
//! Contains types for describing model capabilities and metadata.

// Removed duplicate ModelInfo type - use CandleModelInfo from types module instead

/// Trait for types that can provide model information
pub trait HasModelInfo {
    /// Get the model information
    fn model_info(&self) -> &crate::types::CandleModelInfo;
}
