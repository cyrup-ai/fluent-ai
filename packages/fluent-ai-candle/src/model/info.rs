//! Model information types and utilities
//!
//! Contains types for describing model capabilities and metadata.

// Removed duplicate ModelInfo type - use CandleModelInfo from types module instead

/// Trait for types that can provide model information and metadata
///
/// This trait enables types to expose their model information in a standardized way.
/// Implementors can provide access to model metadata including capabilities,
/// version information, and descriptive details.
///
/// # Example
///
/// ```rust
/// use crate::model::info::HasModelInfo;
///
/// let model = MyModel::new();
/// let info = model.model_info();
/// println!("Model name: {}", info.name);
/// println!("Model version: {}", info.version);
/// ```
pub trait HasModelInfo {
    /// Get the model information and metadata
    ///
    /// Returns a reference to the static model information that describes
    /// the capabilities, version, and other metadata for this model instance.
    ///
    /// # Returns
    ///
    /// A static reference to `CandleModelInfo` containing the model's metadata
    ///
    /// # Example
    ///
    /// ```rust
    /// let info = model.model_info();
    /// assert!(!info.name.is_empty());
    /// assert!(!info.version.is_empty());
    /// ```
    fn model_info(&self) -> &crate::types::CandleModelInfo;
}
