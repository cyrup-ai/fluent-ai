//! ModelInfo provider for looking up model information

use crate::model::{ModelInfo, Models};

/// Error types for ModelInfo operations
#[derive(Debug, thiserror::Error)]
pub enum ModelInfoError {
    #[error("Model lookup failed: {reason}")]
    LookupFailed { reason: String },
}

/// Trait for providing ModelInfo data for AI models
pub trait ModelInfoProvider {
    /// Get ModelInfo for the specified model
    fn get_model_info(&self, model: &Models) -> Result<ModelInfo, ModelInfoError>;
}

/// Default implementation that provides model information
#[derive(Debug, Clone, Default)]
pub struct DefaultModelInfoProvider;

impl ModelInfoProvider for DefaultModelInfoProvider {
    fn get_model_info(&self, model: &Models) -> Result<ModelInfo, ModelInfoError> {
        Ok(model.info())
    }
}
