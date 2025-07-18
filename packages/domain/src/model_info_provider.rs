//! ModelInfo provider for looking up model information

// Domain should NOT depend on provider - provider depends on domain
// These types should be defined in domain or passed as generics
use crate::{ModelInfoData, Models};

/// Error types for ModelInfo operations
#[derive(Debug, thiserror::Error)]
pub enum ModelInfoError {
    #[error("Model lookup failed: {reason}")]
    LookupFailed { reason: String },
}

/// Trait for providing ModelInfo data for AI models
pub trait ModelInfoProvider {
    /// Get ModelInfo for the specified model
    fn get_model_info(&self, model: &Models) -> Result<ModelInfoData, ModelInfoError>;
}

/// Default implementation that uses the provider crate's Model trait
#[derive(Debug, Clone, Default)]
pub struct DefaultModelInfoProvider;

impl ModelInfoProvider for DefaultModelInfoProvider {
    fn get_model_info(&self, model: &Models) -> Result<ModelInfoData, ModelInfoError> {
        Ok(model.info())
    }
}
