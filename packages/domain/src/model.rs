//! Core model trait definitions for AI services

use crate::model_info_provider::ModelInfoData;

/// Trait for AI models that can provide information about their capabilities
pub trait Model {
    /// Get information about this model including pricing, token limits, and capabilities
    fn info(&self) -> ModelInfoData;
}