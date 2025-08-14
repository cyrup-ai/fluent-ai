// The ProviderBuilder trait and related build-time logic is in buildlib/
// This file contains only runtime types and interfaces

/// Runtime provider trait for accessing generated model information
pub trait ProviderTrait {
    /// Get the provider name
    fn provider_name(&self) -> &'static str;

    /// Get all available models for this provider  
    fn list_models(&self) -> Vec<crate::common::ModelInfo>;

    /// Get specific model info by name
    fn get_model(&self, name: &str) -> Option<crate::common::ModelInfo>;
}
