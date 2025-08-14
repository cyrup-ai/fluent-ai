// Model information and registry for fluent-ai
// This package consolidates all model-related functionality

pub mod common;
pub mod discovery;
pub mod generated_models;
pub mod providers;
pub mod registry;

// Re-export core types for external use
pub use common::{
    Model,
    // Capabilities and collections
    ModelCapabilities,
    // Error handling
    ModelError,
    // Core types
    ModelInfo,
    ModelInfoBuilder,
    ProviderModels,
    ProviderTrait,
    Result,
};
// Re-export discovery functionality
pub use discovery::{
    ModelDiscoveryRegistry, ModelFilter, ModelQueryResult, Provider as DiscoveryProvider,
    RegistryStats,
};
// Re-export registry functionality
pub use registry::{ModelRegistry, RegisteredModel};
