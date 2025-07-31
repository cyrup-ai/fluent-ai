// Model information and registry for fluent-ai
// This package consolidates all model-related functionality 

pub mod common;
pub mod registry;
pub mod discovery;

// Re-export core types for external use
pub use common::{
    // Core types
    ModelInfo, ModelInfoBuilder, ProviderTrait, Model,
    // Error handling
    ModelError, Result,
    // Capabilities and collections
    ModelCapabilities, ProviderModels,
};

// Re-export registry functionality
pub use registry::{ModelRegistry, RegisteredModel};

// Re-export discovery functionality  
pub use discovery::{
    ModelDiscoveryRegistry, ModelFilter, ModelQueryResult, 
    RegistryStats, Provider as DiscoveryProvider,
};