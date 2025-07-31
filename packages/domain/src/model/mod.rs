//! Model system core module - RE-EXPORTS FROM model-info
//!
//! This module provides domain-specific extensions for AI model management
//! while using model-info package as the single source of truth for all model types.

// Domain-specific modules (not duplicating model-info functionality)
pub mod unified_registry;
pub mod capabilities;
pub mod resolver;
pub mod traits;
pub mod usage;
pub mod validation;
pub mod cache;
pub mod model_validation;

// Re-export modules that contain domain-specific extensions
pub mod info;
pub mod providers;
pub mod models;

// PRIMARY: Re-export all model types from model-info (single source of truth)
pub use model_info::{
    // Core types
    ModelInfo, ProviderTrait, Model, ModelInfoBuilder, ModelError, Result,
    // Discovery Provider enum (unit variants like Provider::Anthropic)
    DiscoveryProvider as Provider,
    // Registry functionality
    ModelRegistry, RegisteredModel,
    // Discovery functionality
    ModelDiscoveryRegistry, ModelFilter, ModelQueryResult, RegistryStats as ModelInfoRegistryStats,
    // Capabilities
    ModelCapabilities, ProviderModels,
};

// Create error module that re-exports from model-info for backward compatibility
pub mod error {
    pub use model_info::{ModelError, Result};
}

// Domain-specific capabilities
pub use capabilities::{Capability, DomainModelCapabilities, ModelPerformance, UseCase};
pub use usage::Usage;
pub use validation::*;

// Provider is already re-exported from model_info above and now has the required methods

// Model registry functionality 
pub use unified_registry::{UnifiedModelRegistry, RegistryStats};

// Model-info integration modules
pub use cache::{ModelCache, CacheStats, CacheConfig};
pub use model_validation::{ModelValidator, ValidationResult, BatchValidationResult};

// Other domain-specific functionality
pub use resolver::*;
pub use traits::*;