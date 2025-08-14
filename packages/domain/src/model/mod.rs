//! Model system core module - RE-EXPORTS FROM model-info
//!
//! This module provides domain-specific extensions for AI model management
//! while using model-info package as the single source of truth for all model types.

// Domain-specific modules (not duplicating model-info functionality)
pub mod cache;
pub mod capabilities;
pub mod model_validation;
pub mod resolver;
pub mod traits;
pub mod unified_registry;
pub mod usage;
pub mod validation;

// Re-export modules that contain domain-specific extensions
pub mod info;
pub mod models;
pub mod providers;

// PRIMARY: Re-export all model types from model-info (single source of truth)
pub use model_info::{
    // Discovery Provider enum (unit variants like Provider::Anthropic)
    DiscoveryProvider as Provider,
    Model,
    // Capabilities
    ModelCapabilities,
    // Discovery functionality
    ModelDiscoveryRegistry,
    ModelError,
    ModelFilter,
    // Core types
    ModelInfo,
    ModelInfoBuilder,
    ModelQueryResult,
    // Registry functionality
    ModelRegistry,
    ProviderModels,
    ProviderTrait,
    RegisteredModel,
    RegistryStats as ModelInfoRegistryStats,
    Result,
};

// Create error module that re-exports from model-info for backward compatibility
pub mod error {
    pub use model_info::{ModelError, Result};
}

// Domain-specific capabilities
// Model-info integration modules
pub use cache::{CacheConfig, CacheStats, ModelCache};
pub use capabilities::{Capability, DomainModelCapabilities, ModelPerformance, UseCase};
pub use model_validation::{BatchValidationResult, ModelValidator, ValidationResult};
// Other domain-specific functionality
pub use resolver::*;
pub use traits::*;
// Provider is already re-exported from model_info above and now has the required methods

// Model registry functionality
pub use unified_registry::{RegistryStats, UnifiedModelRegistry};
pub use usage::Usage;
pub use validation::*;
