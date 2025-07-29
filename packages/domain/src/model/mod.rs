//! Model system core module - RE-EXPORTS FROM model-info
//!
//! This module provides domain-specific extensions for AI model management
//! while using model-info package as the single source of truth for all model types.

// Domain-specific modules (not duplicating model-info functionality)
pub mod unified_registry;
pub mod capabilities;
pub mod registry;
pub mod resolver;
pub mod traits;
pub mod usage;
pub mod validation;

// New modules for model-info integration
pub mod model_registry;
pub mod cache;
pub mod model_validation;

// Re-export modules that contain domain-specific extensions
pub mod info;
pub mod providers;
pub mod models;

// PRIMARY: Re-export all model types from model-info (single source of truth)
pub use model_info::{
    // Core types
    ModelInfo, ProviderTrait, Model, ModelInfoBuilder, ModelError, Result, Provider,
    // Generated model enums
    OpenAi, Mistral, Anthropic, Together, OpenRouter, HuggingFace, Xai,
    // Common module
    common,
};

// Domain-specific capabilities
pub use capabilities::{Capability, DomainModelCapabilities, ModelPerformance, UseCase};
pub use usage::Usage;
pub use validation::*;

// Model registry functionality 
pub use unified_registry::{UnifiedModelRegistry, RegistryStats};

// Model-info integration modules
pub use model_registry::{ModelRegistry as LegacyModelRegistry, ModelFilter, ModelQueryResult};
pub use cache::{ModelCache, CacheStats, CacheConfig};
pub use model_validation::{ModelValidator, ValidationResult, BatchValidationResult};

// Legacy registry (domain-specific)
pub use registry::ModelRegistry;
pub use resolver::*;
pub use traits::*;