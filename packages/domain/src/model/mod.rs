//! Model system core module
//!
//! This module provides the core abstractions and types for AI model management,
//! including traits, information, registry, and error handling. Integrates with
//! model-info package for real-time model data from AI providers.

pub mod capabilities;
pub mod error;
pub mod info;
pub mod provider_traits;
pub mod registry;
pub mod resolver;
pub mod traits;
pub mod usage;
pub mod validation;

// New modules for model-info integration
pub mod model_registry;
pub mod cache;
pub mod model_validation;

// Generated modules - created by build system
pub mod providers;
pub mod models;

// Re-export commonly used types
pub use capabilities::*;
pub use error::{ModelError, Result};
pub use info::ModelInfo;
pub use provider_traits::{ProviderModel, ProviderModelInfo, ProviderTrait, ProviderError, ProviderResult};
pub use registry::ModelRegistry;
pub use resolver::*;
pub use traits::*;
pub use usage::Usage;
pub use validation::*;

// Re-export model-info types for unified access - REAL AI MODEL DATA
pub use model_info::{
    common::{Model as ModelInfoTrait, ModelInfo as RealModelInfo, ProviderTrait as ModelInfoProviderTrait},
    OpenAiModel,
    MistralModel, 
    AnthropicModel,
    TogetherModel,
    OpenRouterModel,
    HuggingFaceModel,
    XaiModel,
};

// Re-export new model-info integration modules
pub use model_registry::{ModelRegistry as UnifiedModelRegistry, ModelFilter, ModelQueryResult, RegistryStats};
pub use cache::{ModelCache, CacheStats, CacheConfig};
pub use model_validation::{ModelValidator, ValidationResult, BatchValidationResult};

// Re-export generated types (commented out until build system generates actual content)
// pub use providers::*;
// pub use models::*;
