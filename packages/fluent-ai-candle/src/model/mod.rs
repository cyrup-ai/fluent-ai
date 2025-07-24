//! Modular model system with sophisticated loading and caching
//!
//! This module is organized into focused sub-modules:
//! - `loading` - Model loading with VarBuilder patterns and progressive loading
//! - `wrappers` - Architecture-specific model wrappers with cache management
//! - `cache` - Lock-free KV cache system with atomic operations
//! - `types` - Model configuration types and enumerations
//! - `metrics` - Performance metrics and statistics
//! - `core` - Main CandleModel implementation with atomic state management

pub mod cache;
pub mod core;
pub mod error;
pub mod fluent;
pub mod info;
pub mod loading;
pub mod metrics;
pub mod model_trait;
pub mod registry;
pub mod types;
pub mod usage;
// Removed useless wrappers module - direct model implementations only

// Re-export key types for backward compatibility
pub use core::CandleModel;

pub use cache::{CacheKey, KVCacheConfig, KVCacheEntry, KVCacheManager, KVCacheStats};
pub use error::{ModelError, ModelResult, ValidationError, ValidationResult};
pub use info::HasModelInfo;
// ModelInfo moved to types module - use CandleModelInfo instead
pub use loading::{
    LoadingStage, ModelLoader, ModelMetadata, ProgressCallback, RecoveryStrategy, TensorInfo,
};
pub use metrics::{GenerationMetrics, ModelMetrics, ModelPerformanceStats};
pub use registry::{ModelRegistry, RegistryError, global_registry, register_model, get_model, list_models, model_exists};
pub use model_trait::Model;
// All model traits are now defined in the canonical types module with Candle prefix:
// - CandleLoadableModel (was LoadableModel)
// - CandleUsageTrackingModel (was UsageTrackingModel)  
// - CandleCompletionModel
// - CandleConfigurableModel
// - CandleTokenizerModel
// Import them from crate::types instead
pub use types::{ModelConfig, ModelType, QuantizationType};
// Usage moved to types module - use CandleUsage instead
// Removed useless wrapper abstractions - direct model implementations only
