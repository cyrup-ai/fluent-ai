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
pub mod loading;
pub mod metrics;
pub mod types;
pub mod wrappers;

// Re-export key types for backward compatibility
pub use core::CandleModel;

pub use cache::{CacheKey, KVCacheConfig, KVCacheEntry, KVCacheManager, KVCacheStats};
pub use loading::{
    LoadingStage, ModelLoader, ModelMetadata, ProgressCallback, RecoveryStrategy, TensorInfo,
};
pub use metrics::{GenerationMetrics, ModelMetrics, ModelPerformanceStats};
pub use types::{ModelConfig, ModelType, QuantizationType};
pub use wrappers::{
    CacheContext, EnhancedLlamaWrapper, GemmaWrapper, LlamaWrapper, MistralWrapper, PhiWrapper,
    QwenWrapper,
};
