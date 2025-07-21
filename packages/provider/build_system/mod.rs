// Zero-allocation build system modules for fluent_ai_provider
//
// This module contains all the build system components with zero-allocation,
// lock-free, and SIMD-optimized implementations.


pub mod errors;
pub mod yaml_processor;
pub mod cache_manager;
pub mod code_generator;
pub mod client_verifier;
pub mod string_utils;
pub mod performance;
pub mod model_loader;
pub mod change_detector;
pub mod yaml_manager;
pub mod incremental_generator;

// Re-export commonly used types for ergonomic usage
pub use {
    errors::{BuildError, BuildResult},
    yaml_processor::{YamlProcessor, ProviderInfo},
    code_generator::CodeGenerator,
    performance::PerformanceMonitor,
    fluent_ai_domain::model::ModelInfo,
    string_utils::sanitize_identifier,
    model_loader::{ModelLoader, ModelMetadata, ExistingModelRegistry},
    change_detector::{ChangeDetector, ModelChangeSet, ModelAddition, ModelModification, ModelDeletion, YamlModelInfo},
    yaml_manager::{YamlManager, YamlManagerBuilder, YamlDownloadResult, YamlCacheMetadata},
    incremental_generator::{IncrementalGenerator, IncrementalGeneratorBuilder, GenerationResult},
};

// Removed MAX_BUFFER_SIZE - not imported/used in build module context

// Removed MAX_CONCURRENT_OPS - not used anywhere in codebase

// Removed CACHE_LINE_SIZE - not imported/used in build module context

// Removed CacheAligned struct - not used anywhere in codebase


