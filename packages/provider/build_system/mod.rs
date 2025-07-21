// Zero-allocation build system modules for fluent_ai_provider
//
// This module contains all the build system components with zero-allocation,
// lock-free, and SIMD-optimized implementations.


pub mod errors;
pub mod http_client;
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
    http_client::HttpClient,
    yaml_processor::{YamlProcessor, ProviderInfo, ModelInfo},
    code_generator::CodeGenerator,
    performance::PerformanceMonitor,
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

/// Helper macro for zero-allocation string formatting
#[macro_export]
macro_rules! format_inline {
    ($($arg:tt)*) => {{
        let mut buf = arrayvec::ArrayString::<{ $crate::build_modules::MAX_BUFFER_SIZE }>::new();
        use std::fmt::Write;
        write!(&mut buf, $($arg)*).expect("Failed to write to ArrayString");
        buf
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_alignment() {
        let aligned = CacheAligned(42u64);
        let ptr = &aligned as *const _ as usize;
        assert_eq!(ptr % 64, 0, "CacheAligned must be 64-byte aligned");
    }

    #[test]
    fn test_format_inline() {
        let s = format_inline!("Hello, {}!", "world");
        assert_eq!(s, "Hello, world!");
    }
}
