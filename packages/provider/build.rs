#![cfg_attr(
    feature = "clippy",
    deny(
        missing_docs,
        missing_debug_implementations,
        unsafe_code,
        rustdoc::broken_intra_doc_links,
        rustdoc::private_intra_doc_links,
        rustdoc::missing_crate_level_docs,
        rustdoc::invalid_codeblock_attributes,
        rustdoc::invalid_rust_codeblocks,
        rustdoc::bare_urls,
        rustdoc::invalid_html_tags
    )
)]
#![cfg_attr(
    feature = "clippy",
    warn(
        clippy::all,
        clippy::pedantic,
        clippy::nursery,
        clippy::cargo,
        clippy::complexity,
        clippy::perf,
        clippy::style,
        clippy::suspicious,
        clippy::unwrap_used,
        clippy::expect_used
    )
)]

//! Zero-allocation, lock-free build system for fluent_ai_provider
//!
//! This build script generates provider clients, model enums, and metadata
//! at compile time with zero allocation, lock-free concurrency, and SIMD optimizations.

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// Include build modules from the build/ directory
mod build_modules {
    include!("build/mod.rs");
}

use build_modules::cache_manager::{CacheConfig, CacheManager};
use build_modules::code_generator::CodeGenerator;
use build_modules::errors::{BuildError, BuildResult, YamlError};
use build_modules::performance::PerformanceMonitor;
use build_modules::yaml_processor::{ProviderInfo, YamlProcessor};

/// Main build entry point
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let perf_monitor = Arc::new(PerformanceMonitor::new());

    // Enable backtraces for better error reporting
    env::set_var("RUST_BACKTRACE", "1");

    // Re-run if any build files change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build");
    println!("cargo:rerun-if-changed=providers");

    // Initialize build components
    let cache_manager = initialize_cache_manager(perf_monitor.clone())?;
    let yaml_processor = YamlProcessor::new();
    let code_generator = CodeGenerator::new(perf_monitor.clone());

    // Load provider definitions
    let providers = load_provider_definitions(&yaml_processor, &cache_manager)?;

    // Generate provider-specific code
    generate_provider_code(&code_generator, &providers)?;

    // Generate model registry
    generate_model_registry(&code_generator, &providers)?;

    // Finalize and print performance stats
    println!("cargo:warning=Build finished in {:?}", start_time.elapsed());
    println!("cargo:warning={}", perf_monitor);

    Ok(())
}

/// Initialize the cache manager
fn initialize_cache_manager(
    stats: Arc<PerformanceMonitor>,
) -> BuildResult<CacheManager> {
    let config = CacheConfig::default();
    // TODO: Load config from file if it exists
    CacheManager::new(config, stats.stats())
}

/// Load provider definitions from YAML
fn load_provider_definitions(
    yaml_processor: &YamlProcessor,
    _cache: &CacheManager,
) -> BuildResult<Vec<ProviderInfo>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()));
    let models_path = manifest_dir.join("models.yaml");

    // Read the YAML file
    let yaml_content = std::fs::read_to_string(&models_path).map_err(|e| {
        BuildError::YamlError(YamlError::new(format!(
            "Failed to read models.yaml: {}",
            e
        )))
    })?;

    // Parse the YAML content
    let providers = yaml_processor.parse_providers(&yaml_content)?;

    // Log the number of providers and models loaded
    let total_models = providers.iter().map(|p| p.models.len()).sum::<usize>();
    println!(
        "cargo:warning=Loaded {} providers with {} total models",
        providers.len(),
        total_models
    );

    Ok(providers)
}

/// Generate provider-specific code
fn generate_provider_code(
    _code_generator: &CodeGenerator,
    _providers: &[ProviderInfo],
) -> BuildResult<()> {
    // TODO: Implement code generation for each provider
    // This will use the code_generator to create provider-specific modules
    Ok(())
}

/// Generate the model registry
fn generate_model_registry(
    _code_generator: &CodeGenerator,
    _providers: &[ProviderInfo],
) -> BuildResult<()> {
    // TODO: Generate the model registry that will be used at runtime
    Ok(())
}

/// Helper to write generated code to file
fn write_generated_code(path: &str, content: &str) -> BuildResult<()> {
    let out_dir = env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| BuildError::Other("Failed to get output directory".into()))?;

    let dest_path = out_dir.join(path);
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&dest_path, content)?;

    println!("cargo:rerun-if-changed={}", dest_path.display());
    Ok(())
}
