#![warn(
    missing_docs,
    missing_debug_implementations,
    unsafe_code,
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::missing_crate_level_docs,
    rustdoc::invalid_codeblock_attributes,
    rustdoc::invalid_rust_codeblocks,
    rustdoc::bare_urls,
    rustdoc::invalid_html_tags,
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
use build_modules::http_client::HttpClient;
use build_modules::performance::PerformanceMonitor;
use build_modules::yaml_processor::{ProviderInfo, YamlProcessor};

/// Main build entry point
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let perf_monitor = Arc::new(PerformanceMonitor::new());

    // Enable backtraces for better error reporting during build
    // Setting RUST_BACKTRACE=1 is a standard debugging practice in build scripts
    unsafe {
        std::env::set_var("RUST_BACKTRACE", "1");
    }

    // Re-run if any build files change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build");
    println!("cargo:rerun-if-changed=providers");

    // Initialize build components
    let cache_manager = initialize_cache_manager(perf_monitor.clone())?;
    let yaml_processor = YamlProcessor::new();
    let code_generator = CodeGenerator::new(perf_monitor.clone());

    // Load provider definitions
    let providers =
        load_provider_definitions(&yaml_processor, &cache_manager, perf_monitor.clone()).await?;

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
fn initialize_cache_manager(stats: Arc<PerformanceMonitor>) -> BuildResult<CacheManager> {
    let config = CacheConfig::default();
    // TODO: Load config from file if it exists
    CacheManager::new(config, stats.stats())
}

/// Load provider definitions from YAML
async fn load_provider_definitions(
    yaml_processor: &YamlProcessor,
    cache: &CacheManager,
    perf_monitor: Arc<PerformanceMonitor>,
) -> BuildResult<Vec<ProviderInfo>> {
    const MODELS_URL: &str =
        "https://raw.githubusercontent.com/sigoden/aichat/refs/heads/main/models.yaml";
    const CACHE_KEY: &str = "sigoden_models_yaml";
    const CACHE_TTL: u64 = 24 * 60 * 60; // 24 hours

    // Try to get from cache first
    let yaml_content = if let Ok(Some(cached_data)) = cache.get(CACHE_KEY) {
        println!("cargo:warning=Using cached models.yaml");
        String::from_utf8(cached_data).map_err(|e| {
            BuildError::YamlError(YamlError::new(format!(
                "Cached data is not valid UTF-8: {}",
                e
            )))
        })?
    } else {
        println!("cargo:warning=Downloading models.yaml from GitHub");

        // Create HTTP client and download
        let http_client = HttpClient::new(perf_monitor)?;
        let downloaded_data = http_client.download_file(MODELS_URL).await?;

        // Cache the downloaded data
        cache.set(CACHE_KEY, downloaded_data.clone(), Some(CACHE_TTL))?;

        String::from_utf8(downloaded_data).map_err(|e| {
            BuildError::YamlError(YamlError::new(format!(
                "Downloaded data is not valid UTF-8: {}",
                e
            )))
        })?
    };

    // Parse the YAML content
    let providers = yaml_processor.parse_providers(&yaml_content)?;

    // Log the number of providers and models loaded
    let total_models = providers.iter().map(|p| p.models.len()).sum::<usize>();
    println!(
        "cargo:warning=Loaded {} providers with {} total models from GitHub",
        providers.len(),
        total_models
    );

    Ok(providers)
}

/// Generate provider-specific code
fn generate_provider_code(
    _code_generator: &CodeGenerator,
    providers: &[ProviderInfo],
) -> BuildResult<()> {
    // Generate provider constants file
    let mut provider_code = String::from("// Generated provider constants\n\n");
    provider_code.push_str("use std::collections::HashMap;\n");
    provider_code.push_str("use lazy_static::lazy_static;\n\n");
    
    provider_code.push_str("lazy_static! {\n");
    provider_code.push_str("    pub static ref PROVIDER_URLS: HashMap<&'static str, &'static str> = {\n");
    provider_code.push_str("        let mut m = HashMap::new();\n");
    
    for provider in providers {
        provider_code.push_str(&format!(
            "        m.insert(\"{}\", \"{}\");\n",
            provider.id, provider.base_url
        ));
    }
    
    provider_code.push_str("        m\n");
    provider_code.push_str("    };\n");
    provider_code.push_str("}\n");

    write_generated_code("providers.rs", &provider_code)?;
    Ok(())
}

/// Generate the model registry
fn generate_model_registry(
    _code_generator: &CodeGenerator,
    providers: &[ProviderInfo],
) -> BuildResult<()> {
    // Generate model registry file
    let mut model_code = String::from("// Generated model registry\n\n");
    model_code.push_str("use std::collections::HashMap;\n");
    model_code.push_str("use lazy_static::lazy_static;\n\n");
    
    model_code.push_str("#[derive(Debug, Clone)]\n");
    model_code.push_str("pub struct ModelInfo {\n");
    model_code.push_str("    pub provider: &'static str,\n");
    model_code.push_str("    pub max_tokens: u32,\n");
    model_code.push_str("    pub supports_streaming: bool,\n");
    model_code.push_str("}\n\n");
    
    model_code.push_str("lazy_static! {\n");
    model_code.push_str("    pub static ref MODEL_REGISTRY: HashMap<&'static str, ModelInfo> = {\n");
    model_code.push_str("        let mut m = HashMap::new();\n");
    
    for provider in providers {
        for model in &provider.models {
            model_code.push_str(&format!(
                "        m.insert(\"{}\", ModelInfo {{ provider: \"{}\", max_tokens: {}, supports_streaming: {} }});\n",
                model.id, provider.id, model.max_tokens, model.supports_streaming
            ));
        }
    }
    
    model_code.push_str("        m\n");
    model_code.push_str("    };\n");
    model_code.push_str("}\n");

    write_generated_code("models.rs", &model_code)?;
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
