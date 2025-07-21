//! Build script with incremental model generation system
//!
//! Zero-allocation, lock-free build system that uses HTTP3 conditional requests,
//! parallel model loading, and incremental change detection for blazing-fast builds.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

// Import the build_system modules
mod build_system;
use build_system::{
    YamlProcessor, CodeGenerator, PerformanceMonitor, BuildResult, BuildError,
    ModelLoader, ChangeDetector, YamlManager, YamlManagerBuilder,
    IncrementalGenerator, IncrementalGeneratorBuilder,
};

/// Main build function with incremental generation
#[tokio::main]
async fn main() -> BuildResult<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=models.yaml");
    println!("cargo:rerun-if-changed=build_system/");

    let dest_path = PathBuf::from("src/generated");
    if !dest_path.exists() {
        fs::create_dir_all(&dest_path)
            .map_err(|e| BuildError::IoError(e))?;
    }

    // Initialize performance monitoring
    let perf_monitor = Arc::new(PerformanceMonitor::new());

    println!("cargo:warning=Starting incremental model generation...");

    // Try incremental generation first, fallback to legacy on failure
    match run_incremental_generation(&dest_path, Arc::clone(&perf_monitor)).await {
        Ok(()) => {
            println!("cargo:warning=Incremental generation completed successfully");
        }
        Err(e) => {
            println!("cargo:warning=Incremental generation failed: {}, falling back to legacy", e);
            run_legacy_generation(&dest_path, perf_monitor)?;
        }
    }

    Ok(())
}

/// Run the new incremental generation system
async fn run_incremental_generation(
    dest_path: &PathBuf,
    perf_monitor: Arc<PerformanceMonitor>,
) -> BuildResult<()> {
    let generation_start = std::time::Instant::now();

    // Step 1: Load existing models from filesystem in parallel
    println!("cargo:warning=Loading existing models from filesystem...");
    let clients_path = PathBuf::from("src/clients");
    let model_loader = ModelLoader::new(&clients_path)
        .with_max_concurrent(num_cpus::get().max(4));
    
    let existing_registry = model_loader.load_existing_models().await?;
    println!("cargo:warning=Loaded {} existing models from {} providers", 
             existing_registry.model_count(), 
             existing_registry.provider_count());

    // Step 2: Download YAML with HTTP3 conditional requests
    println!("cargo:warning=Downloading models.yaml with conditional requests...");
    let cache_dir = env::temp_dir().join("fluent-ai-cache");
    let yaml_manager = YamlManagerBuilder::new()
        .cache_dir(&cache_dir)
        .default_expires(24 * 60 * 60) // 24 hours
        .build()?;

    // Try remote YAML first, fallback to local file
    let yaml_result = match env::var("MODELS_YAML_URL") {
        Ok(url) => {
            println!("cargo:warning=Downloading from remote URL: {}", url);
            yaml_manager.download_yaml(&url).await?
        }
        Err(_) => {
            // No remote URL, use local file with caching metadata
            println!("cargo:warning=Using local models.yaml file");
            let local_content = fs::read_to_string("models.yaml")
                .map_err(|e| BuildError::IoError(e))?;
            
            // Create a synthetic download result for local file
            use build_system::{YamlDownloadResult, YamlCacheMetadata};
            YamlDownloadResult::Downloaded {
                content: local_content,
                metadata: YamlCacheMetadata::new(
                    "local://models.yaml".to_string(),
                    PathBuf::from("models.yaml"),
                ),
            }
        }
    };

    // Parse YAML content
    let yaml_content = yaml_result.content()
        .ok_or_else(|| BuildError::ValidationError("No YAML content available".to_string()))?;
    
    let providers = yaml_manager.parse_providers(yaml_content).await?;
    println!("cargo:warning=Parsed {} providers from YAML", providers.len());

    // Step 3: Detect changes between YAML and existing models
    println!("cargo:warning=Detecting changes...");
    let change_detector = ChangeDetector::new()
        .with_deletions(false) // Don't delete existing models for safety
        .with_deep_comparison(true);
    
    let change_set = change_detector.detect_changes(&providers, &existing_registry)?;
    println!("cargo:warning=Change detection: {}", change_set.summary());

    // Step 4: Generate code incrementally
    println!("cargo:warning=Generating code incrementally...");
    let generator = IncrementalGeneratorBuilder::new()
        .output_dir(dest_path)
        .performance_monitor(Arc::clone(&perf_monitor))
        .fallback(true) // Enable fallback to full generation
        .max_concurrent(num_cpus::get().max(2))
        .build()?;

    let generation_result = generator.generate_incremental(&change_set, &providers).await?;

    let total_time = generation_start.elapsed();
    println!("cargo:warning=Incremental generation completed in {}ms: {} files generated, {} updated, {} preserved",
             total_time.as_millis(),
             generation_result.files_generated,
             generation_result.files_updated,
             generation_result.files_preserved);

    // Report any warnings
    for warning in &generation_result.warnings {
        println!("cargo:warning={}", warning);
    }

    // Log performance metrics
    if generation_result.has_changes() {
        println!("cargo:warning=Generated files: {:?}", 
                 generation_result.generated_files.iter()
                     .map(|p| p.file_name().unwrap_or_default())
                     .collect::<Vec<_>>());
    }

    Ok(())
}

/// Fallback to legacy generation when incremental fails
fn run_legacy_generation(
    dest_path: &PathBuf,
    perf_monitor: Arc<PerformanceMonitor>,
) -> BuildResult<()> {
    println!("cargo:warning=Running legacy generation...");

    // Parse models.yaml using existing YAML processor
    let yaml_content = fs::read_to_string("models.yaml")
        .map_err(|e| BuildError::IoError(e))?;
    
    let yaml_processor = YamlProcessor::new();
    let providers = yaml_processor.parse_providers(&yaml_content)?;

    println!("cargo:warning=Parsed {} providers from models.yaml", providers.len());

    // Generate code using existing code generator
    let code_generator = CodeGenerator::new(perf_monitor)?;
    
    // Generate provider module
    let provider_code = code_generator.generate_provider_module(&providers)?;
    let provider_path = dest_path.join("providers.rs");
    fs::write(&provider_path, provider_code)
        .map_err(|e| BuildError::IoError(e))?;
    
    // Generate model registry
    let model_code = code_generator.generate_model_registry(&providers)?;
    let model_path = dest_path.join("models.rs");
    fs::write(&model_path, model_code)
        .map_err(|e| BuildError::IoError(e))?;

    println!("cargo:warning=Generated providers.rs and models.rs using legacy code generator");
    Ok(())
}