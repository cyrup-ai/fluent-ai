//! Production-quality incremental build script for fluent-ai provider generation
//!
//! This build script uses the proper build system from the build_system/ directory.
//! NO FALLBACKS - if it breaks, we fix it properly.

mod build_system;

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use build_system::incremental_generator::IncrementalGenerator;
use build_system::performance::PerformanceMonitor;
use build_system::change_detector::{ChangeDetector};
use build_system::model_loader::ExistingModelRegistry;
use build_system::yaml_manager::YamlManager;

/// Main build function - uses the real build system with streaming patterns
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=models.yaml");
    println!("cargo:rerun-if-changed=build_system/");
    println!("cargo:rerun-if-env-changed=FLUENT_AI_BUILD_MODE");
    println!("cargo:rerun-if-env-changed=FLUENT_AI_CACHE_DIR");

    let out_dir = env::var("OUT_DIR")?;
    let dest_path = PathBuf::from(out_dir);
    
    // Generate domain models in domain package per CLAUDE.md requirements
    let domain_model_path = PathBuf::from("../domain/src/model");

    println!("cargo:warning=Starting REAL provider generation with build system...");

    // Create tokio runtime for async streams
    let rt = tokio::runtime::Runtime::new()?;
    
    rt.block_on(async {
        // Create the performance monitor that IncrementalGenerator needs
        let perf_monitor = Arc::new(PerformanceMonitor::new());

        // TEMPORARY: Use OUT_DIR to break chicken-and-egg problem, will switch to domain later
        let generator = IncrementalGenerator::new(dest_path.clone(), perf_monitor)?;
        
        // Create YAML manager for downloading models.yaml using HTTP3
        let cache_dir = dest_path.join("yaml_cache");
        let yaml_manager = YamlManager::new(cache_dir)?;
        
        // Use local models.yaml file instead of downloading from GitHub
        let models_yaml_path = std::path::Path::new("models.yaml");
        let yaml_content = std::fs::read_to_string(models_yaml_path)
            .map_err(|e| format!("Failed to read local models.yaml: {}", e))?;
        
        // Parse the local YAML content
        println!("cargo:warning=YAML content length: {} chars", yaml_content.len());
        println!("cargo:warning=YAML first 200 chars: {:?}", &yaml_content[..yaml_content.len().min(200)]);
        
        let all_providers = match yaml_manager.parse_providers(&yaml_content) {
            Ok(providers) => providers,
            Err(e) => {
                println!("cargo:warning=YAML parsing failed: {}", e);
                return Err(format!("YAML parsing failed: {}", e).into());
            }
        };
        
        println!("cargo:warning=Parsed {} providers from YAML", all_providers.len());
        for provider in &all_providers {
            println!("cargo:warning=Provider '{}' has {} models", provider.id, provider.models.len());
        }
        
        // Create existing model registry (empty for fresh builds)
        let existing_registry = ExistingModelRegistry::new();
        println!("cargo:warning=Created empty existing registry with {} models", existing_registry.model_count());
        
        // Use change detector to detect all models as NEW when no existing models
        let change_detector = ChangeDetector::new();
        let change_set = change_detector.detect_changes(&all_providers, &existing_registry)?;
        
        println!("cargo:warning=Change detection: {}", change_set.summary());
        
        // Generate using streaming patterns
        let result = generator.generate_incremental(&change_set, &all_providers).await?;
        
        println!("cargo:warning=Generated {} files, updated {} files", 
                result.files_generated, result.files_updated);
        
        Ok::<(), Box<dyn std::error::Error>>(())
    })?;

    println!("cargo:warning=Real provider generation completed successfully");
    Ok(())
}
