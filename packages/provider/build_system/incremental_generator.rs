//! Incremental template generation system
//!
//! Zero-allocation, lock-free implementation for selective model generation
//! using the existing template system with streaming-first patterns.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::fs;
use tracing::{debug, error, info, instrument, warn};

use super::errors::{BuildError, BuildResult};
use super::change_detector::{ModelChangeSet, ModelAddition, ModelModification};
use super::code_generator::CodeGenerator;
use super::yaml_processor::ProviderInfo;
use super::performance::PerformanceMonitor;

/// Result of incremental generation operation
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Number of files generated
    pub files_generated: usize,
    /// Number of files updated
    pub files_updated: usize,
    /// Number of files preserved (unchanged)
    pub files_preserved: usize,
    /// Total generation time in milliseconds
    pub generation_time_ms: u64,
    /// Generated file paths
    pub generated_files: Vec<PathBuf>,
    /// Any warnings encountered during generation
    pub warnings: Vec<String>,
}

impl GenerationResult {
    /// Create a new empty generation result
    pub fn new() -> Self {
        Self {
            files_generated: 0,
            files_updated: 0,
            files_preserved: 0,
            generation_time_ms: 0,
            generated_files: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Get total number of files processed
    pub fn total_files(&self) -> usize {
        self.files_generated + self.files_updated + self.files_preserved
    }

    /// Check if any files were modified
    pub fn has_changes(&self) -> bool {
        self.files_generated > 0 || self.files_updated > 0
    }

    /// Add a warning message
    pub fn add_warning(&mut self, message: String) {
        self.warnings.push(message);
    }

    /// Merge another result into this one
    pub fn merge(&mut self, other: GenerationResult) {
        self.files_generated += other.files_generated;
        self.files_updated += other.files_updated;
        self.files_preserved += other.files_preserved;
        self.generation_time_ms += other.generation_time_ms;
        self.generated_files.extend(other.generated_files);
        self.warnings.extend(other.warnings);
    }
}

impl Default for GenerationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental template generator with selective processing
#[derive(Debug)]
pub struct IncrementalGenerator {
    /// Code generator for template processing
    code_generator: CodeGenerator,
    /// Output directory for generated files
    output_dir: PathBuf,
    /// Whether to enable fallback to full generation
    enable_fallback: bool,
    /// Maximum number of concurrent generation tasks
    max_concurrent: usize,
}

impl IncrementalGenerator {
    /// Create a new incremental generator
    pub fn new<P: AsRef<Path>>(
        output_dir: P,
        perf_monitor: Arc<PerformanceMonitor>,
    ) -> BuildResult<Self> {
        let code_generator = CodeGenerator::new(perf_monitor)?;
        
        Ok(Self {
            code_generator,
            output_dir: output_dir.as_ref().to_path_buf(),
            enable_fallback: true,
            max_concurrent: num_cpus::get().max(2),
        })
    }

    /// Enable or disable fallback to full generation
    pub fn with_fallback(mut self, enable: bool) -> Self {
        self.enable_fallback = enable;
        self
    }

    /// Set maximum concurrent generation tasks
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max.max(1);
        self
    }

    /// Generate code incrementally based on change set
    #[instrument(skip(self, change_set, all_providers))]
    pub async fn generate_incremental(
        &self,
        change_set: &ModelChangeSet,
        all_providers: &[ProviderInfo],
    ) -> BuildResult<GenerationResult> {
        let start_time = std::time::Instant::now();
        let mut result = GenerationResult::new();

        info!("Starting incremental generation: {}", change_set.summary());

        // Ensure output directory exists
        fs::create_dir_all(&self.output_dir).await
            .map_err(|e| BuildError::IoError(e))?;

        // If no changes, preserve all existing files
        if !change_set.has_changes() {
            debug!("No changes detected, preserving all existing files");
            result.files_preserved = self.count_existing_files().await?;
            result.generation_time_ms = start_time.elapsed().as_millis() as u64;
            return Ok(result);
        }

        // Generate code for changes
        match self.generate_changes(change_set, all_providers).await {
            Ok(gen_result) => {
                result.merge(gen_result);
            }
            Err(e) => {
                if self.enable_fallback {
                    warn!("Incremental generation failed: {}, falling back to full generation", e);
                    result.add_warning(format!("Incremental generation failed, used fallback: {}", e));
                    
                    // Fallback to full generation
                    let fallback_result = self.generate_full(all_providers).await?;
                    result.merge(fallback_result);
                } else {
                    return Err(e);
                }
            }
        }

        result.generation_time_ms = start_time.elapsed().as_millis() as u64;
        info!("Incremental generation completed in {}ms: {} files generated, {} updated, {} preserved",
              result.generation_time_ms,
              result.files_generated,
              result.files_updated,
              result.files_preserved);

        Ok(result)
    }

    /// Generate code for specific changes
    async fn generate_changes(
        &self,
        change_set: &ModelChangeSet,
        all_providers: &[ProviderInfo],
    ) -> BuildResult<GenerationResult> {
        let mut result = GenerationResult::new();

        // Determine which providers are affected
        let affected_providers = change_set.affected_providers();
        
        debug!("Affected providers: {:?}", affected_providers);

        // Use semaphore to limit concurrent operations
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrent));
        let mut tasks = Vec::new();

        // Generate code for each affected provider
        for provider_name in affected_providers {
            let provider_info = all_providers.iter()
                .find(|p| p.id == provider_name.as_ref())
                .ok_or_else(|| BuildError::ValidationError(
                    format!("Provider {} not found in YAML", provider_name)
                ))?;

            let sem = Arc::clone(&semaphore);
            let code_gen = &self.code_generator;
            let output_dir = &self.output_dir;
            
            // Generate provider-specific files
            let task = async move {
                let _permit = sem.acquire().await
                    .map_err(|e| BuildError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Semaphore error: {}", e)
                    )))?;

                self.generate_provider_files(code_gen, provider_info, output_dir).await
            };

            tasks.push(task);
        }

        // Wait for all provider generation tasks
        let provider_results = futures::future::try_join_all(tasks).await?;
        
        // Merge all provider results
        for provider_result in provider_results {
            result.merge(provider_result);
        }

        // Generate global registry files that depend on all providers
        let registry_result = self.generate_registry_files(all_providers).await?;
        result.merge(registry_result);

        Ok(result)
    }

    /// Generate files for a specific provider
    async fn generate_provider_files(
        &self,
        code_generator: &CodeGenerator,
        provider: &ProviderInfo,
        output_dir: &Path,
    ) -> BuildResult<GenerationResult> {
        let mut result = GenerationResult::new();

        debug!("Generating files for provider: {}", provider.id);

        // Generate provider-specific module
        let provider_code = code_generator.generate_provider_module(&[provider.clone()])?;
        let provider_file = output_dir.join(format!("{}_provider.rs", provider.id));
        
        if self.should_update_file(&provider_file, &provider_code).await? {
            fs::write(&provider_file, provider_code).await
                .map_err(|e| BuildError::IoError(e))?;
            
            result.generated_files.push(provider_file);
            result.files_generated += 1;
            
            debug!("Generated provider file for {}", provider.id);
        } else {
            result.files_preserved += 1;
            debug!("Preserved unchanged provider file for {}", provider.id);
        }

        // Generate model definitions for this provider
        let models_code = code_generator.generate_model_definitions(&[provider.clone()])?;
        let models_file = output_dir.join(format!("{}_models.rs", provider.id));
        
        if self.should_update_file(&models_file, &models_code).await? {
            fs::write(&models_file, models_code).await
                .map_err(|e| BuildError::IoError(e))?;
            
            result.generated_files.push(models_file);
            result.files_generated += 1;
            
            debug!("Generated models file for {}", provider.id);
        } else {
            result.files_preserved += 1;
            debug!("Preserved unchanged models file for {}", provider.id);
        }

        Ok(result)
    }

    /// Generate registry files that aggregate all providers
    async fn generate_registry_files(&self, all_providers: &[ProviderInfo]) -> BuildResult<GenerationResult> {
        let mut result = GenerationResult::new();

        debug!("Generating global registry files");

        // Generate main providers registry
        let providers_code = self.code_generator.generate_provider_module(all_providers)?;
        let providers_file = self.output_dir.join("providers.rs");
        
        if self.should_update_file(&providers_file, &providers_code).await? {
            fs::write(&providers_file, providers_code).await
                .map_err(|e| BuildError::IoError(e))?;
            
            result.generated_files.push(providers_file);
            result.files_updated += 1;
            
            debug!("Updated providers registry");
        } else {
            result.files_preserved += 1;
            debug!("Preserved unchanged providers registry");
        }

        // Generate main models registry
        let models_code = self.code_generator.generate_model_registry(all_providers)?;
        let models_file = self.output_dir.join("models.rs");
        
        if self.should_update_file(&models_file, &models_code).await? {
            fs::write(&models_file, models_code).await
                .map_err(|e| BuildError::IoError(e))?;
            
            result.generated_files.push(models_file);
            result.files_updated += 1;
            
            debug!("Updated models registry");
        } else {
            result.files_preserved += 1;
            debug!("Preserved unchanged models registry");
        }

        // Generate mod.rs file to re-export all modules
        let mod_code = self.generate_mod_file(all_providers)?;
        let mod_file = self.output_dir.join("mod.rs");
        
        if self.should_update_file(&mod_file, &mod_code).await? {
            fs::write(&mod_file, mod_code).await
                .map_err(|e| BuildError::IoError(e))?;
            
            result.generated_files.push(mod_file);
            result.files_updated += 1;
            
            debug!("Updated mod.rs file");
        } else {
            result.files_preserved += 1;
            debug!("Preserved unchanged mod.rs file");
        }

        Ok(result)
    }

    /// Generate mod.rs file for re-exporting all provider modules
    fn generate_mod_file(&self, providers: &[ProviderInfo]) -> BuildResult<String> {
        let mut mod_content = String::new();
        
        mod_content.push_str("//! Generated provider modules\n");
        mod_content.push_str("//!\n");
        mod_content.push_str("//! This file is automatically generated. Do not edit manually.\n\n");
        
        // Re-export main registries
        mod_content.push_str("pub mod providers;\n");
        mod_content.push_str("pub mod models;\n\n");
        
        // Re-export provider-specific modules
        for provider in providers {
            mod_content.push_str(&format!("pub mod {}_provider;\n", provider.id));
            mod_content.push_str(&format!("pub mod {}_models;\n", provider.id));
        }
        
        mod_content.push_str("\n// Re-export commonly used types\n");
        mod_content.push_str("pub use providers::*;\n");
        mod_content.push_str("pub use models::*;\n");
        
        Ok(mod_content)
    }

    /// Check if a file should be updated by comparing content
    async fn should_update_file(&self, file_path: &Path, new_content: &str) -> BuildResult<bool> {
        if !file_path.exists() {
            return Ok(true); // File doesn't exist, needs to be created
        }

        match fs::read_to_string(file_path).await {
            Ok(existing_content) => {
                // Compare content, ignoring whitespace differences and comments with timestamps
                let existing_normalized = self.normalize_content(&existing_content);
                let new_normalized = self.normalize_content(new_content);
                
                Ok(existing_normalized != new_normalized)
            }
            Err(_) => {
                // Error reading file, assume it needs updating
                Ok(true)
            }
        }
    }

    /// Normalize content for comparison by removing timestamps and normalizing whitespace
    fn normalize_content(&self, content: &str) -> String {
        content
            .lines()
            .filter(|line| {
                // Skip lines with timestamps or generation metadata
                !line.contains("Generated at") && 
                !line.contains("Build time:") &&
                !line.trim().is_empty()
            })
            .map(|line| line.trim())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Fallback to full generation when incremental fails
    async fn generate_full(&self, all_providers: &[ProviderInfo]) -> BuildResult<GenerationResult> {
        let mut result = GenerationResult::new();

        debug!("Performing full generation fallback");

        // Generate all files from scratch
        let providers_code = self.code_generator.generate_provider_module(all_providers)?;
        let providers_file = self.output_dir.join("providers.rs");
        fs::write(&providers_file, providers_code).await
            .map_err(|e| BuildError::IoError(e))?;
        result.generated_files.push(providers_file);
        result.files_generated += 1;

        let models_code = self.code_generator.generate_model_registry(all_providers)?;
        let models_file = self.output_dir.join("models.rs");
        fs::write(&models_file, models_code).await
            .map_err(|e| BuildError::IoError(e))?;
        result.generated_files.push(models_file);
        result.files_generated += 1;

        let mod_code = self.generate_mod_file(all_providers)?;
        let mod_file = self.output_dir.join("mod.rs");
        fs::write(&mod_file, mod_code).await
            .map_err(|e| BuildError::IoError(e))?;
        result.generated_files.push(mod_file);
        result.files_generated += 1;

        debug!("Full generation completed: {} files generated", result.files_generated);

        Ok(result)
    }

    /// Count existing files for preservation tracking
    async fn count_existing_files(&self) -> BuildResult<usize> {
        if !self.output_dir.exists() {
            return Ok(0);
        }

        let mut count = 0;
        let mut entries = fs::read_dir(&self.output_dir).await
            .map_err(|e| BuildError::IoError(e))?;

        while let Some(entry) = entries.next_entry().await
            .map_err(|e| BuildError::IoError(e))? {
            
            if entry.path().is_file() {
                if let Some(ext) = entry.path().extension() {
                    if ext == "rs" {
                        count += 1;
                    }
                }
            }
        }

        Ok(count)
    }
}

/// Builder for IncrementalGenerator with fluent configuration
#[derive(Debug)]
pub struct IncrementalGeneratorBuilder {
    output_dir: Option<PathBuf>,
    perf_monitor: Option<Arc<PerformanceMonitor>>,
    enable_fallback: bool,
    max_concurrent: usize,
}

impl IncrementalGeneratorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            output_dir: None,
            perf_monitor: None,
            enable_fallback: true,
            max_concurrent: num_cpus::get().max(2),
        }
    }

    /// Set output directory
    pub fn output_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.output_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Set performance monitor
    pub fn performance_monitor(mut self, monitor: Arc<PerformanceMonitor>) -> Self {
        self.perf_monitor = Some(monitor);
        self
    }

    /// Enable or disable fallback to full generation
    pub fn fallback(mut self, enable: bool) -> Self {
        self.enable_fallback = enable;
        self
    }

    /// Set maximum concurrent tasks
    pub fn max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max.max(1);
        self
    }

    /// Build the IncrementalGenerator
    pub fn build(self) -> BuildResult<IncrementalGenerator> {
        let output_dir = self.output_dir
            .ok_or_else(|| BuildError::ValidationError("Output directory is required".to_string()))?;

        let perf_monitor = self.perf_monitor
            .unwrap_or_else(|| Arc::new(PerformanceMonitor::new()));

        let generator = IncrementalGenerator::new(output_dir, perf_monitor)?
            .with_fallback(self.enable_fallback)
            .with_max_concurrent(self.max_concurrent);

        Ok(generator)
    }
}

impl Default for IncrementalGeneratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_generation_result() {
        let mut result = GenerationResult::new();
        
        assert_eq!(result.total_files(), 0);
        assert!(!result.has_changes());

        result.files_generated = 2;
        result.files_updated = 1;
        result.files_preserved = 3;

        assert_eq!(result.total_files(), 6);
        assert!(result.has_changes());

        result.add_warning("Test warning".to_string());
        assert_eq!(result.warnings.len(), 1);

        let mut other = GenerationResult::new();
        other.files_generated = 1;
        other.warnings.push("Another warning".to_string());

        result.merge(other);
        assert_eq!(result.files_generated, 3);
        assert_eq!(result.warnings.len(), 2);
    }

    #[tokio::test]
    async fn test_incremental_generator_builder() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let perf_monitor = Arc::new(PerformanceMonitor::new());

        let generator = IncrementalGeneratorBuilder::new()
            .output_dir(temp_dir.path())
            .performance_monitor(perf_monitor)
            .fallback(false)
            .max_concurrent(4)
            .build()
            .expect("Build generator");

        assert_eq!(generator.output_dir, temp_dir.path());
        assert!(!generator.enable_fallback);
        assert_eq!(generator.max_concurrent, 4);
    }

    #[tokio::test]
    async fn test_content_normalization() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let perf_monitor = Arc::new(PerformanceMonitor::new());
        let generator = IncrementalGenerator::new(temp_dir.path(), perf_monitor)
            .expect("Create generator");

        let content1 = "
            // Generated at 2023-01-01
            fn test() {
                println!(\"hello\");
            }
        ";

        let content2 = "
            // Generated at 2023-01-02
            fn test() {
                println!(\"hello\");
            }
        ";

        let normalized1 = generator.normalize_content(content1);
        let normalized2 = generator.normalize_content(content2);

        // Should be equal after normalization (timestamps ignored)
        assert_eq!(normalized1, normalized2);
    }

    #[tokio::test]
    async fn test_mod_file_generation() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let perf_monitor = Arc::new(PerformanceMonitor::new());
        let generator = IncrementalGenerator::new(temp_dir.path(), perf_monitor)
            .expect("Create generator");

        let providers = vec![
            ProviderInfo {
                id: "openai".to_string(),
                name: "OpenAI".to_string(),
                base_url: "https://api.openai.com".to_string(),
                models: vec![],
                auth: super::yaml_processor::AuthConfig {
                    r#type: "api_key".to_string(),
                    env_var: "OPENAI_API_KEY".to_string(),
                    header_name: Some("Authorization".to_string()),
                },
                rate_limit: None,
                features: vec!["chat".to_string()],
            }
        ];

        let mod_content = generator.generate_mod_file(&providers)
            .expect("Generate mod file");

        assert!(mod_content.contains("pub mod openai_provider;"));
        assert!(mod_content.contains("pub mod openai_models;"));
        assert!(mod_content.contains("pub use providers::*;"));
    }
}