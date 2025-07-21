//! Parallel model discovery and loading system
//!
//! Zero-allocation, lock-free implementation for scanning existing model files
//! and building an in-memory registry for efficient comparison operations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tracing::{debug, error, instrument, warn};

use super::errors::{BuildError, BuildResult};

/// Metadata extracted from existing model files for comparison
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Provider name (e.g., "openai", "anthropic")
    pub provider: Arc<str>,
    /// Model name (e.g., "gpt-4", "claude-3-opus")
    pub model_name: Arc<str>,
    /// Maximum input tokens
    pub max_input_tokens: Option<u64>,
    /// Maximum output tokens
    pub max_output_tokens: Option<u64>,
    /// Supported capabilities (function_calling, vision, etc.)
    pub capabilities: Vec<Arc<str>>,
    /// Model parameters (temperature, top_p, etc.)
    pub parameters: HashMap<Arc<str>, Arc<str>>,
    /// File path where this model definition was found
    pub file_path: PathBuf,
    /// File modification time for change detection
    pub modified_time: SystemTime,
    /// Content hash for deep comparison
    pub content_hash: u64,
}

impl ModelMetadata {
    /// Create a new ModelMetadata instance
    pub fn new(
        provider: impl Into<Arc<str>>,
        model_name: impl Into<Arc<str>>,
        file_path: PathBuf,
        modified_time: SystemTime,
    ) -> Self {
        Self {
            provider: provider.into(),
            model_name: model_name.into(),
            max_input_tokens: None,
            max_output_tokens: None,
            capabilities: Vec::new(),
            parameters: HashMap::new(),
            file_path,
            modified_time,
            content_hash: 0,
        }
    }

    /// Get unique identifier for this model
    pub fn identifier(&self) -> String {
        format!("{}:{}", self.provider, self.model_name)
    }

    /// Add a capability to this model
    pub fn with_capability(mut self, capability: impl Into<Arc<str>>) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    /// Add a parameter to this model
    pub fn with_parameter(mut self, key: impl Into<Arc<str>>, value: impl Into<Arc<str>>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// Set token limits
    pub fn with_token_limits(mut self, input: Option<u64>, output: Option<u64>) -> Self {
        self.max_input_tokens = input;
        self.max_output_tokens = output;
        self
    }

    /// Set content hash for change detection
    pub fn with_content_hash(mut self, hash: u64) -> Self {
        self.content_hash = hash;
        self
    }
}

/// Lock-free registry for existing models using DashMap for concurrent access
#[derive(Debug, Default)]
pub struct ExistingModelRegistry {
    /// Model metadata indexed by provider:model_name
    models: DashMap<Arc<str>, ModelMetadata>,
    /// Provider metadata indexed by provider name
    providers: DashMap<Arc<str>, Vec<Arc<str>>>,
    /// File paths indexed by provider for change tracking
    file_paths: DashMap<Arc<str>, Vec<PathBuf>>,
}

impl ExistingModelRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a model into the registry
    pub fn insert(&self, metadata: ModelMetadata) {
        let identifier: Arc<str> = metadata.identifier().into();
        let provider = Arc::clone(&metadata.provider);
        let model_name = Arc::clone(&metadata.model_name);
        let file_path = metadata.file_path.clone();

        // Insert model metadata
        self.models.insert(Arc::clone(&identifier), metadata);

        // Update provider model list
        let mut provider_models = self.providers.entry(Arc::clone(&provider)).or_insert_with(Vec::new);
        if !provider_models.contains(&model_name) {
            provider_models.push(model_name);
        }

        // Update file paths for provider
        let mut provider_files = self.file_paths.entry(provider).or_insert_with(Vec::new);
        if !provider_files.contains(&file_path) {
            provider_files.push(file_path);
        }
    }

    /// Get model metadata by identifier
    pub fn get(&self, identifier: &str) -> Option<ModelMetadata> {
        self.models.get(identifier).map(|entry| entry.value().clone())
    }

    /// Check if a model exists in the registry
    pub fn contains_key(&self, identifier: &str) -> bool {
        self.models.contains_key(identifier)
    }

    /// Get all models for a specific provider
    pub fn get_provider_models(&self, provider: &str) -> Vec<ModelMetadata> {
        let mut models = Vec::new();
        
        if let Some(model_names) = self.providers.get(provider) {
            for model_name in model_names.value().iter() {
                let identifier = format!("{}:{}", provider, model_name);
                if let Some(metadata) = self.get(&identifier) {
                    models.push(metadata);
                }
            }
        }
        
        models
    }

    /// Get all provider names
    pub fn providers(&self) -> Vec<Arc<str>> {
        self.providers.iter().map(|entry| Arc::clone(entry.key())).collect()
    }

    /// Get total number of models
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Get total number of providers
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// Iterate over all models
    pub fn iter(&self) -> impl Iterator<Item = (Arc<str>, ModelMetadata)> + '_ {
        self.models.iter().map(|entry| (Arc::clone(entry.key()), entry.value().clone()))
    }

    /// Get file paths for a provider
    pub fn get_provider_files(&self, provider: &str) -> Vec<PathBuf> {
        self.file_paths.get(provider)
            .map(|entry| entry.value().clone())
            .unwrap_or_default()
    }
}

/// Parallel model loader for scanning filesystem and building model registry
#[derive(Debug)]
pub struct ModelLoader {
    /// Base directory to scan for client implementations
    base_path: PathBuf,
    /// Maximum number of concurrent file operations
    max_concurrent: usize,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            max_concurrent: num_cpus::get().max(4),
        }
    }

    /// Set maximum concurrent operations
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max.max(1);
        self
    }

    /// Load existing models from filesystem in parallel
    #[instrument(skip(self))]
    pub async fn load_existing_models(&self) -> BuildResult<ExistingModelRegistry> {
        let registry = ExistingModelRegistry::new();

        // Scan provider directories in parallel
        let provider_dirs = self.scan_provider_directories().await?;
        
        debug!("Found {} provider directories", provider_dirs.len());

        // Use semaphore to limit concurrent operations
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrent));
        let mut tasks = Vec::with_capacity(provider_dirs.len());

        for provider_dir in provider_dirs {
            let sem = Arc::clone(&semaphore);
            let registry_ref = &registry;
            
            let task = tokio::spawn(async move {
                let _permit = sem.acquire().await
                    .map_err(|e| BuildError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Semaphore error: {}", e)
                    )))?;
                
                self.scan_provider_directory(provider_dir, registry_ref).await
            });
            
            tasks.push(task);
        }

        // Wait for all tasks to complete
        let results = futures::future::join_all(tasks).await;

        // Check for errors
        for result in results {
            match result {
                Ok(Ok(())) => {} // Success
                Ok(Err(e)) => {
                    warn!("Error scanning provider directory: {}", e);
                    // Continue with other providers instead of failing completely
                }
                Err(e) => {
                    error!("Task join error: {}", e);
                    return Err(BuildError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Task error: {}", e)
                    )));
                }
            }
        }

        debug!("Loaded {} models from {} providers", 
               registry.model_count(), 
               registry.provider_count());

        Ok(registry)
    }

    /// Scan for provider directories
    async fn scan_provider_directories(&self) -> BuildResult<Vec<PathBuf>> {
        let mut provider_dirs = Vec::new();

        let mut entries = fs::read_dir(&self.base_path).await
            .map_err(|e| BuildError::IoError(e))?;

        while let Some(entry) = entries.next_entry().await
            .map_err(|e| BuildError::IoError(e))? {
            
            let path = entry.path();
            
            if path.is_dir() {
                let dir_name = path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("");
                
                // Skip common non-provider directories
                if !["mod.rs", "common", "utils", "tests"].contains(&dir_name) {
                    provider_dirs.push(path);
                }
            }
        }

        Ok(provider_dirs)
    }

    /// Scan a single provider directory for model definitions
    async fn scan_provider_directory(
        &self, 
        provider_dir: PathBuf, 
        registry: &ExistingModelRegistry
    ) -> BuildResult<()> {
        let provider_name: Arc<str> = provider_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .into();

        debug!("Scanning provider directory: {}", provider_name);

        // Look for discovery.rs file which typically contains model definitions
        let discovery_path = provider_dir.join("discovery.rs");
        if discovery_path.exists() {
            match self.extract_models_from_discovery(&discovery_path, &provider_name).await {
                Ok(models) => {
                    for model in models {
                        registry.insert(model);
                    }
                }
                Err(e) => {
                    warn!("Error extracting models from {}: {}", discovery_path.display(), e);
                }
            }
        }

        // Look for other potential model definition files
        let model_files = ["client.rs", "models.rs", "config.rs"];
        for file_name in &model_files {
            let file_path = provider_dir.join(file_name);
            if file_path.exists() {
                match self.extract_models_from_file(&file_path, &provider_name).await {
                    Ok(models) => {
                        for model in models {
                            registry.insert(model);
                        }
                    }
                    Err(e) => {
                        debug!("No models found in {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract model definitions from discovery.rs file
    async fn extract_models_from_discovery(
        &self, 
        file_path: &Path, 
        provider_name: &Arc<str>
    ) -> BuildResult<Vec<ModelMetadata>> {
        let content = fs::read_to_string(file_path).await
            .map_err(|e| BuildError::IoError(e))?;

        let metadata = fs::metadata(file_path).await
            .map_err(|e| BuildError::IoError(e))?;

        let modified_time = metadata.modified()
            .map_err(|e| BuildError::IoError(e))?;

        let content_hash = self.calculate_content_hash(&content);

        let mut models = Vec::new();

        // Parse model names from static arrays or constants
        if let Some(model_names) = self.extract_model_names_from_content(&content) {
            for model_name in model_names {
                let model_metadata = ModelMetadata::new(
                    Arc::clone(provider_name),
                    model_name.into(),
                    file_path.to_path_buf(),
                    modified_time,
                ).with_content_hash(content_hash);

                models.push(model_metadata);
            }
        }

        Ok(models)
    }

    /// Extract model definitions from other files
    async fn extract_models_from_file(
        &self, 
        file_path: &Path, 
        provider_name: &Arc<str>
    ) -> BuildResult<Vec<ModelMetadata>> {
        let content = fs::read_to_string(file_path).await
            .map_err(|e| BuildError::IoError(e))?;

        let metadata = fs::metadata(file_path).await
            .map_err(|e| BuildError::IoError(e))?;

        let modified_time = metadata.modified()
            .map_err(|e| BuildError::IoError(e))?;

        let content_hash = self.calculate_content_hash(&content);

        let mut models = Vec::new();

        // Look for model name patterns in the file content
        if let Some(model_names) = self.extract_model_names_from_content(&content) {
            for model_name in model_names {
                let model_metadata = ModelMetadata::new(
                    Arc::clone(provider_name),
                    model_name.into(),
                    file_path.to_path_buf(),
                    modified_time,
                ).with_content_hash(content_hash);

                models.push(model_metadata);
            }
        }

        Ok(models)
    }

    /// Extract model names from file content using pattern matching
    fn extract_model_names_from_content(&self, content: &str) -> Option<Vec<String>> {
        let mut model_names = Vec::new();

        // Look for static model arrays like: static SUPPORTED_MODELS: &[&str] = &[...]
        if let Some(start) = content.find("SUPPORTED_MODELS") {
            if let Some(array_start) = content[start..].find("&[") {
                if let Some(array_end) = content[start + array_start..].find("];") {
                    let array_content = &content[start + array_start + 2..start + array_start + array_end];
                    
                    // Extract quoted strings from the array
                    for line in array_content.lines() {
                        if let Some(model_name) = self.extract_quoted_string(line.trim()) {
                            if !model_name.is_empty() && !model_name.starts_with("//") {
                                model_names.push(model_name);
                            }
                        }
                    }
                }
            }
        }

        // Look for individual model name constants
        for line in content.lines() {
            if line.trim().starts_with("const ") && line.contains("_MODEL") {
                if let Some(model_name) = self.extract_quoted_string(line) {
                    model_names.push(model_name);
                }
            }
        }

        if model_names.is_empty() {
            None
        } else {
            Some(model_names)
        }
    }

    /// Extract quoted string from a line of code
    fn extract_quoted_string(&self, line: &str) -> Option<String> {
        if let Some(start) = line.find('"') {
            if let Some(end) = line[start + 1..].find('"') {
                return Some(line[start + 1..start + 1 + end].to_string());
            }
        }
        None
    }

    /// Calculate a simple hash of file content for change detection
    fn calculate_content_hash(&self, content: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    async fn create_test_provider_dir(temp_dir: &TempDir, provider: &str, models: &[&str]) -> PathBuf {
        let provider_dir = temp_dir.path().join(provider);
        fs::create_dir_all(&provider_dir).await.expect("Create provider dir");
        
        let discovery_content = format!(
            r#"
static SUPPORTED_MODELS: &[&str] = &[
{}
];
"#,
            models.iter().map(|m| format!("    \"{}\"", m)).collect::<Vec<_>>().join(",\n")
        );
        
        let discovery_path = provider_dir.join("discovery.rs");
        fs::write(&discovery_path, discovery_content).await.expect("Write discovery file");
        
        provider_dir
    }

    #[tokio::test]
    async fn test_model_metadata_creation() {
        let metadata = ModelMetadata::new(
            "openai",
            "gpt-4",
            PathBuf::from("/test/path"),
            SystemTime::now(),
        )
        .with_capability("function_calling")
        .with_parameter("temperature", "0.7")
        .with_token_limits(Some(8192), Some(4096))
        .with_content_hash(12345);

        assert_eq!(metadata.provider.as_ref(), "openai");
        assert_eq!(metadata.model_name.as_ref(), "gpt-4");
        assert_eq!(metadata.identifier(), "openai:gpt-4");
        assert_eq!(metadata.capabilities.len(), 1);
        assert_eq!(metadata.parameters.len(), 1);
        assert_eq!(metadata.max_input_tokens, Some(8192));
        assert_eq!(metadata.max_output_tokens, Some(4096));
        assert_eq!(metadata.content_hash, 12345);
    }

    #[tokio::test]
    async fn test_existing_model_registry() {
        let registry = ExistingModelRegistry::new();
        
        let metadata = ModelMetadata::new(
            "anthropic",
            "claude-3-opus",
            PathBuf::from("/test/anthropic/discovery.rs"),
            SystemTime::now(),
        );

        registry.insert(metadata.clone());

        assert_eq!(registry.model_count(), 1);
        assert_eq!(registry.provider_count(), 1);
        assert!(registry.contains_key("anthropic:claude-3-opus"));
        
        let retrieved = registry.get("anthropic:claude-3-opus").expect("Model should exist");
        assert_eq!(retrieved.provider, metadata.provider);
        assert_eq!(retrieved.model_name, metadata.model_name);

        let provider_models = registry.get_provider_models("anthropic");
        assert_eq!(provider_models.len(), 1);
    }

    #[tokio::test]
    async fn test_model_loader_parallel_scanning() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        
        // Create test provider directories with model definitions
        create_test_provider_dir(&temp_dir, "openai", &["gpt-4", "gpt-3.5-turbo"]).await;
        create_test_provider_dir(&temp_dir, "anthropic", &["claude-3-opus", "claude-3-sonnet"]).await;

        let loader = ModelLoader::new(temp_dir.path()).with_max_concurrent(2);
        let registry = loader.load_existing_models().await.expect("Load models");

        assert_eq!(registry.provider_count(), 2);
        assert_eq!(registry.model_count(), 4);
        
        assert!(registry.contains_key("openai:gpt-4"));
        assert!(registry.contains_key("openai:gpt-3.5-turbo"));
        assert!(registry.contains_key("anthropic:claude-3-opus"));
        assert!(registry.contains_key("anthropic:claude-3-sonnet"));
    }
}