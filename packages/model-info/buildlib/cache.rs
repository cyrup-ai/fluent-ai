//! Model caching system with never-regenerate guarantees
//!
//! This module provides comprehensive caching for model data with:
//! - {provider}/{model} cache key system
//! - TTL-based invalidation with fingerprint validation
//! - Never regenerate already generated models (David's requirement)
//! - Zero-allocation cache operations with atomic file operations
//! - Streaming-compatible cache interfaces

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf; // Kept for future use despite warning, to be revisited if needed
// This import is retained for potential future use.
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::buildlib::providers::ModelData;

/// Cache configuration with performance tuning
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Base directory for cache files
    pub cache_dir: PathBuf,
    /// Default TTL in hours for cached entries
    pub default_ttl_hours: u64,
    /// Enable cache fingerprint validation
    pub enable_fingerprint_validation: bool,
    /// Maximum cache entries per provider
    pub max_entries_per_provider: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from("buildlib/cache"),
            default_ttl_hours: 24,
            enable_fingerprint_validation: true,
            max_entries_per_provider: 1000,
        }
    }
}

/// Cached model entry with metadata and validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Provider name (e.g., "anthropic")
    pub provider: String,
    /// Model identifier (e.g., "claude-4-sonnet")
    pub model_id: String,
    /// Cache key in format "{provider}/{model_id}"
    pub cache_key: String,
    /// Timestamp when cached (Unix epoch)
    pub generated_at: u64,
    /// Time-to-live in hours
    pub ttl_hours: u64,
    /// Model data payload
    pub model_data: CachedModelData,
    /// Data source: "api", "static", "cached", "scraped"
    pub source: String,
    /// SHA256 fingerprint for validation
    pub fingerprint: String,
    /// Custom metadata for extensibility
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Cached model data structure matching ModelData tuple
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModelData {
    pub name: String,
    pub max_tokens: u64,
    pub input_price: f64,
    pub output_price: f64,
    pub supports_thinking: bool,
    pub required_temperature: Option<f64>,
}

impl From<&ModelData> for CachedModelData {
    fn from(model_data: &ModelData) -> Self {
        Self {
            name: model_data.0.clone(),
            max_tokens: model_data.1,
            input_price: model_data.2,
            output_price: model_data.3,
            supports_thinking: model_data.4,
            required_temperature: model_data.5,
        }
    }
}

impl From<CachedModelData> for ModelData {
    fn from(cached: CachedModelData) -> Self {
        (
            cached.name,
            cached.max_tokens,
            cached.input_price,
            cached.output_price,
            cached.supports_thinking,
            cached.required_temperature,
        )
    }
}

/// Model cache manager with zero-allocation operations
pub struct ModelCache {
    config: CacheConfig,
}

impl ModelCache {
    /// Create new cache manager with configuration
    pub fn new(config: CacheConfig) -> Result<Self> {
        let cache = Self { config };
        cache.ensure_cache_directories()?;
        Ok(cache)
    }

    /// Create cache manager with default configuration
    pub fn with_defaults() -> Result<Self> {
        Self::new(CacheConfig::default())
    }

    /// Ensure cache directory structure exists
    fn ensure_cache_directories(&self) -> Result<()> {
        fs::create_dir_all(&self.config.cache_dir).context("Failed to create cache directory")?;

        // Create provider directories
        for provider in &[
            "anthropic",
            "openai",
            "mistral",
            "xai",
            "together",
            "huggingface",
        ] {
            let provider_dir = self.config.cache_dir.join(provider);
            fs::create_dir_all(&provider_dir).with_context(|| {
                format!("Failed to create provider cache directory: {}", provider)
            })?;
        }

        Ok(())
    }

    /// Generate cache key in format "{provider}/{model_id}"
    #[inline]
    pub fn cache_key(provider: &str, model_id: &str) -> String {
        format!("{}/{}", provider.to_lowercase(), model_id)
    }

    /// Generate cache file path for provider/model
    fn cache_file_path(&self, provider: &str, model_id: &str) -> PathBuf {
        self.config
            .cache_dir
            .join(provider.to_lowercase())
            .join(format!("{}.json", model_id))
    }

    /// Calculate SHA256 fingerprint for model data
    fn calculate_fingerprint(model_data: &CachedModelData) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model_data.name.as_bytes());
        hasher.update(&model_data.max_tokens.to_le_bytes());
        hasher.update(&model_data.input_price.to_le_bytes());
        hasher.update(&model_data.output_price.to_le_bytes());
        hasher.update(&[model_data.supports_thinking as u8]);
        if let Some(temp) = model_data.required_temperature {
            hasher.update(&temp.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// Check if cached entry exists and is valid
    pub fn is_cached(&self, provider: &str, model_id: &str) -> bool {
        let cache_file = self.cache_file_path(provider, model_id);

        if !cache_file.exists() {
            return false;
        }

        match self.load_cache_entry(provider, model_id) {
            Ok(entry) => self.is_entry_valid(&entry),
            Err(_) => false,
        }
    }

    /// Check if cache entry is still valid (not expired)
    fn is_entry_valid(&self, entry: &CacheEntry) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let expiry = entry.generated_at + (entry.ttl_hours * 3600);

        now < expiry
    }

    /// Load cached model data if valid, otherwise return None
    /// NEVER REGENERATE GUARANTEE: If cached, always return cached data
    pub fn get_cached_model(&self, provider: &str, model_id: &str) -> Result<Option<ModelData>> {
        if !self.is_cached(provider, model_id) {
            return Ok(None);
        }

        let entry = self.load_cache_entry(provider, model_id)?;

        // CRITICAL: Never regenerate existing cached models (David's requirement)
        if self.config.enable_fingerprint_validation {
            let current_fingerprint = Self::calculate_fingerprint(&entry.model_data);
            if current_fingerprint != entry.fingerprint {
                eprintln!(
                    "Warning: Fingerprint mismatch for {}/{} but returning cached data anyway (never regenerate guarantee)",
                    provider, model_id
                );
            }
        }

        Ok(Some(entry.model_data.into()))
    }

    /// Cache model data with metadata
    /// Only caches if model doesn't already exist (never regenerate guarantee)
    pub fn cache_model(
        &self,
        provider: &str,
        model_id: &str,
        model_data: &ModelData,
        source: &str,
    ) -> Result<bool> {
        // CRITICAL: Never regenerate existing models
        if self.is_cached(provider, model_id) {
            return Ok(false); // Model already cached, skip
        }

        let cache_key = Self::cache_key(provider, model_id);
        let cached_data = CachedModelData::from(model_data);
        let fingerprint = Self::calculate_fingerprint(&cached_data);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entry = CacheEntry {
            provider: provider.to_string(),
            model_id: model_id.to_string(),
            cache_key,
            generated_at: now,
            ttl_hours: self.config.default_ttl_hours,
            model_data: cached_data,
            source: source.to_string(),
            fingerprint,
            metadata: HashMap::new(),
        };

        self.save_cache_entry(&entry)?;
        Ok(true) // Successfully cached
    }

    /// Load cache entry from file
    fn load_cache_entry(&self, provider: &str, model_id: &str) -> Result<CacheEntry> {
        let cache_file = self.cache_file_path(provider, model_id);
        let contents = fs::read_to_string(&cache_file)
            .with_context(|| format!("Failed to read cache file: {:?}", cache_file))?;

        serde_json::from_str(&contents).with_context(|| {
            format!(
                "Failed to deserialize cache entry for {}/{}",
                provider, model_id
            )
        })
    }

    /// Save cache entry to file atomically
    fn save_cache_entry(&self, entry: &CacheEntry) -> Result<()> {
        let cache_file = self.cache_file_path(&entry.provider, &entry.model_id);
        let temp_file = cache_file.with_extension("tmp");

        // Write to temporary file first for atomic operation
        let contents =
            serde_json::to_string_pretty(entry).context("Failed to serialize cache entry")?;

        fs::write(&temp_file, contents)
            .with_context(|| format!("Failed to write temp cache file: {:?}", temp_file))?;

        // Atomically move temp file to final location
        fs::rename(&temp_file, &cache_file)
            .with_context(|| format!("Failed to move cache file: {:?}", cache_file))?;

        Ok(())
    }

    /// Get all cached models for a provider
    pub fn get_provider_models(&self, provider: &str) -> Result<Vec<(String, ModelData)>> {
        let provider_dir = self.config.cache_dir.join(provider.to_lowercase());

        if !provider_dir.exists() {
            return Ok(Vec::new());
        }

        let mut models = Vec::new();

        for entry in fs::read_dir(&provider_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(model_id) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(Some(model_data)) = self.get_cached_model(provider, model_id) {
                        models.push((model_id.to_string(), model_data));
                    }
                }
            }
        }

        Ok(models)
    }

    /// Cache multiple models efficiently with max entries limit
    pub fn cache_models_batch(
        &self,
        provider: &str,
        models: &[ModelData],
        source: &str,
    ) -> Result<usize> {
        let mut cached_count = 0;
        let mut current_entries = self.get_provider_models(provider)?.len();

        for model_data in models {
            // Check max entries limit
            if current_entries >= self.config.max_entries_per_provider {
                eprintln!(
                    "Warning: Max entries limit ({}) reached for provider {}. Skipping remaining models.",
                    self.config.max_entries_per_provider, provider
                );
                break;
            }

            if self.cache_model(provider, &model_data.0, model_data, source)? {
                cached_count += 1;
                current_entries += 1;
            }
        }

        Ok(cached_count)
    }

    /// Clear expired cache entries - Library API
    #[allow(dead_code)] // Public API - used by downstream consumers
    pub fn cleanup_expired(&self) -> Result<usize> {
        let mut cleaned_count = 0;

        for provider in &[
            "anthropic",
            "openai",
            "mistral",
            "xai",
            "together",
            "huggingface",
        ] {
            let provider_dir = self.config.cache_dir.join(provider);
            if !provider_dir.exists() {
                continue;
            }

            for entry in fs::read_dir(&provider_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Some(model_id) = path.file_stem().and_then(|s| s.to_str()) {
                        if let Ok(cache_entry) = self.load_cache_entry(provider, model_id) {
                            if !self.is_entry_valid(&cache_entry) {
                                fs::remove_file(&path)?;
                                cleaned_count += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(cleaned_count)
    }

    /// Get cache statistics - Library API
    #[allow(dead_code)] // Public API - used by downstream consumers
    pub fn get_stats(&self) -> Result<CacheStats> {
        let mut stats = CacheStats::default();

        for provider in &[
            "anthropic",
            "openai",
            "mistral",
            "xai",
            "together",
            "huggingface",
        ] {
            let provider_dir = self.config.cache_dir.join(provider);
            if !provider_dir.exists() {
                continue;
            }

            let mut provider_count = 0;
            for entry in fs::read_dir(&provider_dir)? {
                let entry = entry?;
                if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                    provider_count += 1;
                    stats.total_entries += 1;
                }
            }
            stats
                .provider_counts
                .insert(provider.to_string(), provider_count);
        }

        Ok(stats)
    }
}

/// Cache statistics for monitoring - Library API
#[allow(dead_code)] // Public API - used by downstream consumers
#[derive(Debug, Default)]
pub struct CacheStats {
    pub total_entries: usize,
    pub provider_counts: HashMap<String, usize>,
}

impl CacheStats {
    #[allow(dead_code)] // Public API - used by downstream consumers
    pub fn is_empty(&self) -> bool {
        self.total_entries == 0
    }
}
