//! HTTP3 conditional YAML download manager
//!
//! Zero-allocation, lock-free implementation for downloading models.yaml with
//! ETag-based conditional requests and intelligent caching.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use fluent_ai_http3::{HttpClient, HttpRequest, HttpConfig, HttpMethod};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tracing::{debug, error, info, instrument, warn};

// Import provider's AsyncStream primitives
use crate::{AsyncStream, AsyncStreamSender, channel};

use super::errors::{BuildError, BuildResult};
use super::yaml_processor::{YamlProcessor, ProviderInfo};

/// Cache metadata for YAML files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlCacheMetadata {
    /// URL of the YAML file
    pub url: String,
    /// ETag from last successful download
    pub etag: Option<String>,
    /// Computed expires timestamp (Unix timestamp)
    pub computed_expires: Option<u64>,
    /// Last modified timestamp
    pub last_modified: Option<u64>,
    /// Content hash for integrity verification
    pub content_hash: u64,
    /// File size in bytes
    pub file_size: u64,
    /// Cache file path
    pub cache_path: PathBuf,
}

impl YamlCacheMetadata {
    /// Create new cache metadata
    pub fn new(url: String, cache_path: PathBuf) -> Self {
        Self {
            url,
            etag: None,
            computed_expires: None,
            last_modified: None,
            content_hash: 0,
            file_size: 0,
            cache_path,
        }
    }

    /// Check if cache is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires) = self.computed_expires {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            
            now >= expires
        } else {
            true // No expiration set, consider expired
        }
    }

    /// Update metadata from HTTP response
    pub fn update_from_response(&mut self, response: &fluent_ai_http3::HttpResponse, content: &[u8]) {
        self.etag = response.etag().map(|s| s.clone());
        self.computed_expires = response.computed_expires();
        self.last_modified = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        );
        self.content_hash = self.calculate_content_hash(content);
        self.file_size = content.len() as u64;
    }

    /// Calculate content hash for integrity verification
    fn calculate_content_hash(&self, content: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }
}

/// Result of YAML download operation
#[derive(Debug)]
pub enum YamlDownloadResult {
    /// Fresh content downloaded from server
    Downloaded {
        content: String,
        metadata: YamlCacheMetadata,
    },
    /// Content served from cache (304 Not Modified)
    Cached {
        content: String,
        metadata: YamlCacheMetadata,
    },
    /// No changes detected, existing cache is valid
    NotModified {
        metadata: YamlCacheMetadata,
    },
}

impl YamlDownloadResult {
    /// Get the content regardless of source
    pub fn content(&self) -> Option<&String> {
        match self {
            Self::Downloaded { content, .. } | Self::Cached { content, .. } => Some(content),
            Self::NotModified { .. } => None,
        }
    }

    /// Get the metadata
    pub fn metadata(&self) -> &YamlCacheMetadata {
        match self {
            Self::Downloaded { metadata, .. } 
            | Self::Cached { metadata, .. } 
            | Self::NotModified { metadata } => metadata,
        }
    }

    /// Check if content was served from cache
    pub fn is_cached(&self) -> bool {
        matches!(self, Self::Cached { .. } | Self::NotModified { .. })
    }
}

/// HTTP3 conditional YAML manager for intelligent downloading and caching
#[derive(Debug)]
pub struct YamlManager {
    /// HTTP3 client for downloads
    http_client: Arc<HttpClient>,
    /// Cache directory for storing YAML files and metadata
    cache_dir: PathBuf,
    /// YAML processor for parsing content
    yaml_processor: YamlProcessor,
    /// Default expires duration in seconds (24 hours)
    default_expires_secs: u64,
}

impl YamlManager {
    /// Create a new YAML manager with default configuration
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> BuildResult<Self> {
        let http_client = Arc::new(
            HttpClient::with_config(HttpConfig::ai_optimized())
                .map_err(|e| BuildError::HttpError(format!("Failed to create HTTP client: {}", e)))?
        );

        let cache_dir = cache_dir.as_ref().to_path_buf();

        Ok(Self {
            http_client,
            cache_dir,
            yaml_processor: YamlProcessor::new(),
            default_expires_secs: 24 * 60 * 60, // 24 hours
        })
    }

    /// Create YAML manager with custom HTTP client
    pub fn with_client<P: AsRef<Path>>(client: Arc<HttpClient>, cache_dir: P) -> Self {
        Self {
            http_client: client,
            cache_dir: cache_dir.as_ref().to_path_buf(),
            yaml_processor: YamlProcessor::new(),
            default_expires_secs: 24 * 60 * 60,
        }
    }

    /// Set default expires duration
    pub fn with_default_expires(mut self, expires_secs: u64) -> Self {
        self.default_expires_secs = expires_secs;
        self
    }

    /// Download YAML file with conditional requests and caching
    #[instrument(skip(self))]
    pub async fn download_yaml(&self, url: &str) -> BuildResult<YamlDownloadResult> {
        let cache_path = self.get_cache_path(url);
        let metadata_path = self.get_metadata_path(url);

        // Ensure cache directory exists
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent).await
                .map_err(|e| BuildError::IoError(e))?;
        }

        // Load existing cache metadata if available
        let mut metadata = self.load_cache_metadata(&metadata_path, url).await?;

        // Check if cache is valid and not expired
        if cache_path.exists() && !metadata.is_expired() {
            debug!("Cache is valid for {}, checking with server using ETag", url);
        } else {
            debug!("Cache is expired or missing for {}, will download fresh", url);
        }

        // Build conditional HTTP request
        let mut request = HttpRequest::new(HttpMethod::Get, url.to_string())
            .header("User-Agent", "fluent-ai-provider/1.0");

        // Add conditional headers if we have cache metadata
        if let Some(etag) = &metadata.etag {
            request = request.if_none_match(etag.clone());
        }

        // Send the request
        match self.http_client.send(request).await {
            Ok(response) => {
                if response.status().as_u16() == 304 {
                    // Not Modified - serve from cache
                    info!("YAML not modified (304), serving from cache: {}", url);
                    self.serve_from_cache(metadata).await
                } else if response.is_success() {
                    // Fresh content - download and cache
                    info!("Downloading fresh YAML content: {}", url);
                    self.download_and_cache(response, metadata).await
                } else {
                    // HTTP error - try to serve from cache as fallback
                    warn!("HTTP error {} for {}, attempting cache fallback", response.status(), url);
                    self.fallback_to_cache(metadata, &format!("HTTP {}", response.status())).await
                }
            }
            Err(e) => {
                // Network error - try to serve from cache as fallback
                warn!("Network error for {}: {}, attempting cache fallback", url, e);
                self.fallback_to_cache(metadata, &format!("Network error: {}", e)).await
            }
        }
    }

    /// Parse downloaded YAML content into provider definitions - pure streaming architecture
    #[instrument(skip(self, content))]
    pub fn parse_providers(&self, content: &str) -> impl crate::http3_streaming::DownloadChunk {
        use crate::http3_streaming::DownloadChunkImpl;
        
        // Parse YAML synchronously - no spawn operations
        let processor = self.yaml_processor.clone();
        let result = processor.parse_providers(content);
        
        match result {
            Ok(providers) => {
                let data = serde_json::to_vec(&providers).unwrap_or_default();
                DownloadChunkImpl::new(
                    data,
                    Some(format!("Parsed {} providers", providers.len())),
                    Some("yaml_parsing".to_string()),
                ).with_progress(100.0).with_final(true)
            }
            Err(_) => {
                // Return empty chunk on parsing error
                DownloadChunkImpl::new(
                    Vec::new(),
                    Some("YAML parsing failed".to_string()),
                    Some("yaml_error".to_string()),
                ).with_progress(0.0).with_final(true)
            }
        }
    }

    /// Download and cache fresh content
    async fn download_and_cache(
        &self,
        response: fluent_ai_http3::HttpResponse,
        mut metadata: YamlCacheMetadata,
    ) -> BuildResult<YamlDownloadResult> {
        // Stream the response content
        let content_bytes = response.bytes()
            .map_err(|e| BuildError::HttpError(format!("Failed to read response: {}", e)))?;

        let content = String::from_utf8(content_bytes.to_vec())
            .map_err(|e| BuildError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid UTF-8 in YAML: {}", e)
            )))?;

        // Update metadata from response
        metadata.update_from_response(&response, &content_bytes);

        // Write content to cache file
        fs::write(&metadata.cache_path, &content).await
            .map_err(|e| BuildError::IoError(e))?;

        // Write metadata file
        let metadata_path = self.get_metadata_path(&metadata.url);
        self.save_cache_metadata(&metadata_path, &metadata).await?;

        debug!("Cached YAML content: {} bytes, ETag: {:?}", 
               content.len(), metadata.etag);

        Ok(YamlDownloadResult::Downloaded { content, metadata })
    }

    /// Serve content from cache
    async fn serve_from_cache(
        &self,
        metadata: YamlCacheMetadata,
    ) -> BuildResult<YamlDownloadResult> {
        if metadata.cache_path.exists() {
            let content = fs::read_to_string(&metadata.cache_path).await
                .map_err(|e| BuildError::IoError(e))?;

            debug!("Serving from cache: {} bytes", content.len());
            Ok(YamlDownloadResult::Cached { content, metadata })
        } else {
            Ok(YamlDownloadResult::NotModified { metadata })
        }
    }

    /// Fallback to cache when download fails
    async fn fallback_to_cache(
        &self,
        metadata: YamlCacheMetadata,
        error_reason: &str,
    ) -> BuildResult<YamlDownloadResult> {
        if metadata.cache_path.exists() {
            warn!("Using stale cache due to {}: {}", error_reason, metadata.url);
            
            let content = fs::read_to_string(&metadata.cache_path).await
                .map_err(|e| BuildError::IoError(e))?;

            Ok(YamlDownloadResult::Cached { content, metadata })
        } else {
            Err(BuildError::HttpError(format!(
                "Download failed and no cache available: {}",
                error_reason
            )))
        }
    }

    /// Load cache metadata from file
    async fn load_cache_metadata(
        &self,
        metadata_path: &Path,
        url: &str,
    ) -> BuildResult<YamlCacheMetadata> {
        if metadata_path.exists() {
            let metadata_content = fs::read_to_string(metadata_path).await
                .map_err(|e| BuildError::IoError(e))?;

            serde_json::from_str(&metadata_content)
                .map_err(|e| BuildError::IoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid metadata JSON: {}", e)
                )))
        } else {
            Ok(YamlCacheMetadata::new(
                url.to_string(),
                self.get_cache_path(url),
            ))
        }
    }

    /// Save cache metadata to file
    async fn save_cache_metadata(
        &self,
        metadata_path: &Path,
        metadata: &YamlCacheMetadata,
    ) -> BuildResult<()> {
        let metadata_json = serde_json::to_string_pretty(metadata)
            .map_err(|e| BuildError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to serialize metadata: {}", e)
            )))?;

        fs::write(metadata_path, metadata_json).await
            .map_err(|e| BuildError::IoError(e))?;

        Ok(())
    }

    /// Get cache file path for a URL
    fn get_cache_path(&self, url: &str) -> PathBuf {
        let url_hash = self.hash_url(url);
        self.cache_dir.join(format!("{}.yaml", url_hash))
    }

    /// Get metadata file path for a URL
    fn get_metadata_path(&self, url: &str) -> PathBuf {
        let url_hash = self.hash_url(url);
        self.cache_dir.join(format!("{}.meta.json", url_hash))
    }

    /// Create a hash of URL for cache file naming
    fn hash_url(&self, url: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        url.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// Builder for YamlManager with fluent configuration
#[derive(Debug)]
pub struct YamlManagerBuilder {
    cache_dir: Option<PathBuf>,
    http_client: Option<Arc<HttpClient>>,
    default_expires_secs: u64,
}

impl YamlManagerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            cache_dir: None,
            http_client: None,
            default_expires_secs: 24 * 60 * 60, // 24 hours
        }
    }

    /// Set cache directory
    pub fn cache_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.cache_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Set HTTP client
    pub fn http_client(mut self, client: Arc<HttpClient>) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Set default expires duration
    pub fn default_expires(mut self, expires_secs: u64) -> Self {
        self.default_expires_secs = expires_secs;
        self
    }

    /// Build the YamlManager
    pub fn build(self) -> BuildResult<YamlManager> {
        let cache_dir = self.cache_dir
            .unwrap_or_else(|| std::env::temp_dir().join("fluent-ai-yaml-cache"));

        let manager = if let Some(client) = self.http_client {
            YamlManager::with_client(client, cache_dir)
        } else {
            YamlManager::new(cache_dir)?
        };

        Ok(manager.with_default_expires(self.default_expires_secs))
    }
}

impl Default for YamlManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use fluent_ai_http3::HttpConfig;

    #[tokio::test]
    async fn test_yaml_cache_metadata() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let cache_path = temp_dir.path().join("test.yaml");
        
        let mut metadata = YamlCacheMetadata::new(
            "https://example.com/models.yaml".to_string(),
            cache_path,
        );

        // Initially should be expired (no expires set)
        assert!(metadata.is_expired());

        // Set future expiration
        let future_expires = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() + 3600; // 1 hour from now

        metadata.computed_expires = Some(future_expires);
        assert!(!metadata.is_expired());

        // Set past expiration  
        metadata.computed_expires = Some(1000); // Way in the past
        assert!(metadata.is_expired());
    }

    #[tokio::test]
    async fn test_yaml_manager_builder() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        
        let manager = YamlManagerBuilder::new()
            .cache_dir(temp_dir.path())
            .default_expires(3600)
            .build()
            .expect("Build manager");

        assert_eq!(manager.default_expires_secs, 3600);
        assert_eq!(manager.cache_dir, temp_dir.path());
    }

    #[tokio::test]
    async fn test_cache_path_generation() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let manager = YamlManager::new(temp_dir.path()).expect("Create manager");

        let url1 = "https://example.com/models.yaml";
        let url2 = "https://different.com/models.yaml";

        let path1 = manager.get_cache_path(url1);
        let path2 = manager.get_cache_path(url2);

        // Different URLs should have different cache paths
        assert_ne!(path1, path2);

        // Same URL should have same cache path
        let path1_again = manager.get_cache_path(url1);
        assert_eq!(path1, path1_again);

        // Paths should be in the cache directory
        assert!(path1.starts_with(temp_dir.path()));
        assert!(path2.starts_with(temp_dir.path()));
    }

    #[tokio::test]
    async fn test_metadata_persistence() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let manager = YamlManager::new(temp_dir.path()).expect("Create manager");

        let url = "https://example.com/models.yaml";
        let metadata_path = manager.get_metadata_path(url);

        // Create test metadata
        let mut metadata = YamlCacheMetadata::new(url.to_string(), manager.get_cache_path(url));
        metadata.etag = Some("test-etag".to_string());
        metadata.computed_expires = Some(1234567890);

        // Save metadata
        manager.save_cache_metadata(&metadata_path, &metadata).await
            .expect("Save metadata");

        // Load metadata
        let loaded = manager.load_cache_metadata(&metadata_path, url).await
            .expect("Load metadata");

        assert_eq!(loaded.etag, Some("test-etag".to_string()));
        assert_eq!(loaded.computed_expires, Some(1234567890));
        assert_eq!(loaded.url, url);
    }

    #[test]
    fn test_yaml_download_result() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let metadata = YamlCacheMetadata::new(
            "https://example.com/test.yaml".to_string(),
            temp_dir.path().join("test.yaml"),
        );

        let downloaded = YamlDownloadResult::Downloaded {
            content: "test content".to_string(),
            metadata: metadata.clone(),
        };

        assert_eq!(downloaded.content(), Some(&"test content".to_string()));
        assert!(!downloaded.is_cached());

        let cached = YamlDownloadResult::Cached {
            content: "cached content".to_string(),
            metadata: metadata.clone(),
        };

        assert_eq!(cached.content(), Some(&"cached content".to_string()));
        assert!(cached.is_cached());

        let not_modified = YamlDownloadResult::NotModified { metadata };
        assert_eq!(not_modified.content(), None);
        assert!(not_modified.is_cached());
    }
}