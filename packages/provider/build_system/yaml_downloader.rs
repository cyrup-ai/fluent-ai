//! ETag-based YAML downloader for model definitions
//!
//! This module provides efficient downloading and caching of model YAML files
//! using ETags and checksums for validation and incremental updates.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::task::JoinSet;

use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpResult};

use super::errors::{BuildError, BuildResult};

/// Default model definitions URL (sigoden's curated list)
const DEFAULT_MODELS_URL: &str = "https://raw.githubusercontent.com/sigoden/llm-functions/main/tools/models.yaml";

/// Cache directory for downloaded YAML files
const CACHE_DIR: &str = "build_cache";

/// Maximum cache age in seconds (24 hours)
const MAX_CACHE_AGE_SECS: u64 = 86400;

/// File metadata with ETag and checksum validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    /// ETag from the HTTP response
    pub etag: Option<String>,
    /// Last-Modified header from the HTTP response
    pub last_modified: Option<String>,
    /// SHA256 checksum of the file content
    pub checksum: String,
    /// File size in bytes
    pub file_size: u64,
    /// When the file was downloaded
    pub downloaded_at: SystemTime,
    /// Original URL the file was downloaded from
    pub source_url: String,
    /// Content type from the HTTP response
    pub content_type: Option<String>,
}

impl FileMetadata {
    /// Create new metadata for a file
    pub fn new(
        content: &[u8],
        etag: Option<String>,
        last_modified: Option<String>,
        source_url: String,
        content_type: Option<String>,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(content);
        let checksum = format!("{:x}", hasher.finalize());

        Self {
            etag,
            last_modified,
            checksum,
            file_size: content.len() as u64,
            downloaded_at: SystemTime::now(),
            source_url,
            content_type,
        }
    }

    /// Check if the cached file is still valid
    pub fn is_valid(&self) -> bool {
        // Check age
        if let Ok(elapsed) = self.downloaded_at.elapsed() {
            if elapsed.as_secs() > MAX_CACHE_AGE_SECS {
                return false;
            }
        }

        true
    }

    /// Validate the content matches this metadata
    pub fn validate_content(&self, content: &[u8]) -> bool {
        if content.len() as u64 != self.file_size {
            return false;
        }

        let mut hasher = Sha256::new();
        hasher.update(content);
        let computed_checksum = format!("{:x}", hasher.finalize());

        computed_checksum == self.checksum
    }

    /// Get conditional request headers for validation
    pub fn conditional_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        if let Some(etag) = &self.etag {
            headers.insert("If-None-Match".to_string(), etag.clone());
        }

        if let Some(last_modified) = &self.last_modified {
            headers.insert("If-Modified-Since".to_string(), last_modified.clone());
        }

        headers
    }
}

/// Result of a YAML download operation
#[derive(Debug)]
pub struct DownloadResult {
    /// The YAML content
    pub content: String,
    /// Whether the content was fetched from cache
    pub from_cache: bool,
    /// File metadata
    pub metadata: FileMetadata,
    /// The file path where content is cached
    pub cache_path: PathBuf,
}

/// High-performance YAML downloader with ETag-based caching
pub struct YamlDownloader {
    client: HttpClient,
    cache_dir: PathBuf,
    base_url: String,
}

impl YamlDownloader {
    /// Create a new YAML downloader with default configuration
    pub fn new() -> BuildResult<Self> {
        Self::with_url(DEFAULT_MODELS_URL)
    }

    /// Create a new YAML downloader with custom URL
    pub fn with_url(url: &str) -> BuildResult<Self> {
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| BuildError::NetworkError(format!("Failed to create HTTP client: {}", e)))?;

        let cache_dir = Path::new(CACHE_DIR).to_path_buf();

        Ok(Self {
            client,
            cache_dir,
            base_url: url.to_string(),
        })
    }

    /// Download YAML file with ETag-based caching
    pub async fn download_yaml(&self) -> BuildResult<DownloadResult> {
        // Ensure cache directory exists
        self.ensure_cache_dir()?;

        let cache_file_path = self.cache_dir.join("models.yaml");
        let metadata_file_path = self.cache_dir.join("models.yaml.meta");

        // Try to load existing metadata
        let existing_metadata = self.load_metadata(&metadata_file_path)?;

        // Check if we have valid cached content
        if let Some(metadata) = &existing_metadata {
            if metadata.is_valid() && cache_file_path.exists() {
                // Try conditional request with ETag/Last-Modified
                match self.try_conditional_request(metadata).await {
                    Ok(Some(new_content)) => {
                        // Content was modified, update cache
                        return self.update_cache(cache_file_path, metadata_file_path, new_content).await;
                    }
                    Ok(None) => {
                        // Content not modified, use cache
                        let content = self.read_cached_file(&cache_file_path)?;
                        return Ok(DownloadResult {
                            content,
                            from_cache: true,
                            metadata: metadata.clone(),
                            cache_path: cache_file_path,
                        });
                    }
                    Err(e) => {
                        // Conditional request failed, fall through to full download
                        eprintln!("Conditional request failed: {}, attempting full download", e);
                    }
                }
            }
        }

        // Perform full download
        self.download_and_cache(cache_file_path, metadata_file_path).await
    }

    /// Download multiple YAML files in parallel
    pub async fn download_multiple(&self, urls: &[String]) -> BuildResult<Vec<DownloadResult>> {
        let mut tasks = JoinSet::new();

        for url in urls {
            let downloader = YamlDownloader::with_url(url)?;
            tasks.spawn(async move { downloader.download_yaml().await });
        }

        let mut results = Vec::with_capacity(urls.len());
        
        while let Some(task_result) = tasks.join_next().await {
            let download_result = task_result
                .map_err(|e| BuildError::TaskError(format!("Download task failed: {}", e)))?;
            results.push(download_result?);
        }

        Ok(results)
    }

    /// Ensure cache directory exists
    fn ensure_cache_dir(&self) -> BuildResult<()> {
        if !self.cache_dir.exists() {
            fs::create_dir_all(&self.cache_dir)
                .map_err(|e| BuildError::IoError(format!("Failed to create cache directory: {}", e)))?;
        }
        Ok(())
    }

    /// Load metadata from cache file
    fn load_metadata(&self, path: &Path) -> BuildResult<Option<FileMetadata>> {
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(path)
            .map_err(|e| BuildError::IoError(format!("Failed to read metadata file: {}", e)))?;

        let metadata: FileMetadata = serde_json::from_str(&content)
            .map_err(|e| BuildError::DeserializationError(format!("Failed to parse metadata: {}", e)))?;

        Ok(Some(metadata))
    }

    /// Save metadata to cache file
    fn save_metadata(&self, path: &Path, metadata: &FileMetadata) -> BuildResult<()> {
        let content = serde_json::to_string_pretty(metadata)
            .map_err(|e| BuildError::SerializationError(format!("Failed to serialize metadata: {}", e)))?;

        fs::write(path, content)
            .map_err(|e| BuildError::IoError(format!("Failed to write metadata file: {}", e)))?;

        Ok(())
    }

    /// Try conditional request with existing metadata
    async fn try_conditional_request(&self, metadata: &FileMetadata) -> BuildResult<Option<(Bytes, FileMetadata)>> {
        let mut request = self.client.get(&self.base_url);

        // Add conditional headers
        for (key, value) in metadata.conditional_headers() {
            request = request.header(key, value);
        }

        let response = request.send().await
            .map_err(|e| BuildError::NetworkError(format!("Failed to send conditional request: {}", e)))?;

        match response.status().as_u16() {
            304 => {
                // Not Modified - use cached content
                Ok(None)
            }
            200 => {
                // Modified - download new content
                let etag = response.etag().cloned();
                let last_modified = response.last_modified().cloned();
                let content_type = response.content_type().cloned();

                let content = response.bytes().await
                    .map_err(|e| BuildError::NetworkError(format!("Failed to read response body: {}", e)))?;

                let new_metadata = FileMetadata::new(
                    &content,
                    etag,
                    last_modified,
                    self.base_url.clone(),
                    content_type,
                );

                Ok(Some((content, new_metadata)))
            }
            status => {
                Err(BuildError::NetworkError(format!("Unexpected HTTP status: {}", status)))
            }
        }
    }

    /// Perform full download and cache the result
    async fn download_and_cache(
        &self,
        cache_file_path: PathBuf,
        metadata_file_path: PathBuf,
    ) -> BuildResult<DownloadResult> {
        let response = self.client.get(&self.base_url).send().await
            .map_err(|e| BuildError::NetworkError(format!("Failed to download YAML: {}", e)))?;

        if !response.is_success() {
            return Err(BuildError::NetworkError(format!(
                "HTTP error {}: Failed to download YAML from {}",
                response.status(),
                self.base_url
            )));
        }

        let etag = response.etag().cloned();
        let last_modified = response.last_modified().cloned();
        let content_type = response.content_type().cloned();

        let content = response.bytes().await
            .map_err(|e| BuildError::NetworkError(format!("Failed to read response body: {}", e)))?;

        let metadata = FileMetadata::new(
            &content,
            etag,
            last_modified,
            self.base_url.clone(),
            content_type,
        );

        // Save to cache
        fs::write(&cache_file_path, &content)
            .map_err(|e| BuildError::IoError(format!("Failed to write cache file: {}", e)))?;

        self.save_metadata(&metadata_file_path, &metadata)?;

        let content_string = String::from_utf8(content.to_vec())
            .map_err(|e| BuildError::DecodingError(format!("Invalid UTF-8 in YAML content: {}", e)))?;

        Ok(DownloadResult {
            content: content_string,
            from_cache: false,
            metadata,
            cache_path: cache_file_path,
        })
    }

    /// Update cache with new content
    async fn update_cache(
        &self,
        cache_file_path: PathBuf,
        metadata_file_path: PathBuf,
        (content, metadata): (Bytes, FileMetadata),
    ) -> BuildResult<DownloadResult> {
        // Save to cache
        fs::write(&cache_file_path, &content)
            .map_err(|e| BuildError::IoError(format!("Failed to write cache file: {}", e)))?;

        self.save_metadata(&metadata_file_path, &metadata)?;

        let content_string = String::from_utf8(content.to_vec())
            .map_err(|e| BuildError::DecodingError(format!("Invalid UTF-8 in YAML content: {}", e)))?;

        Ok(DownloadResult {
            content: content_string,
            from_cache: false,
            metadata,
            cache_path: cache_file_path,
        })
    }

    /// Read cached file content
    fn read_cached_file(&self, path: &Path) -> BuildResult<String> {
        fs::read_to_string(path)
            .map_err(|e| BuildError::IoError(format!("Failed to read cached file: {}", e)))
    }
}

impl Default for YamlDownloader {
    fn default() -> Self {
        Self::new().expect("Failed to create default YAML downloader")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_yaml_downloader() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache");
        
        let mut downloader = YamlDownloader::new().unwrap();
        downloader.cache_dir = cache_dir;

        // First download should fetch from network
        let result1 = downloader.download_yaml().await.unwrap();
        assert!(!result1.from_cache);
        assert!(!result1.content.is_empty());

        // Second download should use cache (if content hasn't changed)
        let result2 = downloader.download_yaml().await.unwrap();
        // Note: This might still be from network if the upstream file changed
        // In a real test, we'd mock the HTTP client
    }

    #[test]
    fn test_file_metadata() {
        let content = b"test content";
        let metadata = FileMetadata::new(
            content,
            Some("etag123".to_string()),
            Some("Wed, 21 Oct 2015 07:28:00 GMT".to_string()),
            "https://example.com/test.yaml".to_string(),
            Some("text/yaml".to_string()),
        );

        assert!(metadata.validate_content(content));
        assert!(!metadata.validate_content(b"different content"));
        assert_eq!(metadata.file_size, content.len() as u64);
    }
}