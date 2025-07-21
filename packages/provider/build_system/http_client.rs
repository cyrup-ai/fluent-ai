//! HTTP client for fetching external model data during build
//!
//! This module provides zero-allocation HTTP client functionality for downloading
//! model definitions from external sources like sigoden's models.yaml.

use std::sync::Arc;
use std::time::Duration;

use super::errors::{BuildError, BuildResult};
use super::performance::PerformanceMonitor;

/// High-performance HTTP client for build-time data fetching
pub struct HttpClient {
    client: reqwest::blocking::Client,
    perf_monitor: Arc<PerformanceMonitor>,
}

impl HttpClient {
    /// Create new HTTP client with optimized configuration
    pub fn new(perf_monitor: Arc<PerformanceMonitor>) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("fluent-ai-provider-build/1.0")
            .gzip(true)
            .brotli(true)
            .deflate(true)
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());

        Self {
            client,
            perf_monitor,
        }
    }

    /// Fetch YAML content from URL with error handling and performance monitoring
    pub fn fetch_yaml(&self, url: &str) -> BuildResult<String> {
        let _timer = self.perf_monitor.start_timer("http_fetch_yaml");
        
        let response = self.client
            .get(url)
            .send()
            .map_err(|e| BuildError::Network(format!("Failed to fetch {}: {}", url, e).into()))?;

        if !response.status().is_success() {
            return Err(BuildError::Network(format!(
                "HTTP {} when fetching {}", 
                response.status(), 
                url
            ).into()));
        }

        let content = response
            .text()
            .map_err(|e| BuildError::Network(format!("Failed to read response body: {}", e).into()))?;

        // Validate that we received valid YAML content
        if content.trim().is_empty() {
            return Err(BuildError::Network(format!("Empty response from {}", url).into()));
        }

        // Basic YAML validation - should start with array or object
        let trimmed = content.trim_start();
        if !trimmed.starts_with('[') && !trimmed.starts_with('{') && !trimmed.starts_with('-') {
            return Err(BuildError::Network(format!(
                "Invalid YAML format from {}: content does not appear to be valid YAML", 
                url
            ).into()));
        }

        Ok(content)
    }

    /// Fetch JSON content from URL with error handling and performance monitoring
    pub fn fetch_json(&self, url: &str) -> BuildResult<String> {
        let _timer = self.perf_monitor.start_timer("http_fetch_json");
        
        let response = self.client
            .get(url)
            .header("Accept", "application/json")
            .send()
            .map_err(|e| BuildError::Network(format!("Failed to fetch {}: {}", url, e).into()))?;

        if !response.status().is_success() {
            return Err(BuildError::Network(format!(
                "HTTP {} when fetching {}", 
                response.status(), 
                url
            ).into()));
        }

        let content = response
            .text()
            .map_err(|e| BuildError::Network(format!("Failed to read response body: {}", e).into()))?;

        // Validate that we received valid JSON content
        if content.trim().is_empty() {
            return Err(BuildError::Network(format!("Empty response from {}", url).into()));
        }

        // Basic JSON validation - should start with array or object
        let trimmed = content.trim_start();
        if !trimmed.starts_with('[') && !trimmed.starts_with('{') {
            return Err(BuildError::Network(format!(
                "Invalid JSON format from {}: content does not appear to be valid JSON", 
                url
            ).into()));
        }

        Ok(content)
    }

    /// Check if URL is accessible without downloading full content
    pub fn check_url(&self, url: &str) -> BuildResult<bool> {
        let _timer = self.perf_monitor.start_timer("http_check_url");
        
        let response = self.client
            .head(url)
            .send()
            .map_err(|e| BuildError::Network(format!("Failed to check {}: {}", url, e).into()))?;

        Ok(response.status().is_success())
    }
}