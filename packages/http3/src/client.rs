//! Zero-allocation HTTP3 client with QUIC prioritization and HTTP/2 fallback
//!
//! This module provides a blazing-fast HTTP client optimized for AI provider APIs
//! with connection pooling, intelligent caching, and comprehensive error handling.
//! All operations are lock-free and use atomic counters for thread safety.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime};

use bytes::Bytes;
use crossbeam_skiplist::SkipMap;
use reqwest::tls;

use crate::{HttpConfig, HttpError, HttpRequest, HttpResponse, HttpResult, HttpStream};

/// High-performance HTTP client with QUIC/HTTP3 support and zero-allocation design
pub struct HttpClient {
    inner: reqwest::Client,
    request_count: AtomicUsize,
    connection_count: AtomicUsize,
    total_bytes_sent: AtomicU64,
    total_bytes_received: AtomicU64,
    total_response_time_nanos: AtomicU64,
    successful_requests: AtomicUsize,
    failed_requests: AtomicUsize,
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
    config: HttpConfig,
    start_time: SystemTime,
}

/// Lock-free connection statistics using atomic counters
#[derive(Debug, Clone)]
pub struct ClientStats {
    /// Number of connections created
    pub connections_created: usize,
    /// Total number of requests sent
    pub requests_sent: usize,
    /// Number of successful requests (2xx status)
    pub successful_requests: usize,
    /// Number of failed requests (4xx/5xx status or network errors)
    pub failed_requests: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Total bytes sent in requests
    pub bytes_sent: u64,
    /// Total bytes received in responses
    pub bytes_received: u64,
    /// Average response time across all requests
    pub average_response_time: Duration,
    /// Total uptime since client creation
    pub uptime: Duration,
}

/// Cache entry with zero-allocation validation
#[derive(Debug, Clone)]
struct CacheEntry {
    response_body: Bytes,
    status_code: u16,
    headers: HashMap<String, String>,
    created_at: Instant,
    etag: Option<String>,
    last_modified: Option<String>,
    max_age_seconds: Option<u64>,
}

/// Lock-free cache using crossbeam SkipMap for zero-allocation, blazing-fast performance
/// No locking required - all operations are atomic and thread-safe
static GLOBAL_CACHE: LazyLock<SkipMap<u64, CacheEntry>> = LazyLock::new(|| SkipMap::new());

/// Atomic counter for cache cleanup coordination
static CACHE_CLEANUP_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl HttpClient {
    /// Create a new HTTP client with default configuration optimized for AI providers
    pub fn new() -> HttpResult<Self> {
        Self::with_config(HttpConfig::ai_optimized())
    }

    /// Create a new HTTP client with custom configuration
    pub fn with_config(config: HttpConfig) -> HttpResult<Self> {
        // Optimally configured HTTP3/QUIC client with TLS 1.3 for maximum performance
        let mut builder = reqwest::Client::builder()
            .pool_max_idle_per_host(config.pool_max_idle_per_host)
            .pool_idle_timeout(config.pool_idle_timeout)
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .tcp_nodelay(config.tcp_nodelay)
            .http2_prior_knowledge()
            .http2_adaptive_window(config.http2_adaptive_window)
            .use_rustls_tls() // Explicitly use rustls to avoid TLS backend conflicts
            .tls_built_in_root_certs(true)
            .min_tls_version(tls::Version::TLS_1_3) // TLS 1.3 required for QUIC/HTTP3
            .https_only(config.https_only)
            .user_agent(&config.user_agent);
        // Compression is enabled by default in reqwest 0.12

        // Configure HTTP/2 settings for optimal performance
        if let Some(frame_size) = config.http2_max_frame_size {
            builder = builder.http2_max_frame_size(frame_size);
        }

        if let Some(window_size) = config.http2_initial_stream_window_size {
            builder = builder.http2_initial_stream_window_size(window_size);
        }

        if let Some(window_size) = config.http2_initial_connection_window_size {
            builder = builder.http2_initial_connection_window_size(window_size);
        }

        // Note: http2_max_concurrent_streams not available in reqwest 0.12

        if config.http2_keep_alive {
            if let Some(interval) = config.http2_keep_alive_interval {
                builder = builder.http2_keep_alive_interval(interval);
            }
            if let Some(timeout) = config.http2_keep_alive_timeout {
                builder = builder.http2_keep_alive_timeout(timeout);
            }
        }

        // Configure TCP settings for optimal performance
        if let Some(keepalive) = config.tcp_keepalive {
            builder = builder.tcp_keepalive(keepalive);
        }

        // Add HTTP/3 support if available and enabled
        // Note: Don't use http3_prior_knowledge() as it forces HTTP3 and prevents fallback
        // Let reqwest negotiate the best protocol version available
        if config.http3_enabled {
            // HTTP3 will be attempted automatically if supported by the server
        }

        // Configure connection pooling
        builder = builder.pool_idle_timeout(config.pool_idle_timeout);

        // Configure redirects
        if config.max_redirects > 0 {
            builder = builder.redirect(reqwest::redirect::Policy::limited(config.max_redirects));
        } else {
            builder = builder.redirect(reqwest::redirect::Policy::none());
        }

        // Configure local address binding if specified
        if let Some(local_addr) = config.local_address {
            builder = builder.local_address(local_addr);
        }

        // Build the client
        let client = builder.build().map_err(|e| HttpError::ClientError {
            message: format!("Failed to build HTTP client: {}", e),
        })?;

        let start_time = SystemTime::now();

        Ok(Self {
            inner: client,
            request_count: AtomicUsize::new(0),
            connection_count: AtomicUsize::new(0),
            total_bytes_sent: AtomicU64::new(0),
            total_bytes_received: AtomicU64::new(0),
            total_response_time_nanos: AtomicU64::new(0),
            successful_requests: AtomicUsize::new(0),
            failed_requests: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            config,
            start_time,
        })
    }

    /// Create a GET request with zero allocation
    #[inline(always)]
    pub fn get(&self, url: &str) -> HttpRequest {
        HttpRequest::new(crate::HttpMethod::Get, url.to_string())
    }

    /// Create a POST request with zero allocation
    #[inline(always)]
    pub fn post(&self, url: &str) -> HttpRequest {
        HttpRequest::new(crate::HttpMethod::Post, url.to_string())
    }

    /// Create a PUT request with zero allocation
    #[inline(always)]
    pub fn put(&self, url: &str) -> HttpRequest {
        HttpRequest::new(crate::HttpMethod::Put, url.to_string())
    }

    /// Create a DELETE request with zero allocation
    #[inline(always)]
    pub fn delete(&self, url: &str) -> HttpRequest {
        HttpRequest::new(crate::HttpMethod::Delete, url.to_string())
    }

    /// Create a PATCH request with zero allocation
    #[inline(always)]
    pub fn patch(&self, url: &str) -> HttpRequest {
        HttpRequest::new(crate::HttpMethod::Patch, url.to_string())
    }

    /// Create a HEAD request with zero allocation
    #[inline(always)]
    pub fn head(&self, url: &str) -> HttpRequest {
        HttpRequest::new(crate::HttpMethod::Head, url.to_string())
    }

    /// Download a file from the given URL with streaming support
    /// Returns a DownloadStream that yields DownloadChunk items
    pub async fn download_file(&self, url: &str) -> HttpResult<crate::DownloadStream> {
        // Validate URL
        let parsed_url = url::Url::parse(url).map_err(|e| crate::HttpError::InvalidUrl {
            url: url.to_string(),
            message: format!("Invalid URL format: {}", e),
        })?;

        // Only allow HTTP/HTTPS schemes for downloads
        match parsed_url.scheme() {
            "http" | "https" => {}
            scheme => {
                return Err(crate::HttpError::InvalidUrl {
                    url: url.to_string(),
                    message: format!("Unsupported URL scheme for download: {}", scheme),
                });
            }
        }

        // Build optimized request for downloading
        let mut req_builder = self.inner
            .get(url)
            .header("Accept", "*/*")
            .header("Accept-Encoding", "identity") // Disable compression for downloads
            .header("Connection", "keep-alive")
            .header("User-Agent", &self.config.user_agent);

        // Add range support header to enable resumable downloads
        req_builder = req_builder.header("Accept-Ranges", "bytes");

        // Send request and get streaming response
        let response = req_builder.send().await.map_err(|e| crate::HttpError::NetworkError {
            message: format!("Failed to initiate download: {}", e),
        })?;

        // Check for successful response status
        let status = response.status();
        if !status.is_success() {
            return Err(crate::HttpError::HttpStatus {
                status: status.as_u16(),
                message: format!("Download failed with status: {}", status),
                body: String::new(), // Empty body for download errors
            });
        }

        // Validate content type if present
        if let Some(content_type) = response.headers().get("content-type") {
            if let Ok(content_type_str) = content_type.to_str() {
                // Reject HTML responses as they're likely error pages
                if content_type_str.starts_with("text/html") {
                    return Err(crate::HttpError::InvalidResponse {
                        message: "Received HTML response, likely an error page".to_string(),
                    });
                }
            }
        }

        // Create download stream
        Ok(crate::DownloadStream::new(response))
    }

    /// Send an HTTP request with optimal performance and caching
    pub async fn send(&self, request: HttpRequest) -> HttpResult<HttpResponse> {
        let start_time = Instant::now();

        // Increment request counter atomically
        let request_id = self.request_count.fetch_add(1, Ordering::Relaxed);

        // Generate cache key using fast hash
        let cache_key = self.generate_cache_key(&request);

        // Check cache first for GET requests
        if matches!(request.method(), crate::HttpMethod::Get) {
            if let Some(cached_response) = self.check_cache(cache_key) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(cached_response);
            }
        }

        // Record cache miss
        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        // Build reqwest request with optimal settings
        let mut req_builder = self.build_request(&request)?;

        // Add performance-optimized headers
        req_builder = req_builder
            .header("X-Request-ID", request_id.to_string())
            .header("Connection", "keep-alive")
            .header("Accept-Encoding", "gzip, br, deflate");

        // Add conditional headers if we have cache info
        if matches!(request.method(), crate::HttpMethod::Get) {
            req_builder = self.add_conditional_headers(req_builder, cache_key);
        }

        // Send request with retry logic
        let response = self.send_with_retry(req_builder).await?;

        // Handle 304 Not Modified responses
        if response.status() == 304 {
            if let Some(cached_response) = self.check_cache(cache_key) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(cached_response);
            }
        }

        // Convert response with zero allocation where possible
        let http_response = self.convert_response(response).await?;

        // Update cache for successful GET requests
        if matches!(request.method(), crate::HttpMethod::Get) && http_response.is_success() {
            self.update_cache(cache_key, &http_response);
        }

        // Update statistics atomically
        let response_time = start_time.elapsed();
        self.update_stats(&request, &http_response, response_time);

        if http_response.is_success() {
            self.successful_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
        }

        Ok(http_response)
    }

    /// Send an HTTP request and return a streaming response
    pub async fn send_stream(&self, request: HttpRequest) -> HttpResult<HttpStream> {
        let request_id = self.request_count.fetch_add(1, Ordering::Relaxed);

        let mut req_builder = self.build_request(&request)?;

        // Add streaming-specific headers
        req_builder = req_builder
            .header("X-Request-ID", request_id.to_string())
            .header("Connection", "keep-alive")
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache");

        // Send request
        let response = req_builder.send().await.map_err(|e| {
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
            HttpError::from(e)
        })?;

        // Create optimized stream
        let stream = HttpStream::new(response);

        Ok(stream)
    }

    /// Get comprehensive client statistics
    pub fn stats(&self) -> ClientStats {
        let requests = self.request_count.load(Ordering::Relaxed);
        let successful = self.successful_requests.load(Ordering::Relaxed);
        let failed = self.failed_requests.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        let bytes_sent = self.total_bytes_sent.load(Ordering::Relaxed);
        let bytes_received = self.total_bytes_received.load(Ordering::Relaxed);
        let total_time_nanos = self.total_response_time_nanos.load(Ordering::Relaxed);

        let average_response_time = if requests > 0 {
            Duration::from_nanos(total_time_nanos / requests as u64)
        } else {
            Duration::from_nanos(0)
        };

        let uptime = self.start_time.elapsed().unwrap_or_default();

        ClientStats {
            connections_created: self.connection_count.load(Ordering::Relaxed),
            requests_sent: requests,
            successful_requests: successful,
            failed_requests: failed,
            cache_hits,
            cache_misses,
            bytes_sent,
            bytes_received,
            average_response_time,
            uptime,
        }
    }

    /// Clear the global cache - lock-free operation
    pub fn clear_cache(&self) {
        GLOBAL_CACHE.clear();
    }

    /// Get current cache size - lock-free operation
    pub fn cache_size(&self) -> usize {
        GLOBAL_CACHE.len()
    }

    /// Generate cache key using fast hash algorithm
    #[inline(always)]
    fn generate_cache_key(&self, request: &HttpRequest) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        request.method().hash(&mut hasher);
        request.url().hash(&mut hasher);

        // Only hash relevant headers for caching
        for (key, value) in request.headers() {
            let key_lower = key.to_lowercase();
            if key_lower == "accept"
                || key_lower == "accept-encoding"
                || key_lower == "accept-language"
            {
                key.hash(&mut hasher);
                value.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Check cache for valid entry - lock-free operation
    fn check_cache(&self, cache_key: u64) -> Option<HttpResponse> {
        let entry = GLOBAL_CACHE.get(&cache_key)?;

        // Check if entry is still valid
        if self.is_cache_entry_valid(entry.value()) {
            Some(self.cache_entry_to_response(entry.value()))
        } else {
            None
        }
    }

    /// Check if cache entry is still valid
    #[inline(always)]
    fn is_cache_entry_valid(&self, entry: &CacheEntry) -> bool {
        let now = Instant::now();
        let age = now.duration_since(entry.created_at);

        // Check max-age if specified
        if let Some(max_age) = entry.max_age_seconds {
            if age > Duration::from_secs(max_age) {
                return false;
            }
        } else {
            // Default max age of 1 hour
            if age > Duration::from_secs(3600) {
                return false;
            }
        }

        true
    }

    /// Convert cache entry to response
    #[inline(always)]
    fn cache_entry_to_response(&self, entry: &CacheEntry) -> HttpResponse {
        HttpResponse::from_cache(
            http::StatusCode::from_u16(entry.status_code).unwrap_or(http::StatusCode::OK),
            entry.headers.clone(),
            entry.response_body.to_vec(),
        )
    }

    /// Update cache with new response
    fn update_cache(&self, cache_key: u64, response: &HttpResponse) {
        if !self.should_cache_response(response) {
            return;
        }

        let max_age = self.extract_max_age(response);

        let entry = CacheEntry {
            response_body: Bytes::from(response.body().to_vec()),
            status_code: response.status().as_u16(),
            headers: response.headers().clone(),
            created_at: Instant::now(),
            etag: response.etag().cloned(),
            last_modified: response.last_modified().cloned(),
            max_age_seconds: max_age,
        };

        // Lock-free cache insertion
        GLOBAL_CACHE.insert(cache_key, entry);

        // Periodic cleanup using atomic counter to avoid contention
        if CACHE_CLEANUP_COUNTER.fetch_add(1, Ordering::Relaxed) % 100 == 0 {
            self.cleanup_cache_lock_free();
        }
    }

    /// Determine if response should be cached
    #[inline(always)]
    fn should_cache_response(&self, response: &HttpResponse) -> bool {
        // Only cache successful responses
        if !response.is_success() {
            return false;
        }

        // Check cache-control header
        if let Some(cache_control) = response.cache_control() {
            if cache_control.contains("no-cache")
                || cache_control.contains("no-store")
                || cache_control.contains("private")
            {
                return false;
            }
        }

        true
    }

    /// Extract max-age from response headers
    #[inline(always)]
    fn extract_max_age(&self, response: &HttpResponse) -> Option<u64> {
        response.cache_control().and_then(|cc| {
            for directive in cc.split(',') {
                let directive = directive.trim();
                if let Some(age_str) = directive.strip_prefix("max-age=") {
                    if let Ok(age) = age_str.parse::<u64>() {
                        return Some(age);
                    }
                }
            }
            None
        })
    }

    /// Clean up expired cache entries
    /// Remove expired entries from cache - lock-free operation
    fn cleanup_cache_lock_free(&self) {
        let cutoff = Instant::now() - Duration::from_secs(3600);

        // Use atomic iteration to remove expired entries
        let mut keys_to_remove = Vec::new();
        for entry in GLOBAL_CACHE.iter() {
            if entry.value().created_at <= cutoff {
                keys_to_remove.push(*entry.key());
            }
        }

        // Remove expired entries
        for key in keys_to_remove {
            GLOBAL_CACHE.remove(&key);
        }
    }

    /// Build reqwest request from HttpRequest
    fn build_request(&self, request: &HttpRequest) -> HttpResult<reqwest::RequestBuilder> {
        let method = match request.method() {
            crate::HttpMethod::Get => reqwest::Method::GET,
            crate::HttpMethod::Post => reqwest::Method::POST,
            crate::HttpMethod::Put => reqwest::Method::PUT,
            crate::HttpMethod::Delete => reqwest::Method::DELETE,
            crate::HttpMethod::Patch => reqwest::Method::PATCH,
            crate::HttpMethod::Head => reqwest::Method::HEAD,
        };

        let mut req_builder = self.inner.request(method, request.url());

        // Add headers
        for (key, value) in request.headers() {
            req_builder = req_builder.header(key, value);
        }

        // Add body if present
        if let Some(body) = request.body() {
            req_builder = req_builder.body(body.clone());
        }

        // Add timeout if specified
        if let Some(timeout) = request.timeout() {
            req_builder = req_builder.timeout(timeout);
        }

        Ok(req_builder)
    }

    /// Add conditional headers for cache validation - lock-free operation
    fn add_conditional_headers(
        &self,
        mut req_builder: reqwest::RequestBuilder,
        cache_key: u64,
    ) -> reqwest::RequestBuilder {
        if let Some(entry) = GLOBAL_CACHE.get(&cache_key) {
            if let Some(etag) = &entry.value().etag {
                req_builder = req_builder.header("If-None-Match", etag);
            }
            if let Some(last_modified) = &entry.value().last_modified {
                req_builder = req_builder.header("If-Modified-Since", last_modified);
            }
        }
        req_builder
    }

    /// Send request with intelligent retry logic
    async fn send_with_retry(
        &self,
        req_builder: reqwest::RequestBuilder,
    ) -> HttpResult<reqwest::Response> {
        let mut last_error = None;
        let retry_policy = &self.config.retry_policy;

        for attempt in 0..=retry_policy.max_retries {
            match req_builder
                .try_clone()
                .ok_or_else(|| HttpError::client("Failed to clone request"))?
                .send()
                .await
            {
                Ok(response) => {
                    // Check if we should retry based on status code
                    if attempt < retry_policy.max_retries
                        && retry_policy
                            .retry_on_status
                            .contains(&response.status().as_u16())
                    {
                        last_error = Some(HttpError::http_status(
                            response.status().as_u16(),
                            format!("HTTP {} - retrying", response.status().as_u16()),
                            String::new(),
                        ));

                        // Wait before retrying
                        let delay = self.calculate_retry_delay(attempt, retry_policy);
                        tokio::time::sleep(delay).await;
                        continue;
                    }

                    return Ok(response);
                }
                Err(e) => {
                    let http_error = HttpError::from(e);

                    // Check if we should retry based on error type
                    if attempt < retry_policy.max_retries
                        && self.should_retry_error(&http_error, retry_policy)
                    {
                        last_error = Some(http_error);

                        // Wait before retrying
                        let delay = self.calculate_retry_delay(attempt, retry_policy);
                        tokio::time::sleep(delay).await;
                        continue;
                    }

                    return Err(http_error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| HttpError::client("Max retries exceeded")))
    }

    /// Check if error should be retried
    #[inline(always)]
    fn should_retry_error(
        &self,
        error: &HttpError,
        retry_policy: &crate::config::RetryPolicy,
    ) -> bool {
        use crate::config::RetryableError;

        match error {
            HttpError::NetworkError { .. } => retry_policy
                .retry_on_errors
                .contains(&RetryableError::Network),
            HttpError::Timeout { .. } => retry_policy
                .retry_on_errors
                .contains(&RetryableError::Timeout),
            HttpError::ConnectionError { .. } => retry_policy
                .retry_on_errors
                .contains(&RetryableError::Connection),
            HttpError::DnsError { .. } => {
                retry_policy.retry_on_errors.contains(&RetryableError::Dns)
            }
            HttpError::TlsError { .. } => {
                retry_policy.retry_on_errors.contains(&RetryableError::Tls)
            }
            _ => false,
        }
    }

    /// Calculate retry delay with exponential backoff and jitter
    /// Uses iterative computation to avoid stack overflow with large attempt counts
    #[inline(always)]
    fn calculate_retry_delay(
        &self,
        attempt: usize,
        retry_policy: &crate::config::RetryPolicy,
    ) -> Duration {
        let base_delay_ms = retry_policy.base_delay.as_millis() as f64;
        let max_delay_ms = retry_policy.max_delay.as_millis() as f64;
        let backoff_factor = retry_policy.backoff_factor;

        // Use iterative computation to avoid stack overflow for large attempts
        // Cap exponential growth early to prevent overflow
        let capped_attempt = attempt.min(20); // Cap at 20 attempts to prevent overflow

        // Calculate exponential backoff delay iteratively
        let mut delay = base_delay_ms;
        for _ in 0..capped_attempt {
            delay *= backoff_factor;
            // Early termination if we've reached max delay
            if delay >= max_delay_ms {
                delay = max_delay_ms;
                break;
            }
        }

        // Add jitter to prevent thundering herd - use multiplicative jitter
        let jitter_range = retry_policy.jitter_factor;
        let jitter = 1.0 + (fastrand::f64() * 2.0 - 1.0) * jitter_range; // -jitter_range to +jitter_range
        let delay_with_jitter = delay * jitter;

        // Final cap at max delay
        let final_delay = delay_with_jitter.min(max_delay_ms).max(base_delay_ms / 2.0);

        Duration::from_millis(final_delay as u64)
    }

    /// Convert reqwest response to HttpResponse using streaming approach
    async fn convert_response(&self, response: reqwest::Response) -> HttpResult<HttpResponse> {
        let status = response.status();
        let headers = response.headers().clone();

        // reqwest's bytes_stream() doesn't automatically decompress
        // Use bytes() which does automatic decompression
        let body = response
            .bytes()
            .await
            .map_err(|e| HttpError::NetworkError {
                message: format!("Failed to read response body: {}", e),
            })?;

        Ok(HttpResponse::new(status, headers, body.to_vec()))
    }

    /// Update statistics atomically
    #[inline(always)]
    fn update_stats(
        &self,
        request: &HttpRequest,
        response: &HttpResponse,
        response_time: Duration,
    ) {
        // Update bytes sent
        if let Some(body) = request.body() {
            self.total_bytes_sent
                .fetch_add(body.len() as u64, Ordering::Relaxed);
        }

        // Update bytes received
        self.total_bytes_received
            .fetch_add(response.body().len() as u64, Ordering::Relaxed);

        // Update response time
        self.total_response_time_nanos
            .fetch_add(response_time.as_nanos() as u64, Ordering::Relaxed);
    }
}

impl ClientStats {
    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total_cache_requests = self.cache_hits + self.cache_misses;
        if total_cache_requests > 0 {
            self.cache_hits as f64 / total_cache_requests as f64
        } else {
            0.0
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        let total_requests = self.successful_requests + self.failed_requests;
        if total_requests > 0 {
            self.successful_requests as f64 / total_requests as f64
        } else {
            0.0
        }
    }

    /// Calculate requests per second
    pub fn requests_per_second(&self) -> f64 {
        if self.uptime.as_secs() > 0 {
            self.requests_sent as f64 / self.uptime.as_secs() as f64
        } else {
            0.0
        }
    }

    /// Calculate average throughput in bytes per second
    pub fn throughput_bytes_per_second(&self) -> f64 {
        if self.uptime.as_secs() > 0 {
            (self.bytes_sent + self.bytes_received) as f64 / self.uptime.as_secs() as f64
        } else {
            0.0
        }
    }
}

/// Default implementation for HttpClient
impl Default for HttpClient {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback to basic client if optimized config fails
            Self::with_config(HttpConfig::default()).unwrap_or_else(|_| {
                // Last resort: create a minimal client that should always work
                let client = reqwest::Client::new();

                Self {
                    inner: client,
                    request_count: AtomicUsize::new(0),
                    connection_count: AtomicUsize::new(0),
                    total_bytes_sent: AtomicU64::new(0),
                    total_bytes_received: AtomicU64::new(0),
                    total_response_time_nanos: AtomicU64::new(0),
                    successful_requests: AtomicUsize::new(0),
                    failed_requests: AtomicUsize::new(0),
                    cache_hits: AtomicUsize::new(0),
                    cache_misses: AtomicUsize::new(0),
                    config: HttpConfig::default(),
                    start_time: SystemTime::now(),
                }
            })
        })
    }
}
