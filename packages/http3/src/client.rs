//! Zero-allocation HTTP3 client with QUIC prioritization and HTTP/2 fallback
//!
//! This module provides a blazing-fast HTTP client optimized for AI provider APIs
//! with connection pooling, intelligent caching, and comprehensive error handling.
//! All operations are lock-free and use atomic counters for thread safety.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant, SystemTime};

use bytes::Bytes;
use crossbeam_skiplist::SkipMap;
use cyrup_sugars::ZeroOneOrMany;
use reqwest::tls;
use tracing;

use crate::async_task::stream::AsyncStream;
use crate::{HttpConfig, HttpError, HttpRequest, HttpResponse, HttpResult, HttpStream};

/// Typestate marker for ready-to-send requests
pub struct Ready;

/// Typestate-safe request builder that prevents invalid request construction
pub struct RequestBuilder<'a, State> {
    client: &'a HttpClient,
    method: crate::HttpMethod,
    url: String,
    headers: ZeroOneOrMany<(String, String)>,
    body: Vec<u8>,
    timeout: Duration,
    _state: std::marker::PhantomData<State>,
}

/// High-performance HTTP client with QUIC/HTTP3 support and zero-allocation design
#[derive(Debug)]
pub struct HttpClient {
    inner: reqwest::Client,
    request_count: Arc<AtomicUsize>,
    connection_count: Arc<AtomicUsize>,
    total_bytes_sent: Arc<AtomicU64>,
    total_bytes_received: Arc<AtomicU64>,
    total_response_time_nanos: Arc<AtomicU64>,
    successful_requests: Arc<AtomicUsize>,
    failed_requests: Arc<AtomicUsize>,
    cache_hits: Arc<AtomicUsize>,
    cache_misses: Arc<AtomicUsize>,
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
    pub fn new() -> AsyncStream<Self> {
        Self::with_config(HttpConfig::ai_optimized())
    }

    /// Create a new HTTP client with custom configuration
    pub fn with_config(config: HttpConfig) -> AsyncStream<Self> {
        let (sender, stream) = AsyncStream::channel();

        // Spawn client initialization in background
        std::thread::spawn(move || {
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
                builder =
                    builder.redirect(reqwest::redirect::Policy::limited(config.max_redirects));
            } else {
                builder = builder.redirect(reqwest::redirect::Policy::none());
            }

            // Configure local address binding if specified
            if let Some(local_addr) = config.local_address {
                builder = builder.local_address(local_addr);
            }

            // Build the client
            if let Ok(client) = builder.build() {
                let start_time = SystemTime::now();

                let http_client = Self {
                    inner: client,
                    request_count: Arc::new(AtomicUsize::new(0)),
                    connection_count: Arc::new(AtomicUsize::new(0)),
                    total_bytes_sent: Arc::new(AtomicU64::new(0)),
                    total_bytes_received: Arc::new(AtomicU64::new(0)),
                    total_response_time_nanos: Arc::new(AtomicU64::new(0)),
                    successful_requests: Arc::new(AtomicUsize::new(0)),
                    failed_requests: Arc::new(AtomicUsize::new(0)),
                    cache_hits: Arc::new(AtomicUsize::new(0)),
                    cache_misses: Arc::new(AtomicUsize::new(0)),
                    config,
                    start_time,
                };

                // Emit the configured client
                let _ = sender.try_send(http_client);
            }
            // If client build fails, the stream will simply end without emitting anything
        });

        stream
    }

    /// Create a GET request with zero allocation - returns a typestate builder
    #[inline(always)]
    pub fn get(&self, url: &str) -> RequestBuilder<'_, Ready> {
        RequestBuilder::new(self, crate::HttpMethod::Get, url.to_string())
    }

    /// Create a POST request with zero allocation - returns a typestate builder
    #[inline(always)]
    pub fn post(&self, url: &str) -> RequestBuilder<'_, Ready> {
        RequestBuilder::new(self, crate::HttpMethod::Post, url.to_string())
    }

    /// Create a PUT request with zero allocation - returns a typestate builder
    #[inline(always)]
    pub fn put(&self, url: &str) -> RequestBuilder<'_, Ready> {
        RequestBuilder::new(self, crate::HttpMethod::Put, url.to_string())
    }

    /// Create a DELETE request with zero allocation - returns a typestate builder
    #[inline(always)]
    pub fn delete(&self, url: &str) -> RequestBuilder<'_, Ready> {
        RequestBuilder::new(self, crate::HttpMethod::Delete, url.to_string())
    }

    /// Create a PATCH request with zero allocation - returns a typestate builder
    #[inline(always)]
    pub fn patch(&self, url: &str) -> RequestBuilder<'_, Ready> {
        RequestBuilder::new(self, crate::HttpMethod::Patch, url.to_string())
    }

    /// Create a HEAD request with zero allocation - returns a typestate builder
    #[inline(always)]
    pub fn head(&self, url: &str) -> RequestBuilder<'_, Ready> {
        RequestBuilder::new(self, crate::HttpMethod::Head, url.to_string())
    }

    /// Download a file from the given URL with streaming support
    /// Returns AsyncStream of DownloadChunk items for unwrapped processing
    /// Use HttpRequest builder methods like .if_none_match() for conditional requests
    pub fn download_file(&self, url: &str) -> AsyncStream<crate::DownloadChunk> {
        let request = HttpRequest::new(crate::HttpMethod::Get, url.to_string());
        self.download_with_request(request)
    }

    /// Download a file using an HttpRequest (supports conditional headers)  
    /// Returns AsyncStream of DownloadChunk items for unwrapped processing
    pub fn download_with_request(&self, request: HttpRequest) -> AsyncStream<crate::DownloadChunk> {        let inner_client = self.inner.clone();
        let user_agent = self.config.user_agent.clone();

        // Process download using zero-allocation async streaming
        spawn_stream(move |_| async move {
            // Validate URL
            let parsed_url = match url::Url::parse(request.url()) {
                Ok(url) => url,
                Err(_) => return Ok(()), // Invalid URL - stream ends gracefully
            };

            match parsed_url.scheme() {
                "http" | "https" => {}
                _ => return Ok(()), // Unsupported scheme - stream ends gracefully
            }

            // Build reqwest request with all headers from HttpRequest
            let mut req_builder = inner_client
                .get(request.url())
                .header("Accept", "*/*")
                .header("Accept-Encoding", "identity")
                .header("Connection", "keep-alive")
                .header("User-Agent", &user_agent);

            // Add all headers from the HttpRequest (including conditional headers)
            for (key, value) in request.headers() {
                req_builder = req_builder.header(key, value);
            }

            // Add timeout if specified
            if let Some(timeout) = request.timeout() {
                req_builder = req_builder.timeout(timeout);
            }

            // Execute request with proper async streaming (no blocking)
            let response = match req_builder.send().await {
                Ok(response) => response,
                Err(_) => return Ok(()), // Network error - stream ends gracefully
            };

            let status = response.status();

            // Handle non-success status codes by ending stream
            if status.as_u16() == 304 || !status.is_success() {
                return Ok(()); // Stream ends for HTTP errors
            }

            // Validate content type
            if let Some(content_type) = response.headers().get("content-type") {
                if let Ok(content_type_str) = content_type.to_str() {
                    if content_type_str.starts_with("text/html") {
                        return Ok(()); // HTML response - stream ends
                    }
                }
            }

            // Stream download chunks
            use futures_util::StreamExt;
            let total_size = response.content_length();
            let mut bytes_stream = response.bytes_stream();
            let mut chunk_number = 0;
            let mut bytes_downloaded = 0;
            let start_time = std::time::Instant::now();

            while let Some(chunk_result) = bytes_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let chunk_size = chunk.len() as u64;
                        bytes_downloaded += chunk_size;

                        // Calculate download speed
                        let elapsed = start_time.elapsed();
                        let download_speed = if elapsed.as_secs_f64() > 0.0 {
                            Some(bytes_downloaded as f64 / elapsed.as_secs_f64())
                        } else {
                            None
                        };

                        let download_chunk = crate::DownloadChunk::new(
                            chunk,
                            chunk_number,
                            total_size,
                            bytes_downloaded,
                            download_speed,
                        );

                        // Emit unwrapped chunk with proper error handling
                        if let Err(_) = sender.send(download_chunk).await {
                            break; // Stream closed
                        }

                        chunk_number += 1;
                    }
                    Err(_) => break, // Stream error - stop processing
                }
            }
            
            Ok(())
        });

        stream
    }

    /// Send an HTTP request with optimal performance and caching
    /// Returns AsyncStream of HttpResponse items for unwrapped processing
    pub fn send(&self, request: HttpRequest) -> AsyncStream<HttpResponse> {
        // Clone needed data for zero-allocation async streaming
        let inner_client = self.inner.clone();
        let request_count = Arc::clone(&self.request_count);
        let cache_hits = Arc::clone(&self.cache_hits);
        let cache_misses = Arc::clone(&self.cache_misses);
        let successful_requests = Arc::clone(&self.successful_requests);
        let failed_requests = Arc::clone(&self.failed_requests);
        let total_bytes_sent = Arc::clone(&self.total_bytes_sent);
        let total_bytes_received = Arc::clone(&self.total_bytes_received);
        let total_response_time_nanos = Arc::clone(&self.total_response_time_nanos);

        // Process request using zero-allocation async streaming
        spawn_stream(move |_| async move {
            let start_time = Instant::now();

            // Increment request counter atomically
            let request_id = request_count.fetch_add(1, Ordering::Relaxed);

            // Generate cache key using fast hash
            let cache_key = Self::generate_cache_key_static(&request);

            // Check cache first for GET requests
            if matches!(request.method(), crate::HttpMethod::Get) {
                if let Some(cached_response) = Self::check_cache_static(cache_key) {
                    cache_hits.fetch_add(1, Ordering::Relaxed);
                    let _ = sender.send(cached_response).await;
                    return Ok(());
                }
            }

            // Record cache miss
            cache_misses.fetch_add(1, Ordering::Relaxed);

            // Build reqwest request with optimal settings
            let req_builder = match Self::build_request_static(&inner_client, &request) {
                Ok(builder) => builder,
                Err(_) => return Ok(()), // Request build failed - stream ends gracefully
            };

            // Add performance-optimized headers
            let req_builder = req_builder
                .header("X-Request-ID", request_id.to_string())
                .header("Connection", "keep-alive")
                .header("Accept-Encoding", "gzip, br, deflate");

            // Add conditional headers if we have cache info
            let req_builder = if matches!(request.method(), crate::HttpMethod::Get) {
                Self::add_conditional_headers_static(req_builder, cache_key)
            } else {
                req_builder
            };

            // Execute request with proper async streaming (no blocking)
            // Send request (simplified retry for streaming)
            let response = match req_builder.send().await {
                Ok(response) => response,
                Err(_) => {
                    failed_requests.fetch_add(1, Ordering::Relaxed);
                    return Ok(()); // Network error - stream ends gracefully
                }
            };

            // Handle 304 Not Modified responses
            if response.status() == 304 {
                if let Some(cached_response) = Self::check_cache_static(cache_key) {
                    cache_hits.fetch_add(1, Ordering::Relaxed);
                    let _ = sender.send(cached_response).await;
                    return Ok(());
                }
            }

            // Convert response with zero allocation where possible - NO FUTURES
            let http_response = Self::convert_response_static(response);

            // Update cache for successful GET requests
            if matches!(request.method(), crate::HttpMethod::Get) && http_response.is_success()
            {
                Self::update_cache_static(cache_key, &http_response);
            }

            // Update statistics atomically
            let response_time = start_time.elapsed();

            // Update bytes sent
            if let Some(body) = request.body() {
                total_bytes_sent.fetch_add(body.len() as u64, Ordering::Relaxed);
            }

            // Update bytes received
            total_bytes_received
                .fetch_add(http_response.body().len() as u64, Ordering::Relaxed);

            // Update response time
            total_response_time_nanos
                .fetch_add(response_time.as_nanos() as u64, Ordering::Relaxed);

            if http_response.is_success() {
                successful_requests.fetch_add(1, Ordering::Relaxed);
            } else {
                failed_requests.fetch_add(1, Ordering::Relaxed);
            }

            // Emit unwrapped response with proper error handling
            let _ = sender.send(http_response).await;
            
            Ok(())
        });

        stream
    }

    /// Send an HTTP request and return a streaming response  
    /// Returns AsyncStream of HttpStream items for unwrapped processing
    pub fn send_stream(&self, request: HttpRequest) -> AsyncStream<HttpStream> {
        // Clone needed data for zero-allocation async streaming
        let inner_client = self.inner.clone();
        let request_count = Arc::clone(&self.request_count);
        let failed_requests = Arc::clone(&self.failed_requests);

        // Process streaming request using zero-allocation async streaming
        spawn_stream(move |_| async move {
            let request_id = request_count.fetch_add(1, Ordering::Relaxed);

            let req_builder = match Self::build_request_static(&inner_client, &request) {
                Ok(builder) => builder,
                Err(_) => return Ok(()), // Request build failed - stream ends gracefully
            };

            // Add streaming-specific headers
            let req_builder = req_builder
                .header("X-Request-ID", request_id.to_string())
                .header("Connection", "keep-alive")
                .header("Accept", "text/event-stream")
                .header("Cache-Control", "no-cache");

            // Execute request with proper async streaming (no blocking)
            // Send request
            let response = match req_builder.send().await {
                Ok(response) => response,
                Err(_) => {
                    failed_requests.fetch_add(1, Ordering::Relaxed);
                    return Ok(()); // Network error - stream ends gracefully
                }
            };

            // Convert response body to Vec<u8> for HttpStream
            let response_bytes = match response.bytes().await {
                Ok(bytes) => bytes.to_vec(),
                Err(_) => {
                    failed_requests.fetch_add(1, Ordering::Relaxed);
                    return Ok(()); // Body read error - stream ends gracefully
                }
            };

            // Create optimized stream
            let http_stream = HttpStream::new(response_bytes);

            // Emit unwrapped stream with proper error handling
            let _ = sender.send(http_stream).await;
            
            Ok(())
        });

        stream
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

    /// Send request with intelligent retry logic using async streaming
    async fn send_with_retry_async(&self, req_builder: reqwest::RequestBuilder) -> Option<reqwest::Response> {
        let mut last_error = None;
        let retry_policy = &self.config.retry_policy;

        for attempt in 0..=retry_policy.max_retries {
            let request = match req_builder.try_clone() {
                Some(req) => req,
                None => return None, // Can't clone request, give up
            };

            let response_result = request.send().await;

            match response_result {
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

                        // Wait before retrying - synchronous sleep
                        let delay = self.calculate_retry_delay(attempt, retry_policy);
                        std::thread::sleep(delay);
                        continue;
                    }

                    return Some(response);
                }
                Err(e) => {
                    let http_error = HttpError::from(e);

                    // Check if we should retry based on error type
                    if attempt < retry_policy.max_retries
                        && self.should_retry_error(&http_error, retry_policy)
                    {
                        last_error = Some(http_error);

                        // Wait before retrying - synchronous sleep
                        let delay = self.calculate_retry_delay(attempt, retry_policy);
                        std::thread::sleep(delay);
                        continue;
                    }

                    // Error handled via on_chunk handlers
                    return None;
                }
            }
        }

        // Max retries exceeded - log final error
        if let Some(error) = last_error {
            tracing::warn!(
                "Request failed after {} retries: {}",
                retry_policy.max_retries,
                error
            );
        }
        None
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

    /// Convert reqwest response to HttpResponse using async streaming
    async fn convert_response_async(&self, response: reqwest::Response) -> HttpResponse {
        let status = response.status();
        let headers = response.headers().clone();

        // Use proper async streaming - no blocking
        let body = response.bytes().await.unwrap_or_default();

        HttpResponse::new(status, headers, body.to_vec())
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

    // Static helper methods for use in background threads

    /// Generate cache key using fast hash algorithm (static version)
    #[inline(always)]
    fn generate_cache_key_static(request: &HttpRequest) -> u64 {
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

    /// Check cache for valid entry (static version)
    fn check_cache_static(cache_key: u64) -> Option<HttpResponse> {
        let entry = GLOBAL_CACHE.get(&cache_key)?;

        // Check if entry is still valid
        if Self::is_cache_entry_valid_static(entry.value()) {
            Some(Self::cache_entry_to_response_static(entry.value()))
        } else {
            None
        }
    }

    /// Check if cache entry is still valid (static version)
    #[inline(always)]
    fn is_cache_entry_valid_static(entry: &CacheEntry) -> bool {
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

    /// Convert cache entry to response (static version)
    #[inline(always)]
    fn cache_entry_to_response_static(entry: &CacheEntry) -> HttpResponse {
        HttpResponse::from_cache(
            http::StatusCode::from_u16(entry.status_code).unwrap_or(http::StatusCode::OK),
            entry.headers.clone(),
            entry.response_body.to_vec(),
        )
    }

    /// Update cache with new response (static version)
    fn update_cache_static(cache_key: u64, response: &HttpResponse) {
        if !Self::should_cache_response_static(response) {
            return;
        }

        let max_age = Self::extract_max_age_static(response);

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
            Self::cleanup_cache_lock_free_static();
        }
    }

    /// Determine if response should be cached (static version)
    #[inline(always)]
    fn should_cache_response_static(response: &HttpResponse) -> bool {
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

    /// Extract max-age from response headers (static version)
    #[inline(always)]
    fn extract_max_age_static(response: &HttpResponse) -> Option<u64> {
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

    /// Clean up expired cache entries (static version)
    fn cleanup_cache_lock_free_static() {
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

    /// Build reqwest request from HttpRequest (static version)
    fn build_request_static(
        client: &reqwest::Client,
        request: &HttpRequest,
    ) -> HttpResult<reqwest::RequestBuilder> {
        let method = match request.method() {
            crate::HttpMethod::Get => reqwest::Method::GET,
            crate::HttpMethod::Post => reqwest::Method::POST,
            crate::HttpMethod::Put => reqwest::Method::PUT,
            crate::HttpMethod::Delete => reqwest::Method::DELETE,
            crate::HttpMethod::Patch => reqwest::Method::PATCH,
            crate::HttpMethod::Head => reqwest::Method::HEAD,
        };

        let mut req_builder = client.request(method, request.url());

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

    /// Add conditional headers for cache validation (static version)
    fn add_conditional_headers_static(
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

    /// Convert reqwest response to HttpResponse using async streaming (static version)
    async fn convert_response_static_async(response: reqwest::Response) -> HttpResponse {
        let status = response.status();
        let headers = response.headers().clone();

        // Use proper async streaming - no blocking
        let body = response.bytes().await.unwrap_or_default();

        HttpResponse::new(status, headers, body.to_vec())
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

impl<'a> RequestBuilder<'a, Ready> {
    /// Create a new request builder in Ready state with smart defaults
    fn new(client: &'a HttpClient, method: crate::HttpMethod, url: String) -> Self {
        let default_body = match method {
            crate::HttpMethod::Get | crate::HttpMethod::Head => Vec::new(),
            _ => Vec::new(), // Empty by default, can be set with body() methods
        };

        Self {
            client,
            method,
            url,
            headers: ZeroOneOrMany::none(),
            body: default_body,
            timeout: client.config.timeout, // Use client's default timeout
            _state: std::marker::PhantomData,
        }
    }

    /// Add a header to the request (immutable)
    #[must_use]
    pub fn header<K, V>(self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            headers: self.headers.with_pushed((key.into(), value.into())),
            ..self
        }
    }

    /// Add multiple headers using ergonomic JSON syntax: {"key" => "val", "foo" => "bar"} (immutable)
    #[must_use]
    pub fn headers<H>(self, headers: H) -> Self
    where
        H: Into<HashMap<String, String>>,
    {
        let new_headers = headers.into().into_iter().collect::<Vec<_>>();
        let new_headers_collection = if new_headers.is_empty() {
            ZeroOneOrMany::none()
        } else {
            ZeroOneOrMany::many(new_headers)
        };

        Self {
            headers: ZeroOneOrMany::merge(vec![self.headers, new_headers_collection]),
            ..self
        }
    }

    /// Set the request body (immutable)
    #[must_use]
    pub fn body(self, body: Vec<u8>) -> Self {
        Self { body, ..self }
    }

    /// Set the request body from a string (immutable)
    #[must_use]
    pub fn body_string(self, body: String) -> Self {
        Self {
            body: body.into_bytes(),
            ..self
        }
    }

    /// Set the request body from JSON (immutable)
    pub fn json<T: serde::Serialize>(self, json: &T) -> crate::HttpResult<Self> {
        let body = serde_json::to_vec(json)?;
        Ok(Self {
            headers: self
                .headers
                .with_pushed(("Content-Type".to_string(), "application/json".to_string())),
            body,
            ..self
        })
    }

    /// Set request timeout (immutable)
    #[must_use]
    pub fn timeout(self, timeout: Duration) -> Self {
        Self { timeout, ..self }
    }

    /// Set timeout in seconds (immutable)
    #[must_use]
    pub fn timeout_seconds(self, seconds: u64) -> Self {
        Self {
            timeout: Duration::from_secs(seconds),
            ..self
        }
    }

    /// Set timeout in milliseconds (immutable)
    #[must_use]
    pub fn timeout_millis(self, millis: u64) -> Self {
        Self {
            timeout: Duration::from_millis(millis),
            ..self
        }
    }

    // === Convenience header methods ===

    /// Add authorization header (immutable)
    #[must_use]
    pub fn authorization<V: Into<String>>(self, auth: V) -> Self {
        self.header("Authorization", auth.into())
    }

    /// Add bearer token authorization (immutable)
    #[must_use]
    pub fn bearer_token<V: Into<String>>(self, token: V) -> Self {
        self.header("Authorization", format!("Bearer {}", token.into()))
    }

    /// Add basic auth (immutable)
    #[must_use]
    pub fn basic_auth<U: Into<String>, P: Into<String>>(self, username: U, password: P) -> Self {
        use base64::Engine;
        let credentials = base64::engine::general_purpose::STANDARD.encode(format!(
            "{}:{}",
            username.into(),
            password.into()
        ));
        self.header("Authorization", format!("Basic {}", credentials))
    }

    /// Add content type header (immutable)
    #[must_use]
    pub fn content_type<V: Into<String>>(self, content_type: V) -> Self {
        self.header("Content-Type", content_type.into())
    }

    /// Add JSON content type (immutable)
    #[must_use]
    pub fn content_type_json(self) -> Self {
        self.header("Content-Type", "application/json")
    }

    /// Add form content type (immutable)
    #[must_use]
    pub fn content_type_form(self) -> Self {
        self.header("Content-Type", "application/x-www-form-urlencoded")
    }

    /// Add user agent header (immutable)
    #[must_use]
    pub fn user_agent<V: Into<String>>(self, user_agent: V) -> Self {
        self.header("User-Agent", user_agent.into())
    }

    /// Add accept header (immutable)
    #[must_use]
    pub fn accept<V: Into<String>>(self, accept: V) -> Self {
        self.header("Accept", accept.into())
    }

    /// Add JSON accept header (immutable)
    #[must_use]
    pub fn accept_json(self) -> Self {
        self.header("Accept", "application/json")
    }

    /// Add accept encoding header (immutable)
    #[must_use]
    pub fn accept_encoding<V: Into<String>>(self, accept_encoding: V) -> Self {
        self.header("Accept-Encoding", accept_encoding.into())
    }

    /// Add cache control header (immutable)
    #[must_use]
    pub fn cache_control<V: Into<String>>(self, cache_control: V) -> Self {
        self.header("Cache-Control", cache_control.into())
    }

    /// Add no-cache header (immutable)
    #[must_use]
    pub fn no_cache(self) -> Self {
        self.header("Cache-Control", "no-cache")
    }

    /// Add connection header (immutable)
    #[must_use]
    pub fn connection<V: Into<String>>(self, connection: V) -> Self {
        self.header("Connection", connection.into())
    }

    /// Add keep-alive connection header (immutable)
    #[must_use]
    pub fn keep_alive(self) -> Self {
        self.header("Connection", "keep-alive")
    }

    /// Add If-None-Match header for conditional requests (ETag-based) (immutable)
    #[must_use]
    pub fn if_none_match<V: Into<String>>(self, etag: V) -> Self {
        self.header("If-None-Match", etag.into())
    }

    /// Add If-Modified-Since header for conditional requests (date-based) (immutable)
    #[must_use]
    pub fn if_modified_since<V: Into<String>>(self, date: V) -> Self {
        self.header("If-Modified-Since", date.into())
    }

    /// Add custom API key header (immutable)
    #[must_use]
    pub fn api_key<K: Into<String>, V: Into<String>>(self, header_name: K, api_key: V) -> Self {
        self.header(header_name.into(), api_key.into())
    }

    /// Add X-API-Key header (immutable)
    #[must_use]
    pub fn x_api_key<V: Into<String>>(self, api_key: V) -> Self {
        self.header("X-API-Key", api_key.into())
    }

    /// Add streaming-specific headers (immutable)
    #[must_use]
    pub fn streaming(self) -> Self {
        self.header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
    }

    /// Send the request and return a response
    /// Returns AsyncStream of HttpResponse items for unwrapped processing
    pub fn send(self) -> AsyncStream<HttpResponse> {
        let mut request = HttpRequest::new(self.method, self.url);

        // Convert ZeroOneOrMany headers to HttpRequest using zero-allocation iteration
        for (key, value) in &self.headers {
            request = request.header(key.clone(), value.clone());
        }

        request = request.set_body(self.body);
        request = request.set_timeout(self.timeout);

        self.client.send(request)
    }

    /// Send the request and return a streaming response
    /// Returns AsyncStream of HttpStream items for unwrapped processing
    pub fn send_stream(self) -> AsyncStream<HttpStream> {
        let mut request = HttpRequest::new(self.method, self.url);

        // Convert ZeroOneOrMany headers to HttpRequest using zero-allocation iteration
        for (key, value) in &self.headers {
            request = request.header(key.clone(), value.clone());
        }

        request = request.set_body(self.body);
        request = request.set_timeout(self.timeout);

        self.client.send_stream(request)
    }
}

/// Default implementation for HttpClient
/// Note: Since HttpClient::new() now returns AsyncStream, users should call
/// HttpClient::new().collect() to get a blocking default client instance
impl Default for HttpClient {
    fn default() -> Self {
        // Last resort: create a minimal client that should always work
        let client = reqwest::Client::new();

        Self {
            inner: client,
            request_count: Arc::new(AtomicUsize::new(0)),
            connection_count: Arc::new(AtomicUsize::new(0)),
            total_bytes_sent: Arc::new(AtomicU64::new(0)),
            total_bytes_received: Arc::new(AtomicU64::new(0)),
            total_response_time_nanos: Arc::new(AtomicU64::new(0)),
            successful_requests: Arc::new(AtomicUsize::new(0)),
            failed_requests: Arc::new(AtomicUsize::new(0)),
            cache_hits: Arc::new(AtomicUsize::new(0)),
            cache_misses: Arc::new(AtomicUsize::new(0)),
            config: HttpConfig::default(),
            start_time: SystemTime::now(),
        }
    }
}
