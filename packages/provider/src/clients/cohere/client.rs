//! Cohere multi-endpoint client with shared HTTP architecture
//!
//! Provides unified access to Cohere's three distinct APIs:
//! - Chat completions via /v1/chat
//! - Text embeddings via /v1/embed  
//! - Document reranking via /v1/rerank
//!
//! Features:
//! - Shared fluent_ai_http3::HttpClient with connection pooling
//! - Zero allocation endpoint routing and request building
//! - Hot-swappable API key management with arc_swap
//! - Atomic performance metrics and circuit breaker integration
//! - CompletionClient + ProviderClient trait implementations for auto-generation

use crate::completion_provider::{CompletionProvider, CompletionError, CompletionResponse};
use crate::client::{CompletionClient, ProviderClient};
use super::completion::CohereCompletionBuilder;
use super::embedding::{CohereEmbedding, EmbeddingRequest, EmbeddingResponse};
use super::reranker::{CohereReranker, RerankRequest, RerankResponse};
use super::error::{CohereError, Result, CohereOperation, AuthErrorCode};
use super::models;
use super::endpoints;
use super::config;
use super::ModelType;

use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest};
use fluent_ai_domain::{AsyncTask, AsyncStream};
use arc_swap::{ArcSwap, Guard};
use atomic_counter::{AtomicCounter, RelaxedCounter};
use arrayvec::{ArrayString};
use smallvec::{SmallVec, smallvec};
use circuit_breaker::{CircuitBreaker, Config as CBConfig, State as CBState};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Shared HTTP client for all Cohere endpoints with connection pooling
static HTTP_CLIENT: LazyLock<HttpClient> = LazyLock::new(|| {
    HttpClient::with_config(HttpConfig::ai_optimized())
        .unwrap_or_else(|_| HttpClient::new())
});

/// Lock-free performance metrics for all endpoints
static CHAT_METRICS: LazyLock<CohereMetrics> = LazyLock::new(|| CohereMetrics::new("chat"));
static EMBED_METRICS: LazyLock<CohereMetrics> = LazyLock::new(|| CohereMetrics::new("embed"));
static RERANK_METRICS: LazyLock<CohereMetrics> = LazyLock::new(|| CohereMetrics::new("rerank"));

/// Circuit breakers for fault tolerance across all endpoints
static CIRCUIT_BREAKERS: LazyLock<CohereCircuitBreakers> = LazyLock::new(CohereCircuitBreakers::new);

/// Multi-endpoint Cohere client with shared connection pool
#[derive(Clone)]
pub struct CohereClient {
    /// Hot-swappable API key using arc_swap for zero-contention updates
    api_key: ArcSwap<ArrayString<128>>,
    
    /// Shared HTTP client for all three endpoints
    http_client: &'static HttpClient,
    
    /// Endpoint routing configuration
    endpoint_router: EndpointRouter,
    
    /// Circuit breaker protection
    circuit_breakers: &'static CohereCircuitBreakers,
    
    /// Request timeout configuration
    timeout: Duration,
    
    /// Maximum retries for transient failures
    max_retries: u8}

/// Zero allocation endpoint routing with compile-time optimizations
#[derive(Debug, Clone)]
struct EndpointRouter {
    chat_url: &'static str,
    embed_url: &'static str,
    rerank_url: &'static str}

/// Lock-free performance metrics per endpoint
struct CohereMetrics {
    endpoint_name: &'static str,
    total_requests: RelaxedCounter,
    successful_requests: RelaxedCounter,
    failed_requests: RelaxedCounter,
    total_latency_ms: RelaxedCounter,
    active_requests: RelaxedCounter,
    peak_active_requests: RelaxedCounter}

/// Circuit breaker protection for all endpoints
struct CohereCircuitBreakers {
    chat: CircuitBreaker<CohereError>,
    embed: CircuitBreaker<CohereError>,
    rerank: CircuitBreaker<CohereError>}

impl CohereClient {
    /// Create new Cohere client with API key validation
    pub fn new(api_key: String) -> Result<Self> {
        // Validate API key format
        config::validate_api_key(&api_key)
            .map_err(|err| CohereError::InvalidApiKey {
                reason: err,
                key_length: api_key.len(),
                format_valid: !api_key.is_empty()})?;
        
        // Convert to fixed-size array string for zero allocation storage
        let api_key_array = ArrayString::from(&api_key)
            .map_err(|_| CohereError::InvalidApiKey {
                reason: ArrayString::from("API key too long").unwrap_or_default(),
                key_length: api_key.len(),
                format_valid: true})?;
        
        Ok(Self {
            api_key: ArcSwap::from_pointee(api_key_array),
            http_client: &HTTP_CLIENT,
            endpoint_router: EndpointRouter::new(),
            circuit_breakers: &CIRCUIT_BREAKERS,
            timeout: Duration::from_secs(config::DEFAULT_TIMEOUT_SECS),
            max_retries: config::MAX_RETRIES})
    }
    
    /// Update API key with zero downtime using hot-swapping
    #[inline]
    pub fn update_api_key(&self, new_api_key: String) -> Result<()> {
        config::validate_api_key(&new_api_key)
            .map_err(|err| CohereError::InvalidApiKey {
                reason: err,
                key_length: new_api_key.len(),
                format_valid: !new_api_key.is_empty()})?;
        
        let api_key_array = ArrayString::from(&new_api_key)
            .map_err(|_| CohereError::InvalidApiKey {
                reason: ArrayString::from("API key too long").unwrap_or_default(),
                key_length: new_api_key.len(),
                format_valid: true})?;
        
        self.api_key.store(Arc::new(api_key_array));
        Ok(())
    }
    
    /// Get current API key guard for request authentication
    #[inline]
    fn api_key_guard(&self) -> Guard<Arc<ArrayString<128>>> {
        self.api_key.load()
    }
    
    /// Route model to appropriate endpoint with compile-time optimization
    #[inline]
    const fn route_model(&self, model: &str) -> (&'static str, &'static CohereMetrics) {
        match models::get_model_type(model) {
            ModelType::Chat => (self.endpoint_router.chat_url, &CHAT_METRICS),
            ModelType::Embedding => (self.endpoint_router.embed_url, &EMBED_METRICS),
            ModelType::Reranking => (self.endpoint_router.rerank_url, &RERANK_METRICS),
            ModelType::Unknown => ("", &CHAT_METRICS), // Fallback, will error in validation
        }
    }
    
    /// Get circuit breaker for endpoint
    #[inline]
    fn get_circuit_breaker(&self, model_type: ModelType) -> &CircuitBreaker<CohereError> {
        match model_type {
            ModelType::Chat => &self.circuit_breakers.chat,
            ModelType::Embedding => &self.circuit_breakers.embed,
            ModelType::Reranking => &self.circuit_breakers.rerank,
            ModelType::Unknown => &self.circuit_breakers.chat, // Fallback
        }
    }
    
    /// Build authentication headers with zero allocation
    #[inline]
    fn build_auth_headers(&self) -> SmallVec<[(&'static str, ArrayString<140>); 4]> {
        let api_key_guard = self.api_key_guard();
        
        let mut auth_header = ArrayString::<140>::new();
        let _ = auth_header.try_push_str("Bearer ");
        let _ = auth_header.try_push_str(&api_key_guard);
        
        let user_agent = ArrayString::from(super::utils::user_agent()).unwrap_or_default();
        
        smallvec![
            ("Authorization", auth_header),
            ("Content-Type", ArrayString::from("application/json").unwrap_or_default()),
            ("User-Agent", user_agent),
            ("Accept", ArrayString::from("application/json").unwrap_or_default()),
        ]
    }
    
    /// Execute HTTP request with circuit breaker protection
    async fn execute_request_protected<T, F, Fut>(
        &self,
        model_type: ModelType,
        operation: F,
    ) -> Result<T>
    where
        F: FnOnce() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T>> + Send,
        T: Send,
    {
        let circuit_breaker = self.get_circuit_breaker(model_type);
        
        match circuit_breaker.state() {
            CBState::Closed | CBState::HalfOpen => {
                match operation().await {
                    Ok(result) => {
                        circuit_breaker.record_success();
                        Ok(result)
                    }
                    Err(error) => {
                        circuit_breaker.record_failure();
                        Err(error)
                    }
                }
            }
            CBState::Open => {
                Err(CohereError::CircuitBreaker {
                    endpoint: ArrayString::from(match model_type {
                        ModelType::Chat => "chat",
                        ModelType::Embedding => "embed",
                        ModelType::Reranking => "rerank",
                        ModelType::Unknown => "unknown"}).unwrap_or_default(),
                    state: super::error::CircuitBreakerState::Open,
                    failure_count: 0, // Circuit breaker doesn't expose this
                    success_count: 0, // Circuit breaker doesn't expose this
                    next_retry_ms: 60000, // Default 1 minute
                })
            }
        }
    }
    
    /// Validate API key by making a lightweight request
    #[inline]
    pub async fn validate_api_key(&self) -> Result<()> {
        // Use chat endpoint with minimal request for validation
        let headers = self.build_auth_headers();
        let request_body = br#"{"model":"command-r7b-12-2024","message":"test","max_tokens":1}"#;
        
        let http_request = HttpRequest::post(self.endpoint_router.chat_url, request_body.to_vec())
            .map_err(|e| CohereError::Configuration {
                setting: ArrayString::from("http_request").unwrap_or_default(),
                reason: ArrayString::from(&e.to_string()).unwrap_or_default(),
                current_value: ArrayString::new(),
                valid_range: None})?
            .headers(headers.iter().map(|(k, v)| (*k, v.as_str())))
            .timeout(Duration::from_secs(10));
        
        let response = self.http_client.send(http_request).await
            .map_err(CohereError::Http)?;
        
        match response.status().as_u16() {
            200..=299 => Ok(()),
            401 => Err(CohereError::auth_error(
                "chat",
                "Invalid API key",
                AuthErrorCode::InvalidApiKey,
                false,
            )),
            403 => Err(CohereError::auth_error(
                "chat",
                "Insufficient permissions",
                AuthErrorCode::InsufficientPermissions,
                false,
            )),
            status => Err(CohereError::from(status))}
    }
    
    /// Get client statistics for monitoring
    #[inline]
    pub fn get_stats(&self) -> ClientStats {
        ClientStats {
            chat_requests: CHAT_METRICS.total_requests.get(),
            chat_success: CHAT_METRICS.successful_requests.get(),
            chat_failures: CHAT_METRICS.failed_requests.get(),
            embed_requests: EMBED_METRICS.total_requests.get(),
            embed_success: EMBED_METRICS.successful_requests.get(),
            embed_failures: EMBED_METRICS.failed_requests.get(),
            rerank_requests: RERANK_METRICS.total_requests.get(),
            rerank_success: RERANK_METRICS.successful_requests.get(),
            rerank_failures: RERANK_METRICS.failed_requests.get(),
            active_requests: CHAT_METRICS.active_requests.get() 
                + EMBED_METRICS.active_requests.get() 
                + RERANK_METRICS.active_requests.get()}
    }
    
    /// Create embedding client for text embedding operations
    #[inline]
    pub fn embedding(&self) -> CohereEmbedding {
        CohereEmbedding::new(
            self.http_client,
            self.api_key.load(),
            self.endpoint_router.embed_url,
            &EMBED_METRICS,
        )
    }
    
    /// Create reranker client for document reranking operations
    #[inline]
    pub fn reranker(&self) -> CohereReranker {
        CohereReranker::new(
            self.http_client,
            self.api_key.load(),
            self.endpoint_router.rerank_url,
            &RERANK_METRICS,
        )
    }
    
    /// Convenience method for embedding texts
    pub async fn embed_texts(
        &self,
        texts: &[&str],
        model: &str,
    ) -> Result<EmbeddingResponse> {
        // Validate model is embedding type
        if !models::is_embedding_model(model) {
            return Err(CohereError::model_not_supported(
                model,
                CohereOperation::Embedding,
                models::EMBEDDING_MODELS,
                "embed",
            ));
        }
        
        let embedding_client = self.embedding();
        let request = EmbeddingRequest::new(texts.to_vec(), model.to_string());
        embedding_client.embed(request).await
    }
    
    /// Convenience method for reranking documents
    pub async fn rerank_documents(
        &self,
        query: &str,
        documents: &[&str],
        model: &str,
    ) -> Result<RerankResponse> {
        // Validate model is reranking type
        if !models::is_reranking_model(model) {
            return Err(CohereError::model_not_supported(
                model,
                CohereOperation::Reranking,
                models::RERANKING_MODELS,
                "rerank",
            ));
        }
        
        let reranker_client = self.reranker();
        let request = RerankRequest::new(query.to_string(), documents.to_vec(), model.to_string());
        reranker_client.rerank(request).await
    }
}

/// Implementation of CompletionClient trait for auto-generation integration
impl CompletionClient for CohereClient {
    type Model = Result<CohereCompletionBuilder>;
    
    #[inline]
    fn completion_model(&self, model: &str) -> Self::Model {
        // Validate model is chat completion type
        if !models::is_chat_model(model) {
            return Err(CohereError::model_not_supported(
                model,
                CohereOperation::Chat,
                models::CHAT_MODELS,
                "chat",
            ));
        }
        
        // Use static string or leak for zero allocation
        let static_model = match model {
            models::COMMAND_A_03_2025 => models::COMMAND_A_03_2025,
            models::COMMAND_R7B_12_2024 => models::COMMAND_R7B_12_2024,
            _ => Box::leak(model.to_string().into_boxed_str())};
        
        CohereCompletionBuilder::new(
            self.http_client,
            self.api_key.load(),
            static_model,
            self.endpoint_router.chat_url,
            &CHAT_METRICS,
        )
    }
}

/// Implementation of ProviderClient trait for auto-generation integration  
impl ProviderClient for CohereClient {
    #[inline]
    fn provider_name(&self) -> &'static str {
        "cohere"
    }
    
    #[inline]
    fn test_connection(&self) -> AsyncTask<Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let client = self.clone();
        AsyncTask::spawn(async move {
            client.validate_api_key().await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
    }
}

impl EndpointRouter {
    #[inline]
    const fn new() -> Self {
        Self {
            chat_url: endpoints::CHAT_URL,
            embed_url: endpoints::EMBED_URL,
            rerank_url: endpoints::RERANK_URL}
    }
}

impl CohereMetrics {
    #[inline]
    fn new(endpoint_name: &'static str) -> Self {
        Self {
            endpoint_name,
            total_requests: RelaxedCounter::new(0),
            successful_requests: RelaxedCounter::new(0),
            failed_requests: RelaxedCounter::new(0),
            total_latency_ms: RelaxedCounter::new(0),
            active_requests: RelaxedCounter::new(0),
            peak_active_requests: RelaxedCounter::new(0)}
    }
    
    #[inline]
    pub fn record_request_start(&self) {
        self.total_requests.inc();
        let active = self.active_requests.inc();
        
        // Update peak if necessary (lock-free)
        let mut current_peak = self.peak_active_requests.get();
        while active > current_peak {
            match self.peak_active_requests.compare_exchange_weak(
                current_peak,
                active,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_peak = actual}
        }
    }
    
    #[inline]
    pub fn record_request_success(&self, latency_ms: u64) {
        self.successful_requests.inc();
        self.active_requests.dec();
        self.total_latency_ms.add(latency_ms);
    }
    
    #[inline]
    pub fn record_request_failure(&self) {
        self.failed_requests.inc();
        self.active_requests.dec();
    }
    
    #[inline]
    pub fn average_latency_ms(&self) -> u64 {
        let total_requests = self.total_requests.get();
        if total_requests == 0 {
            0
        } else {
            self.total_latency_ms.get() / total_requests
        }
    }
    
    #[inline]
    pub fn success_rate(&self) -> f64 {
        let total = self.total_requests.get();
        if total == 0 {
            0.0
        } else {
            self.successful_requests.get() as f64 / total as f64
        }
    }
}

impl CohereCircuitBreakers {
    fn new() -> Self {
        let config = CBConfig {
            failure_threshold: config::CIRCUIT_BREAKER_THRESHOLD,
            recovery_timeout: Duration::from_secs(60),
            expected_update_interval: Duration::from_secs(10)};
        
        Self {
            chat: CircuitBreaker::new(config.clone()),
            embed: CircuitBreaker::new(config.clone()),
            rerank: CircuitBreaker::new(config)}
    }
}

/// Client performance statistics
#[derive(Debug, Clone)]
pub struct ClientStats {
    pub chat_requests: usize,
    pub chat_success: usize,
    pub chat_failures: usize,
    pub embed_requests: usize,
    pub embed_success: usize,
    pub embed_failures: usize,
    pub rerank_requests: usize,
    pub rerank_success: usize,
    pub rerank_failures: usize,
    pub active_requests: usize}

impl ClientStats {
    #[inline]
    pub fn total_requests(&self) -> usize {
        self.chat_requests + self.embed_requests + self.rerank_requests
    }
    
    #[inline]
    pub fn total_success(&self) -> usize {
        self.chat_success + self.embed_success + self.rerank_success
    }
    
    #[inline]
    pub fn total_failures(&self) -> usize {
        self.chat_failures + self.embed_failures + self.rerank_failures
    }
    
    #[inline]
    pub fn overall_success_rate(&self) -> f64 {
        let total = self.total_requests();
        if total == 0 {
            0.0
        } else {
            self.total_success() as f64 / total as f64
        }
    }
}

/// Request timing utility for performance monitoring
pub struct RequestTimer {
    start_time: SystemTime,
    metrics: &'static CohereMetrics}

impl RequestTimer {
    #[inline]
    pub fn start(metrics: &'static CohereMetrics) -> Self {
        metrics.record_request_start();
        Self {
            start_time: SystemTime::now(),
            metrics}
    }
    
    #[inline]
    pub fn finish_success(self) {
        let latency_ms = self.start_time
            .elapsed()
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        self.metrics.record_request_success(latency_ms);
    }
    
    #[inline]
    pub fn finish_failure(self) {
        self.metrics.record_request_failure();
    }
}