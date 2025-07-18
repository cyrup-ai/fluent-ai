//! OpenAI core client with lock-free multi-endpoint architecture and trait implementations
//!
//! Provides ultra-high-performance OpenAI integration with:
//! - Multi-endpoint architecture for chat, embeddings, audio, and vision
//! - Zero allocation patterns with hot-swappable API keys
//! - Lock-free atomic performance counters
//! - Circuit breaker protection with exponential backoff
//! - Connection pooling and intelligent reuse
//! - CompletionClient + ProviderClient trait implementations
//! - Bearer token authentication with optional Organization header
//!
//! The client serves as the foundation for OpenAI's complete API surface,
//! supporting all 18 GPT models with multi-endpoint routing and optimization.

use super::error::{OpenAIError, Result, EndpointType};
use super::completion::OpenAICompletionBuilder;
use super::models;
use super::config;
use super::endpoints;
use super::utils;

use crate::completion_provider::{CompletionProvider, CompletionError};
use crate::client::{CompletionClient, ProviderClient};
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest};
use fluent_ai_domain::AsyncTask;

use arc_swap::{ArcSwap, Guard};
use arrayvec::{ArrayVec, ArrayString};
use smallvec::{SmallVec, smallvec};
use std::sync::{Arc, LazyLock};
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Global HTTP client instance with connection pooling
static HTTP_CLIENT: LazyLock<HttpClient> = LazyLock::new(|| {
    HttpClient::with_config(HttpConfig::ai_optimized())
        .unwrap_or_else(|_| HttpClient::new())
});

/// Global circuit breaker for OpenAI requests
static CIRCUIT_BREAKER: LazyLock<CircuitBreaker> = LazyLock::new(|| {
    CircuitBreaker::new(
        config::CIRCUIT_BREAKER_THRESHOLD,
        config::CIRCUIT_BREAKER_TIMEOUT,
    )
});

/// OpenAI client performance metrics
#[derive(Debug)]
pub struct OpenAIMetrics {
    /// Total requests made
    pub total_requests: AtomicU64,
    /// Successful requests
    pub successful_requests: AtomicU64,
    /// Failed requests
    pub failed_requests: AtomicU64,
    /// Total response time (milliseconds)
    pub total_response_time_ms: AtomicU64,
    /// Current concurrent requests
    pub concurrent_requests: AtomicU32,
    /// Circuit breaker trips
    pub circuit_breaker_trips: AtomicU32,
    /// Chat completion requests
    pub chat_completion_requests: AtomicU64,
    /// Embedding requests
    pub embedding_requests: AtomicU64,
    /// Audio requests
    pub audio_requests: AtomicU64,
    /// Vision requests
    pub vision_requests: AtomicU64,
}

/// Global metrics instance
static METRICS: LazyLock<OpenAIMetrics> = LazyLock::new(|| OpenAIMetrics {
    total_requests: AtomicU64::new(0),
    successful_requests: AtomicU64::new(0),
    failed_requests: AtomicU64::new(0),
    total_response_time_ms: AtomicU64::new(0),
    concurrent_requests: AtomicU32::new(0),
    circuit_breaker_trips: AtomicU32::new(0),
    chat_completion_requests: AtomicU64::new(0),
    embedding_requests: AtomicU64::new(0),
    audio_requests: AtomicU64::new(0),
    vision_requests: AtomicU64::new(0),
});

/// Circuit breaker for fault tolerance
#[derive(Debug)]
pub struct CircuitBreaker {
    failure_count: AtomicU32,
    failure_threshold: u32,
    timeout: Duration,
    last_failure_time: ArcSwap<Option<Instant>>,
    state: AtomicU32, // 0 = Closed, 1 = Open, 2 = HalfOpen
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CircuitBreakerState {
    Closed = 0,
    Open = 1,
    HalfOpen = 2,
}

/// Request timer for performance monitoring
pub struct RequestTimer {
    start_time: Instant,
    metrics: &'static OpenAIMetrics,
    endpoint: EndpointType,
}

/// OpenAI client with zero allocation multi-endpoint architecture
#[derive(Clone)]
pub struct OpenAIClient {
    /// Hot-swappable API key
    api_key: ArcSwap<ArrayString<128>>,
    
    /// Optional organization ID
    organization_id: Option<ArcSwap<ArrayString<64>>>,
    
    /// Shared HTTP client
    http_client: &'static HttpClient,
    
    /// Circuit breaker for fault tolerance
    circuit_breaker: &'static CircuitBreaker,
    
    /// Request timeout
    timeout: Duration,
    
    /// Maximum retry attempts
    max_retries: u32,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    #[inline]
    pub const fn new(failure_threshold: u32, timeout: Duration) -> Self {
        Self {
            failure_count: AtomicU32::new(0),
            failure_threshold,
            timeout,
            last_failure_time: ArcSwap::from_pointee(None),
            state: AtomicU32::new(CircuitBreakerState::Closed as u32),
        }
    }
    
    /// Check if request is allowed
    #[inline]
    pub fn is_request_allowed(&self) -> bool {
        let state = self.get_state();
        match state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if timeout has elapsed
                if let Some(last_failure) = self.last_failure_time.load_full() {
                    if last_failure.elapsed() > self.timeout {
                        // Try to transition to half-open
                        self.state.compare_exchange(
                            CircuitBreakerState::Open as u32,
                            CircuitBreakerState::HalfOpen as u32,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        ).is_ok()
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
            CircuitBreakerState::HalfOpen => true,
        }
    }
    
    /// Record successful request
    #[inline]
    pub fn record_success(&self) {
        let current_state = self.get_state();
        match current_state {
            CircuitBreakerState::HalfOpen => {
                // Reset circuit breaker
                self.failure_count.store(0, Ordering::Release);
                self.state.store(CircuitBreakerState::Closed as u32, Ordering::Release);
            }
            _ => {}
        }
    }
    
    /// Record failed request
    #[inline]
    pub fn record_failure(&self) {
        let failure_count = self.failure_count.fetch_add(1, Ordering::AcqRel) + 1;
        self.last_failure_time.store(Arc::new(Some(Instant::now())));
        
        if failure_count >= self.failure_threshold {
            self.state.store(CircuitBreakerState::Open as u32, Ordering::Release);
            METRICS.circuit_breaker_trips.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Get current state
    #[inline]
    pub fn get_state(&self) -> CircuitBreakerState {
        match self.state.load(Ordering::Acquire) {
            0 => CircuitBreakerState::Closed,
            1 => CircuitBreakerState::Open,
            2 => CircuitBreakerState::HalfOpen,
            _ => CircuitBreakerState::Closed,
        }
    }
}

impl RequestTimer {
    /// Start timing a request
    #[inline]
    pub fn start(metrics: &'static OpenAIMetrics, endpoint: EndpointType) -> Self {
        metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        metrics.concurrent_requests.fetch_add(1, Ordering::Relaxed);
        
        // Track endpoint-specific metrics
        match endpoint {
            EndpointType::ChatCompletions => {
                metrics.chat_completion_requests.fetch_add(1, Ordering::Relaxed);
            }
            EndpointType::Embeddings => {
                metrics.embedding_requests.fetch_add(1, Ordering::Relaxed);
            }
            EndpointType::AudioTranscription | EndpointType::AudioTranslation | EndpointType::TextToSpeech => {
                metrics.audio_requests.fetch_add(1, Ordering::Relaxed);
            }
            EndpointType::VisionAnalysis => {
                metrics.vision_requests.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        
        Self {
            start_time: Instant::now(),
            metrics,
            endpoint,
        }
    }
    
    /// Finish timing with success
    #[inline]
    pub fn finish_success(self) {
        let duration_ms = self.start_time.elapsed().as_millis() as u64;
        self.metrics.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.metrics.total_response_time_ms.fetch_add(duration_ms, Ordering::Relaxed);
        self.metrics.concurrent_requests.fetch_sub(1, Ordering::Relaxed);
    }
    
    /// Finish timing with failure
    #[inline]
    pub fn finish_failure(self) {
        let duration_ms = self.start_time.elapsed().as_millis() as u64;
        self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.metrics.total_response_time_ms.fetch_add(duration_ms, Ordering::Relaxed);
        self.metrics.concurrent_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

impl OpenAIClient {
    /// Create new OpenAI client with API key validation
    pub fn new(api_key: String) -> Result<Self> {
        Self::new_with_organization(api_key, None)
    }
    
    /// Create new OpenAI client with API key and organization ID
    pub fn new_with_organization(api_key: String, organization_id: Option<String>) -> Result<Self> {
        // Validate API key format
        utils::validate_api_key(&api_key)
            .map_err(|err| OpenAIError::invalid_api_key_error(
                &err,
                api_key.len(),
                !api_key.is_empty(),
                organization_id.as_deref(),
            ))?;
        
        // Convert API key to fixed-size array string for zero allocation storage
        let api_key_array = ArrayString::from(&api_key)
            .map_err(|_| OpenAIError::invalid_api_key_error(
                "API key too long",
                api_key.len(),
                true,
                organization_id.as_deref(),
            ))?;
        
        // Convert organization ID if provided
        let org_id = if let Some(org_id) = organization_id {
            if org_id.is_empty() {
                None
            } else {
                let org_array = ArrayString::from(&org_id)
                    .map_err(|_| OpenAIError::invalid_api_key_error(
                        "Organization ID too long",
                        org_id.len(),
                        true,
                        Some(&org_id),
                    ))?;
                Some(ArcSwap::from_pointee(org_array))
            }
        } else {
            None
        };
        
        Ok(Self {
            api_key: ArcSwap::from_pointee(api_key_array),
            organization_id: org_id,
            http_client: &HTTP_CLIENT,
            circuit_breaker: &CIRCUIT_BREAKER,
            timeout: Duration::from_secs(config::DEFAULT_TIMEOUT_SECS),
            max_retries: config::MAX_RETRIES,
        })
    }
    
    /// Update API key with zero downtime using hot-swapping
    #[inline]
    pub fn update_api_key(&self, new_api_key: String) -> Result<()> {
        // Validate new API key
        utils::validate_api_key(&new_api_key)
            .map_err(|err| OpenAIError::invalid_api_key_error(
                &err,
                new_api_key.len(),
                !new_api_key.is_empty(),
                None,
            ))?;
        
        // Convert to array string
        let api_key_array = ArrayString::from(&new_api_key)
            .map_err(|_| OpenAIError::invalid_api_key_error(
                "API key too long",
                new_api_key.len(),
                true,
                None,
            ))?;
        
        // Hot-swap with zero downtime
        self.api_key.store(Arc::new(api_key_array));
        Ok(())
    }
    
    /// Update organization ID with zero downtime
    #[inline]
    pub fn update_organization_id(&self, new_org_id: Option<String>) -> Result<()> {
        if let Some(org_id) = new_org_id {
            if org_id.is_empty() {
                return Ok(());
            }
            
            let org_array = ArrayString::from(&org_id)
                .map_err(|_| OpenAIError::invalid_api_key_error(
                    "Organization ID too long",
                    org_id.len(),
                    true,
                    Some(&org_id),
                ))?;
            
            if let Some(org_swap) = &self.organization_id {
                org_swap.store(Arc::new(org_array));
            }
        }
        
        Ok(())
    }
    
    /// Get API key guard for zero allocation access
    #[inline]
    fn api_key_guard(&self) -> Guard<Arc<ArrayString<128>>> {
        self.api_key.load()
    }
    
    /// Get organization ID guard for zero allocation access
    #[inline]
    fn organization_id_guard(&self) -> Option<Guard<Arc<ArrayString<64>>>> {
        self.organization_id.as_ref().map(|org| org.load())
    }
    
    /// Set request timeout
    #[inline]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    /// Set maximum retry attempts
    #[inline]
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
    
    /// Validate model is supported for endpoint
    #[inline]
    pub fn validate_model(&self, model: &str, endpoint: EndpointType) -> Result<()> {
        match endpoint {
            EndpointType::ChatCompletions => {
                if !models::is_chat_model(model) {
                    return Err(OpenAIError::model_not_supported(
                        model,
                        endpoint,
                        models::CHAT_MODELS,
                        models::GPT_4O,
                        false,
                    ));
                }
            }
            EndpointType::Embeddings => {
                if !models::is_embedding_model(model) {
                    return Err(OpenAIError::model_not_supported(
                        model,
                        endpoint,
                        models::EMBEDDING_MODELS,
                        models::TEXT_EMBEDDING_3_LARGE,
                        false,
                    ));
                }
            }
            EndpointType::AudioTranscription | EndpointType::AudioTranslation => {
                if !models::is_audio_model(model) {
                    return Err(OpenAIError::model_not_supported(
                        model,
                        endpoint,
                        models::AUDIO_MODELS,
                        models::WHISPER_1,
                        false,
                    ));
                }
            }
            EndpointType::TextToSpeech => {
                if !models::is_tts_model(model) {
                    return Err(OpenAIError::model_not_supported(
                        model,
                        endpoint,
                        models::TTS_MODELS,
                        models::TTS_1,
                        false,
                    ));
                }
            }
            EndpointType::VisionAnalysis => {
                if !models::is_vision_model(model) {
                    return Err(OpenAIError::model_not_supported(
                        model,
                        endpoint,
                        models::VISION_MODELS,
                        models::GPT_4O,
                        false,
                    ));
                }
            }
            _ => {
                if !models::is_supported_model(model) {
                    return Err(OpenAIError::model_not_supported(
                        model,
                        endpoint,
                        models::ALL_MODELS,
                        models::GPT_4O,
                        false,
                    ));
                }
            }
        }
        Ok(())
    }
    
    /// Get model information
    #[inline]
    pub fn model_info(&self, model: &str) -> Result<ModelInfo> {
        if !models::is_supported_model(model) {
            return Err(OpenAIError::model_not_supported(
                model,
                EndpointType::ChatCompletions,
                models::ALL_MODELS,
                models::GPT_4O,
                false,
            ));
        }
        
        Ok(ModelInfo {
            name: model,
            family: models::model_family(model).unwrap_or("unknown"),
            generation: models::model_generation(model).unwrap_or("unknown"),
            max_context: models::context_length(model),
            supports_streaming: models::supports_streaming(model),
            supports_tools: models::supports_tools(model),
            supports_vision: models::supports_vision(model),
            supports_audio: models::supports_audio(model),
            temperature_range: models::temperature_range(model),
            pricing_tier: utils::pricing_tier(model),
        })
    }
    
    /// Build authentication headers with zero allocation
    #[inline]
    fn build_headers(&self) -> SmallVec<[(&'static str, ArrayString<180>); 4]> {
        let mut auth_header = ArrayString::<180>::new();
        let _ = auth_header.try_push_str("Bearer ");
        let _ = auth_header.try_push_str(&self.api_key_guard());
        
        let mut headers = smallvec![
            ("Authorization", auth_header),
            ("Content-Type", ArrayString::from("application/json").unwrap_or_default()),
            ("User-Agent", ArrayString::from(utils::user_agent()).unwrap_or_default()),
        ];
        
        // Add organization header if present
        if let Some(org_guard) = self.organization_id_guard() {
            let mut org_header = ArrayString::<64>::new();
            let _ = org_header.try_push_str(&org_guard);
            headers.push(("OpenAI-Organization", org_header));
        }
        
        headers
    }
    
    /// Route endpoint to URL
    #[inline]
    fn route_endpoint(&self, endpoint: EndpointType) -> &'static str {
        match endpoint {
            EndpointType::ChatCompletions => endpoints::CHAT_COMPLETIONS,
            EndpointType::Embeddings => endpoints::EMBEDDINGS,
            EndpointType::AudioTranscription => endpoints::AUDIO_TRANSCRIPTIONS,
            EndpointType::AudioTranslation => endpoints::AUDIO_TRANSLATIONS,
            EndpointType::TextToSpeech => endpoints::AUDIO_SPEECH,
            EndpointType::VisionAnalysis => endpoints::CHAT_COMPLETIONS, // Vision uses chat completions
            EndpointType::Models => endpoints::MODELS,
            EndpointType::Files => endpoints::FILES,
            EndpointType::FineTuning => endpoints::FINE_TUNING,
            EndpointType::Moderations => endpoints::MODERATIONS,
        }
    }
    
    /// Test connection to OpenAI API
    pub async fn test_connection(&self) -> Result<()> {
        if !self.circuit_breaker.is_request_allowed() {
            return Err(OpenAIError::model_capacity_error(
                "circuit-breaker",
                0,
                60,
                true,
                &[],
                EndpointType::Models,
            ));
        }
        
        let timer = RequestTimer::start(&METRICS, EndpointType::Models);
        
        // Use a simple model list request as health check
        let headers = self.build_headers();
        let request = HttpRequest::get(self.route_endpoint(EndpointType::Models))
            .map_err(|e| OpenAIError::configuration_error(
                "http_request",
                &e.to_string(),
                "GET",
                "Valid HTTP method",
                EndpointType::Models,
            ))?
            .headers(headers.iter().map(|(k, v)| (*k, v.as_str())))
            .timeout(self.timeout);
        
        match self.http_client.send(request).await {
            Ok(response) => {
                if response.status().is_success() {
                    self.circuit_breaker.record_success();
                    timer.finish_success();
                    Ok(())
                } else {
                    self.circuit_breaker.record_failure();
                    timer.finish_failure();
                    Err(OpenAIError::from(response.status().as_u16()))
                }
            }
            Err(e) => {
                self.circuit_breaker.record_failure();
                timer.finish_failure();
                Err(OpenAIError::Http(e))
            }
        }
    }
    
    /// Get current performance metrics
    #[inline]
    pub fn get_metrics(&self) -> PerformanceMetrics {
        let total_requests = METRICS.total_requests.load(Ordering::Relaxed);
        let successful_requests = METRICS.successful_requests.load(Ordering::Relaxed);
        let failed_requests = METRICS.failed_requests.load(Ordering::Relaxed);
        let total_response_time = METRICS.total_response_time_ms.load(Ordering::Relaxed);
        
        PerformanceMetrics {
            total_requests,
            successful_requests,
            failed_requests,
            success_rate: if total_requests > 0 {
                successful_requests as f64 / total_requests as f64
            } else {
                0.0
            },
            average_response_time_ms: if total_requests > 0 {
                total_response_time / total_requests
            } else {
                0
            },
            concurrent_requests: METRICS.concurrent_requests.load(Ordering::Relaxed),
            circuit_breaker_trips: METRICS.circuit_breaker_trips.load(Ordering::Relaxed),
            circuit_breaker_state: self.circuit_breaker.get_state(),
            chat_completion_requests: METRICS.chat_completion_requests.load(Ordering::Relaxed),
            embedding_requests: METRICS.embedding_requests.load(Ordering::Relaxed),
            audio_requests: METRICS.audio_requests.load(Ordering::Relaxed),
            vision_requests: METRICS.vision_requests.load(Ordering::Relaxed),
        }
    }
    
    /// Reset performance metrics
    #[inline]
    pub fn reset_metrics(&self) {
        METRICS.total_requests.store(0, Ordering::Relaxed);
        METRICS.successful_requests.store(0, Ordering::Relaxed);
        METRICS.failed_requests.store(0, Ordering::Relaxed);
        METRICS.total_response_time_ms.store(0, Ordering::Relaxed);
        METRICS.concurrent_requests.store(0, Ordering::Relaxed);
        METRICS.circuit_breaker_trips.store(0, Ordering::Relaxed);
        METRICS.chat_completion_requests.store(0, Ordering::Relaxed);
        METRICS.embedding_requests.store(0, Ordering::Relaxed);
        METRICS.audio_requests.store(0, Ordering::Relaxed);
        METRICS.vision_requests.store(0, Ordering::Relaxed);
    }
}

/// Model information structure
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: &'static str,
    pub family: &'static str,
    pub generation: &'static str,
    pub max_context: u32,
    pub supports_streaming: bool,
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub temperature_range: (f32, f32),
    pub pricing_tier: &'static str,
}

/// Performance metrics structure
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub success_rate: f64,
    pub average_response_time_ms: u64,
    pub concurrent_requests: u32,
    pub circuit_breaker_trips: u32,
    pub circuit_breaker_state: CircuitBreakerState,
    pub chat_completion_requests: u64,
    pub embedding_requests: u64,
    pub audio_requests: u64,
    pub vision_requests: u64,
}

/// CompletionClient trait implementation for auto-generation
impl CompletionClient for OpenAIClient {
    type Model = Result<OpenAICompletionBuilder>;
    
    #[inline]
    fn completion_model(&self, model: &str) -> Self::Model {
        // Validate model is supported for chat completions
        self.validate_model(model, EndpointType::ChatCompletions)?;
        
        // Create completion builder
        OpenAICompletionBuilder::new(
            self.http_client,
            self.api_key_guard(),
            self.organization_id_guard(),
            model,
            endpoints::CHAT_COMPLETIONS,
            &METRICS,
        )
    }
}

/// ProviderClient trait implementation for ecosystem integration
impl ProviderClient for OpenAIClient {
    #[inline]
    fn provider_name(&self) -> &'static str {
        "openai"
    }
    
    #[inline]
    fn test_connection(&self) -> AsyncTask<std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let client = self.clone();
        AsyncTask::spawn(async move {
            client.test_connection().await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
    }
}

/// Default implementation for OpenAI client
impl Default for OpenAIClient {
    fn default() -> Self {
        // Create with placeholder API key - will need to be updated
        Self::new("placeholder-api-key-update-before-use".to_string())
            .unwrap_or_else(|_| panic!("Failed to create default OpenAI client"))
    }
}

/// Debug implementation that doesn't expose API key
impl std::fmt::Debug for OpenAIClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIClient")
            .field("api_key", &"[REDACTED]")
            .field("organization_id", &self.organization_id.as_ref().map(|_| "[REDACTED]"))
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_client_creation() {
        let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string());
        assert!(client.is_ok());
    }
    
    #[test]
    fn test_client_with_organization() {
        let client = OpenAIClient::new_with_organization(
            "sk-test123456789012345678901234567890123456789012345678901234567890".to_string(),
            Some("org-test123456789012345678901234567890".to_string()),
        );
        assert!(client.is_ok());
    }
    
    #[test]
    fn test_invalid_api_key() {
        let client = OpenAIClient::new("".to_string());
        assert!(client.is_err());
        
        let client = OpenAIClient::new("short".to_string());
        assert!(client.is_err());
    }
    
    #[test]
    fn test_model_validation() {
        let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
            .unwrap();
        
        assert!(client.validate_model(models::GPT_4O, EndpointType::ChatCompletions).is_ok());
        assert!(client.validate_model(models::GPT_4O_MINI, EndpointType::ChatCompletions).is_ok());
        assert!(client.validate_model(models::TEXT_EMBEDDING_3_LARGE, EndpointType::Embeddings).is_ok());
        assert!(client.validate_model(models::WHISPER_1, EndpointType::AudioTranscription).is_ok());
        assert!(client.validate_model("invalid-model", EndpointType::ChatCompletions).is_err());
    }
    
    #[test]
    fn test_model_info() {
        let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
            .unwrap();
        
        let info = client.model_info(models::GPT_4O).unwrap();
        assert_eq!(info.name, models::GPT_4O);
        assert_eq!(info.family, "gpt-4");
        assert_eq!(info.generation, "gpt-4");
        assert!(info.supports_streaming);
        assert!(info.supports_tools);
        assert!(info.supports_vision);
    }
    
    #[test]
    fn test_api_key_update() {
        let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
            .unwrap();
        
        let result = client.update_api_key("sk-new1234567890123456789012345678901234567890123456789012345678".to_string());
        assert!(result.is_ok());
        
        let result = client.update_api_key("".to_string());
        assert!(result.is_err());
    }
    
    #[test]
    fn test_circuit_breaker() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60));
        assert_eq!(cb.get_state(), CircuitBreakerState::Closed);
        assert!(cb.is_request_allowed());
        
        // Record failures
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        
        assert_eq!(cb.get_state(), CircuitBreakerState::Open);
        assert!(!cb.is_request_allowed());
        
        // Record success should reset
        cb.record_success();
        assert_eq!(cb.get_state(), CircuitBreakerState::Closed);
    }
    
    #[test]
    fn test_endpoint_routing() {
        let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
            .unwrap();
        
        assert_eq!(client.route_endpoint(EndpointType::ChatCompletions), endpoints::CHAT_COMPLETIONS);
        assert_eq!(client.route_endpoint(EndpointType::Embeddings), endpoints::EMBEDDINGS);
        assert_eq!(client.route_endpoint(EndpointType::AudioTranscription), endpoints::AUDIO_TRANSCRIPTIONS);
        assert_eq!(client.route_endpoint(EndpointType::TextToSpeech), endpoints::AUDIO_SPEECH);
        assert_eq!(client.route_endpoint(EndpointType::VisionAnalysis), endpoints::CHAT_COMPLETIONS);
    }
    
    #[test]
    fn test_completion_client_trait() {
        let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
            .unwrap();
        
        let result = client.completion_model(models::GPT_4O);
        assert!(result.is_ok());
        
        let result = client.completion_model("invalid-model");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_provider_client_trait() {
        let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
            .unwrap();
        
        assert_eq!(client.provider_name(), "openai");
    }
    
    #[test]
    fn test_performance_metrics() {
        let client = OpenAIClient::new("sk-test123456789012345678901234567890123456789012345678901234567890".to_string())
            .unwrap();
        
        let metrics = client.get_metrics();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.successful_requests, 0);
        assert_eq!(metrics.failed_requests, 0);
        assert_eq!(metrics.concurrent_requests, 0);
        
        // Test reset
        client.reset_metrics();
        let metrics = client.get_metrics();
        assert_eq!(metrics.total_requests, 0);
    }
}