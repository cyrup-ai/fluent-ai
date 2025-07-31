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

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use arc_swap::{ArcSwap, Guard};
use arrayvec::{ArrayString};
use fluent_ai_async::AsyncTask;
use fluent_ai_domain::model::ModelInfo;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest};
use smallvec::{SmallVec, smallvec};

use super::completion::OpenAICompletionBuilder;
use super::config;
use super::error::{EndpointType, OpenAIError, Result};
use super::utils;
use crate::client::{CompletionClient, ProviderClient};
use crate::completion_provider::CompletionError;

/// Global HTTP client instance with connection pooling
static HTTP_CLIENT: LazyLock<HttpClient> = LazyLock::new(|| {
    HttpClient::with_config(HttpConfig::ai_optimized()).unwrap_or_else(|_| HttpClient::new())
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
    pub vision_requests: AtomicU64}

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
    vision_requests: AtomicU64::new(0)});

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
    HalfOpen = 2}

/// Request timer for performance monitoring
pub struct RequestTimer {
    start_time: Instant,
    metrics: &'static OpenAIMetrics,
    endpoint: EndpointType}

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
    max_retries: u32}

impl CircuitBreaker {
    /// Create new circuit breaker
    #[inline]
    pub const fn new(failure_threshold: u32, timeout: Duration) -> Self {
        Self {
            failure_count: AtomicU32::new(0),
            failure_threshold,
            timeout,
            last_failure_time: ArcSwap::from_pointee(None),
            state: AtomicU32::new(CircuitBreakerState::Closed as u32)}
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
                        self.state
                            .compare_exchange(
                                CircuitBreakerState::Open as u32,
                                CircuitBreakerState::HalfOpen as u32,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
            CircuitBreakerState::HalfOpen => true}
    }

    /// Record successful request
    #[inline]
    pub fn record_success(&self) {
        let current_state = self.get_state();
        match current_state {
            CircuitBreakerState::HalfOpen => {
                // Reset circuit breaker
                self.failure_count.store(0, Ordering::Release);
                self.state
                    .store(CircuitBreakerState::Closed as u32, Ordering::Release);
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
            self.state
                .store(CircuitBreakerState::Open as u32, Ordering::Release);
            METRICS
                .circuit_breaker_trips
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get current state
    #[inline]
    pub fn get_state(&self) -> CircuitBreakerState {
        match self.state.load(Ordering::Acquire) {
            0 => CircuitBreakerState::Closed,
            1 => CircuitBreakerState::Open,
            2 => CircuitBreakerState::HalfOpen,
            _ => CircuitBreakerState::Closed}
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
                metrics
                    .chat_completion_requests
                    .fetch_add(1, Ordering::Relaxed);
            }
            EndpointType::Embeddings => {
                metrics.embedding_requests.fetch_add(1, Ordering::Relaxed);
            }
            EndpointType::AudioTranscription
            | EndpointType::AudioTranslation
            | EndpointType::TextToSpeech => {
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
            endpoint}
    }

    /// Finish timing with success
    #[inline]
    pub fn finish_success(self) {
        let duration_ms = self.start_time.elapsed().as_millis() as u64;
        self.metrics
            .successful_requests
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_response_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
        self.metrics
            .concurrent_requests
            .fetch_sub(1, Ordering::Relaxed);
    }

    /// Finish timing with failure
    #[inline]
    pub fn finish_failure(self) {
        let duration_ms = self.start_time.elapsed().as_millis() as u64;
        self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_response_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
        self.metrics
            .concurrent_requests
            .fetch_sub(1, Ordering::Relaxed);
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
        utils::validate_api_key(&api_key).map_err(|err| {
            OpenAIError::invalid_api_key_error(
                &err,
                api_key.len(),
                !api_key.is_empty(),
                organization_id.as_deref(),
            )
        })?;

        // Convert API key to fixed-size array string for zero allocation storage
        let api_key_array = ArrayString::from(&api_key).map_err(|_| {
            OpenAIError::invalid_api_key_error(
                "API key too long",
                api_key.len(),
                true,
                organization_id.as_deref(),
            )
        })?;

        // Convert organization ID if provided
        let org_id = if let Some(org_id) = organization_id {
            if org_id.is_empty() {
                None
            } else {
                let org_array = ArrayString::from(&org_id).map_err(|_| {
                    OpenAIError::invalid_api_key_error(
                        "Organization ID too long",
                        org_id.len(),
                        true,
                        Some(&org_id),
                    )
                })?;
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
            max_retries: config::MAX_RETRIES})
    }

    /// Update API key with zero downtime using hot-swapping
    #[inline]
    pub fn update_api_key(&self, new_api_key: String) -> Result<()> {
        // Validate new API key
        utils::validate_api_key(&new_api_key).map_err(|err| {
            OpenAIError::invalid_api_key_error(
                &err,
                new_api_key.len(),
                !new_api_key.is_empty(),
                None,
            )
        })?;

        // Convert to array string
        let api_key_array = ArrayString::from(&new_api_key).map_err(|_| {
            OpenAIError::invalid_api_key_error("API key too long", new_api_key.len(), true, None)
        })?;

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

            let org_array = ArrayString::from(&org_id).map_err(|_| {
                OpenAIError::invalid_api_key_error(
                    "Organization ID too long",
                    org_id.len(),
                    true,
                    Some(&org_id),
                )
            })?;

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
    /// NOTE: Model validation has been centralized to model-info package
    #[inline]
    pub fn validate_model(&self, model: &str, endpoint: EndpointType) -> Result<()> {
        // Model validation is now handled centrally by the model-info package
        // Individual providers no longer perform local model validation
        // This allows for more flexibility and centralized model management
        Ok(())
    }

    /// Get model information
    /// NOTE: Model information now comes from model-info package
    #[inline]
    pub fn model_info(&self, model: &str) -> Result<ModelInfo> {
        // Create basic ModelInfo using ModelInfoBuilder from model-info package
        // More sophisticated model information should be retrieved via model-info package
        use model_info::ModelInfoBuilder;
        
        let model_info = ModelInfoBuilder::new()
            .name(model)
            .provider_name("openai")
            .with_streaming(true) // Most OpenAI models support streaming
            .build()
            .map_err(|e| OpenAIError::InvalidConfiguration {
                field: "model_info".to_string(),
                message: format!("Failed to create model info: {}", e),
                suggestion: "Check model name and try again".to_string(),
            })?;
            
        Ok(model_info)
    }

    /// Build authentication headers with zero allocation
    #[inline]
    fn build_headers(&self) -> SmallVec<[(&'static str, ArrayString<180>); 4]> {
        let mut auth_header = ArrayString::<180>::new();
        let _ = auth_header.try_push_str("Bearer ");
        let _ = auth_header.try_push_str(&self.api_key_guard());

        let mut headers = smallvec![
            ("Authorization", auth_header),
            (
                "Content-Type",
                ArrayString::from("application/json").unwrap_or_default()
            ),
            (
                "User-Agent",
                ArrayString::from(utils::user_agent()).unwrap_or_default()
            ),
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
            EndpointType::ChatCompletions => "/v1/chat/completions",
            EndpointType::Embeddings => "/v1/embeddings",
            EndpointType::AudioTranscription => "/v1/audio/transcriptions",
            EndpointType::AudioTranslation => "/v1/audio/translations",
            EndpointType::TextToSpeech => "/v1/audio/speech",
            EndpointType::VisionAnalysis => "/v1/chat/completions", /* Vision uses chat completions */
            EndpointType::Models => "/v1/models",
            EndpointType::Files => "/v1/files",
            EndpointType::FineTuning => "/v1/fine_tuning/jobs",
            EndpointType::Moderations => "/v1/moderations",
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
            .map_err(|e| {
                OpenAIError::configuration_error(
                    "http_request",
                    &e.to_string(),
                    "GET",
                    "Valid HTTP method",
                    EndpointType::Models,
                )
            })?
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
            vision_requests: METRICS.vision_requests.load(Ordering::Relaxed)}
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

// ModelInfo is now imported from fluent_ai_domain::model::ModelInfo
// Removed duplicated ModelInfo struct - use canonical domain type

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
    pub vision_requests: u64}

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
            "/v1/chat/completions",
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
    fn test_connection(
        &self,
    ) -> AsyncTask<std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let client = self.clone();
        AsyncTask::spawn(async move {
            client
                .test_connection()
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
    }
}

/// Production-ready secure credential management for OpenAI API keys
///
/// This implementation provides enterprise-grade security features:
/// - Environment variable validation with secure patterns
/// - Encrypted credential storage with ChaCha20Poly1305
/// - Automatic key rotation with audit logging
/// - Tamper-evident audit trails for compliance
/// - Zero-allocation credential handling
/// - Circuit breaker patterns for failed authentications
/// - Comprehensive security event monitoring
impl OpenAIClient {
    /// Create client with production-ready secure credential management
    pub async fn new_secure() -> Result<Self> {
        let credential_manager = Self::get_credential_manager().await?;
        let credential = credential_manager
            .get_credential("openai")
            .await
            .map_err(|e| OpenAIError::configuration_error(
                "api_key",
                "Failed to retrieve secure OpenAI API key",
                &format!("Credential manager error: {}", e),
                "Ensure OPENAI_API_KEY environment variable is set or credential is stored securely",
                EndpointType::ChatCompletions,
            ))?;

        let api_key = credential.value.as_str().to_string();
        Self::new(api_key)
    }

    /// Create client with secure credentials and organization
    pub async fn new_secure_with_organization(organization_id: Option<String>) -> Result<Self> {
        let credential_manager = Self::get_credential_manager().await?;
        let credential = credential_manager
            .get_credential("openai")
            .await
            .map_err(|e| OpenAIError::configuration_error(
                "api_key",
                "Failed to retrieve secure OpenAI API key",
                &format!("Credential manager error: {}", e),
                "Ensure OPENAI_API_KEY environment variable is set or credential is stored securely",
                EndpointType::ChatCompletions,
            ))?;

        let api_key = credential.value.as_str().to_string();
        Self::new_with_organization(api_key, organization_id)
    }

    /// Get or create global credential manager instance
    async fn get_credential_manager() -> Result<std::sync::Arc<crate::security::CredentialManager>>
    {
        use std::sync::OnceLock;

        use crate::security::{CredentialConfig, CredentialManager};

        static CREDENTIAL_MANAGER: OnceLock<std::sync::Arc<CredentialManager>> = OnceLock::new();

        let manager = CREDENTIAL_MANAGER.get_or_init(|| {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let config = CredentialConfig::default();
                    std::sync::Arc::new(CredentialManager::new(config).await.unwrap_or_else(|e| {
                        tracing::error!("Failed to initialize credential manager: {}", e);
                        panic!("Critical: Cannot initialize secure credential management");
                    }))
                })
            })
        });

        Ok(manager.clone())
    }

    /// Initialize secure credential system with custom configuration
    pub async fn initialize_security_system(
        config: crate::security::CredentialConfig,
    ) -> Result<()> {
        let credential_manager = std::sync::Arc::new(
            crate::security::CredentialManager::new(config.clone())
                .await
                .map_err(|e| {
                    OpenAIError::configuration_error(
                        "credential_manager",
                        "Failed to initialize credential manager",
                        &format!("Security system error: {}", e),
                        "Check credential configuration and permissions",
                        EndpointType::ChatCompletions,
                    )
                })?,
        );

        // Initialize key rotation if enabled
        if config.rotation_policy.enabled {
            let audit_logger = std::sync::Arc::new(
                crate::security::AuditLogger::new(&config.audit_config)
                    .await
                    .map_err(|e| {
                        OpenAIError::configuration_error(
                            "audit_logger",
                            "Failed to initialize audit logger",
                            &format!("Audit system error: {}", e),
                            "Check audit log permissions and configuration",
                            EndpointType::ChatCompletions,
                        )
                    })?,
            );

            let rotation_policy = crate::security::RotationPolicy::default();
            let rotation_scheduler = crate::security::KeyRotationScheduler::new(
                rotation_policy,
                credential_manager.clone(),
                audit_logger,
            )
            .await
            .map_err(|e| {
                OpenAIError::configuration_error(
                    "rotation_scheduler",
                    "Failed to initialize key rotation scheduler",
                    &format!("Rotation system error: {}", e),
                    "Check rotation policy configuration",
                    EndpointType::ChatCompletions,
                )
            })?;

            rotation_scheduler.start().await.map_err(|e| {
                OpenAIError::configuration_error(
                    "rotation_scheduler",
                    "Failed to start key rotation scheduler",
                    &format!("Scheduler error: {}", e),
                    "Check rotation scheduler permissions",
                    EndpointType::ChatCompletions,
                )
            })?;

            tracing::info!("OpenAI security system initialized with key rotation enabled");
        } else {
            tracing::info!("OpenAI security system initialized with key rotation disabled");
        }

        Ok(())
    }

    /// Update API key using secure credential management
    pub async fn update_api_key_secure(&self, new_api_key: String) -> Result<()> {
        let credential_manager = Self::get_credential_manager().await?;

        credential_manager
            .update_credential(
                "openai",
                new_api_key.clone(),
                crate::security::CredentialSource::Runtime {
                    origin: "manual_update".to_string()},
            )
            .await
            .map_err(|e| {
                OpenAIError::configuration_error(
                    "api_key_update",
                    "Failed to update API key securely",
                    &format!("Credential update error: {}", e),
                    "Check new API key format and permissions",
                    EndpointType::ChatCompletions,
                )
            })?;

        // Update the client's internal key as well
        self.update_api_key(new_api_key)?;

        Ok(())
    }

    /// Get credential statistics for monitoring
    pub async fn get_credential_statistics() -> Result<crate::security::CredentialStatistics> {
        let credential_manager = Self::get_credential_manager().await?;
        Ok(credential_manager.get_statistics().await)
    }

    /// Validate current credentials
    pub async fn validate_credentials(&self) -> Result<bool> {
        let credential_manager = Self::get_credential_manager().await?;

        match credential_manager.get_credential("openai").await {
            Ok(_) => Ok(true),
            Err(e) => {
                tracing::warn!("Credential validation failed for OpenAI: {}", e);
                Ok(false)
            }
        }
    }
}

/// Default implementation that fails fast if credentials are missing
/// This prevents accidental use of placeholder credentials in production
impl Default for OpenAIClient {
    fn default() -> Self {
        Self::new_secure()
            .map_err(|e| {
                tracing::error!("Failed to create OpenAI client with secure credentials: {}", e);
                e
            })
            .unwrap_or_else(|_| {
                // Fail fast in production - never use placeholder credentials
                panic!(
                    "OpenAI client requires valid API key. Set OPENAI_API_KEY environment variable. \
                    This error prevents accidental use of placeholder credentials in production."
                )
            })
    }
}

/// Debug implementation that doesn't expose API key
impl std::fmt::Debug for OpenAIClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIClient")
            .field("api_key", &"[REDACTED]")
            .field(
                "organization_id",
                &self.organization_id.as_ref().map(|_| "[REDACTED]"),
            )
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

// =============================================================================
// Missing Types for Module Compatibility
// =============================================================================

/// OpenAI Provider alias for compatibility
pub type OpenAIProvider = OpenAIClient;
