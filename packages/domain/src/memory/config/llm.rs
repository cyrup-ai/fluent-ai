use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use crossbeam_utils::CachePadded;
use serde::{Deserialize, Serialize};

use super::super::primitives::types::{MemoryError, MemoryResult};

/// LLM configuration with streaming HTTP configuration
///
/// Features:
/// - Provider selection with capability matching
/// - Model configuration with parameter validation
/// - API endpoint setup with fluent_ai_http3 integration
/// - Streaming configuration with zero-allocation buffer management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// LLM provider
    pub provider: LLMProvider,
    /// Model configuration
    pub model_config: ModelConfig,
    /// API endpoint configuration
    pub endpoint_config: EndpointConfig,
    /// HTTP3 streaming configuration
    pub streaming_config: StreamingConfig,
    /// Rate limiting configuration
    pub rate_limit_config: RateLimitConfig,
    /// Retry configuration for reliability
    pub retry_config: RetryConfig,
    /// Response caching configuration
    pub cache_config: CacheConfig,
    /// Token usage tracking
    pub token_tracking: TokenTrackingConfig,
}

/// LLM providers with capability definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum LLMProvider {
    /// OpenAI GPT models
    OpenAI = 0,
    /// Anthropic Claude models
    Anthropic = 1,
    /// Google Gemini models
    Google = 2,
    /// Meta Llama models
    Meta = 3,
    /// Mistral AI models
    Mistral = 4,
    /// Cohere models
    Cohere = 5,
    /// Local models via Ollama
    Ollama = 6,
    /// Custom provider
    Custom = 7,
}

impl LLMProvider {
    /// Get default API base URL for provider
    #[inline]
    pub const fn default_api_base(&self) -> &'static str {
        match self {
            Self::OpenAI => "https://api.openai.com/v1",
            Self::Anthropic => "https://api.anthropic.com/v1",
            Self::Google => "https://generativelanguage.googleapis.com/v1",
            Self::Meta => "https://api.llama-api.com/v1",
            Self::Mistral => "https://api.mistral.ai/v1",
            Self::Cohere => "https://api.cohere.ai/v1",
            Self::Ollama => "http://localhost:11434/api",
            Self::Custom => "",
        }
    }

    /// Check if provider supports streaming
    #[inline]
    pub const fn supports_streaming(&self) -> bool {
        match self {
            Self::OpenAI
            | Self::Anthropic
            | Self::Google
            | Self::Mistral
            | Self::Cohere
            | Self::Ollama => true,
            Self::Meta | Self::Custom => false, // Depends on implementation
        }
    }

    /// Check if provider supports function calling
    #[inline]
    pub const fn supports_function_calling(&self) -> bool {
        match self {
            Self::OpenAI | Self::Google | Self::Mistral => true,
            Self::Anthropic | Self::Meta | Self::Cohere | Self::Ollama | Self::Custom => false,
        }
    }

    /// Get maximum context length for provider
    #[inline]
    pub const fn max_context_length(&self) -> usize {
        match self {
            Self::OpenAI => 128000,    // GPT-4 turbo
            Self::Anthropic => 200000, // Claude 3
            Self::Google => 2000000,   // Gemini Pro
            Self::Meta => 32768,       // Llama 2
            Self::Mistral => 32768,    // Mistral Large
            Self::Cohere => 128000,    // Command R+
            Self::Ollama => 131072,    // Depends on model
            Self::Custom => 8192,      // Conservative default
        }
    }

    /// Get default model for provider
    #[inline]
    pub const fn default_model(&self) -> &'static str {
        match self {
            Self::OpenAI => "gpt-4-1106-preview",
            Self::Anthropic => "claude-3-haiku-20240307",
            Self::Google => "gemini-pro",
            Self::Meta => "llama-2-70b-chat",
            Self::Mistral => "mistral-large-latest",
            Self::Cohere => "command-r-plus",
            Self::Ollama => "llama2",
            Self::Custom => "custom-model",
        }
    }

    /// Get authentication header name
    #[inline]
    pub const fn auth_header(&self) -> &'static str {
        match self {
            Self::OpenAI | Self::Mistral | Self::Cohere => "Authorization",
            Self::Anthropic => "x-api-key",
            Self::Google => "x-goog-api-key",
            Self::Meta => "Authorization",
            Self::Ollama => "", // No auth required by default
            Self::Custom => "Authorization",
        }
    }

    /// Get authentication header value prefix
    #[inline]
    pub const fn auth_prefix(&self) -> &'static str {
        match self {
            Self::OpenAI | Self::Meta | Self::Custom => "Bearer ",
            Self::Anthropic | Self::Google | Self::Mistral | Self::Cohere => "",
            Self::Ollama => "",
        }
    }
}

impl std::fmt::Display for LLMProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAI => write!(f, "openai"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::Google => write!(f, "google"),
            Self::Meta => write!(f, "meta"),
            Self::Mistral => write!(f, "mistral"),
            Self::Cohere => write!(f, "cohere"),
            Self::Ollama => write!(f, "ollama"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

/// Model configuration with parameter validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name or identifier
    pub model_name: Arc<str>,
    /// Maximum tokens for completion
    pub max_tokens: Option<usize>,
    /// Temperature for randomness (0.0 to 2.0)
    pub temperature: Option<f32>,
    /// Top-p sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter  
    pub top_k: Option<usize>,
    /// Frequency penalty (-2.0 to 2.0)
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (-2.0 to 2.0)
    pub presence_penalty: Option<f32>,
    /// Stop sequences
    pub stop_sequences: Vec<Arc<str>>,
    /// Custom model parameters
    pub custom_parameters: HashMap<Arc<str>, serde_json::Value>,
}

impl ModelConfig {
    /// Create new model configuration with validation
    pub fn new(model_name: impl Into<Arc<str>>) -> MemoryResult<Self> {
        let name = model_name.into();
        if name.is_empty() {
            return Err(MemoryError::validation("Model name cannot be empty"));
        }

        Ok(Self {
            model_name: name,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: Vec::new(),
            custom_parameters: HashMap::new(),
        })
    }

    /// Set maximum tokens with validation
    pub fn with_max_tokens(mut self, max_tokens: usize) -> MemoryResult<Self> {
        if max_tokens == 0 {
            return Err(MemoryError::validation("Max tokens must be greater than 0"));
        }
        if max_tokens > 200000 {
            return Err(MemoryError::validation(
                "Max tokens exceeds reasonable limit",
            ));
        }
        self.max_tokens = Some(max_tokens);
        Ok(self)
    }

    /// Set temperature with validation
    pub fn with_temperature(mut self, temperature: f32) -> MemoryResult<Self> {
        if !(0.0..=2.0).contains(&temperature) {
            return Err(MemoryError::validation(
                "Temperature must be between 0.0 and 2.0",
            ));
        }
        self.temperature = Some(temperature);
        Ok(self)
    }

    /// Set top-p with validation
    pub fn with_top_p(mut self, top_p: f32) -> MemoryResult<Self> {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(MemoryError::validation("Top-p must be between 0.0 and 1.0"));
        }
        self.top_p = Some(top_p);
        Ok(self)
    }

    /// Set frequency penalty with validation
    pub fn with_frequency_penalty(mut self, penalty: f32) -> MemoryResult<Self> {
        if !(-2.0..=2.0).contains(&penalty) {
            return Err(MemoryError::validation(
                "Frequency penalty must be between -2.0 and 2.0",
            ));
        }
        self.frequency_penalty = Some(penalty);
        Ok(self)
    }

    /// Set presence penalty with validation
    pub fn with_presence_penalty(mut self, penalty: f32) -> MemoryResult<Self> {
        if !(-2.0..=2.0).contains(&penalty) {
            return Err(MemoryError::validation(
                "Presence penalty must be between -2.0 and 2.0",
            ));
        }
        self.presence_penalty = Some(penalty);
        Ok(self)
    }

    /// Add stop sequence
    #[inline]
    pub fn add_stop_sequence(mut self, sequence: impl Into<Arc<str>>) -> Self {
        self.stop_sequences.push(sequence.into());
        self
    }

    /// Add custom parameter
    #[inline]
    pub fn add_custom_parameter(
        mut self,
        key: impl Into<Arc<str>>,
        value: serde_json::Value,
    ) -> Self {
        self.custom_parameters.insert(key.into(), value);
        self
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::new("gpt-4-1106-preview").expect("Default model config should be valid")
    }
}

/// API endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    /// Base API URL
    pub api_base: Arc<str>,
    /// API key for authentication
    pub api_key: Option<Arc<str>>,
    /// Custom headers
    pub headers: HashMap<Arc<str>, Arc<str>>,
    /// Request timeout
    pub timeout: Duration,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Enable TLS verification
    pub verify_tls: bool,
    /// Custom user agent
    pub user_agent: Option<Arc<str>>,
}

impl EndpointConfig {
    /// Create new endpoint configuration
    #[inline]
    pub fn new(api_base: impl Into<Arc<str>>) -> Self {
        Self {
            api_base: api_base.into(),
            api_key: None,
            headers: HashMap::new(),
            timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(10),
            verify_tls: true,
            user_agent: Some(Arc::from("fluent-ai/1.0")),
        }
    }

    /// Set API key
    #[inline]
    pub fn with_api_key(mut self, api_key: impl Into<Arc<str>>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Add custom header
    #[inline]
    pub fn add_header(mut self, key: impl Into<Arc<str>>, value: impl Into<Arc<str>>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set timeout
    #[inline]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Get authorization header value for provider
    pub fn auth_header_value(&self, provider: LLMProvider) -> Option<String> {
        self.api_key
            .as_ref()
            .map(|key| format!("{}{}", provider.auth_prefix(), key))
    }
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self::new(LLMProvider::OpenAI.default_api_base())
    }
}

/// HTTP3 streaming configuration with zero-allocation buffer management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Enable streaming responses
    pub enable_streaming: bool,
    /// Buffer size for streaming data
    pub buffer_size: usize,
    /// Maximum chunk size for processing
    pub max_chunk_size: usize,
    /// Enable Server-Sent Events parsing
    pub enable_sse: bool,
    /// Enable JSON Lines parsing
    pub enable_json_lines: bool,
    /// Backpressure threshold
    pub backpressure_threshold: usize,
    /// Stream timeout
    pub stream_timeout: Duration,
    /// Enable compression for streams
    pub enable_compression: bool,
}

impl StreamingConfig {
    /// Create optimized streaming configuration
    #[inline]
    pub fn optimized() -> Self {
        Self {
            enable_streaming: true,
            buffer_size: 8192,    // 8KB buffer
            max_chunk_size: 4096, // 4KB chunks
            enable_sse: true,
            enable_json_lines: true,
            backpressure_threshold: 16384, // 16KB backpressure
            stream_timeout: Duration::from_secs(300), // 5 minute stream timeout
            enable_compression: true,
        }
    }

    /// Create minimal streaming configuration
    #[inline]
    pub fn minimal() -> Self {
        Self {
            enable_streaming: false,
            buffer_size: 1024,
            max_chunk_size: 512,
            enable_sse: false,
            enable_json_lines: false,
            backpressure_threshold: 2048,
            stream_timeout: Duration::from_secs(30),
            enable_compression: false,
        }
    }

    /// Check if configuration supports zero-allocation streaming
    #[inline]
    pub fn supports_zero_allocation(&self) -> bool {
        self.enable_streaming && (self.enable_sse || self.enable_json_lines)
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self::optimized()
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Requests per minute limit
    pub requests_per_minute: Option<usize>,
    /// Tokens per minute limit
    pub tokens_per_minute: Option<usize>,
    /// Concurrent requests limit
    pub concurrent_requests: Option<usize>,
    /// Rate limit tracking window
    pub window_duration: Duration,
    /// Atomic counters for tracking
    #[serde(skip)]
    pub request_counter: Arc<CachePadded<AtomicUsize>>,
    #[serde(skip)]
    pub token_counter: Arc<CachePadded<AtomicUsize>>,
    #[serde(skip)]
    pub concurrent_counter: Arc<CachePadded<AtomicUsize>>,
}

impl RateLimitConfig {
    /// Create new rate limit configuration
    #[inline]
    pub fn new() -> Self {
        Self {
            enabled: true,
            requests_per_minute: Some(1000),
            tokens_per_minute: Some(90000),
            concurrent_requests: Some(10),
            window_duration: Duration::from_secs(60),
            request_counter: Arc::new(CachePadded::new(AtomicUsize::new(0))),
            token_counter: Arc::new(CachePadded::new(AtomicUsize::new(0))),
            concurrent_counter: Arc::new(CachePadded::new(AtomicUsize::new(0))),
        }
    }

    /// Check if request is allowed
    pub fn check_request_limit(&self) -> bool {
        if !self.enabled {
            return true;
        }

        if let Some(limit) = self.requests_per_minute {
            let current = self.request_counter.load(Ordering::Relaxed);
            if current >= limit {
                return false;
            }
        }

        if let Some(limit) = self.concurrent_requests {
            let current = self.concurrent_counter.load(Ordering::Relaxed);
            if current >= limit {
                return false;
            }
        }

        true
    }

    /// Check if token usage is allowed
    #[inline]
    pub fn check_token_limit(&self, tokens: usize) -> bool {
        if !self.enabled {
            return true;
        }

        if let Some(limit) = self.tokens_per_minute {
            let current = self.token_counter.load(Ordering::Relaxed);
            return current + tokens <= limit;
        }

        true
    }

    /// Record request start
    #[inline]
    pub fn record_request_start(&self) {
        if self.enabled {
            self.request_counter.fetch_add(1, Ordering::Relaxed);
            self.concurrent_counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record request completion
    #[inline]
    pub fn record_request_complete(&self, tokens_used: usize) {
        if self.enabled {
            self.token_counter.fetch_add(tokens_used, Ordering::Relaxed);
            self.concurrent_counter.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Reset counters for new window
    #[inline]
    pub fn reset_window(&self) {
        if self.enabled {
            self.request_counter.store(0, Ordering::Relaxed);
            self.token_counter.store(0, Ordering::Relaxed);
            // Don't reset concurrent counter as it tracks active requests
        }
    }

    /// Get current usage statistics
    #[inline]
    pub fn current_usage(&self) -> (usize, usize, usize) {
        (
            self.request_counter.load(Ordering::Relaxed),
            self.token_counter.load(Ordering::Relaxed),
            self.concurrent_counter.load(Ordering::Relaxed),
        )
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Retry configuration for reliability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Enable retries
    pub enabled: bool,
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Initial retry delay
    pub initial_delay: Duration,
    /// Maximum retry delay
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Enable jitter
    pub enable_jitter: bool,
    /// Retryable status codes
    pub retryable_status_codes: Vec<u16>,
}

impl RetryConfig {
    /// Create optimized retry configuration
    #[inline]
    pub fn optimized() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            initial_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            enable_jitter: true,
            retryable_status_codes: vec![429, 500, 502, 503, 504],
        }
    }

    /// Calculate next retry delay
    pub fn next_delay(&self, attempt: usize) -> Duration {
        if attempt >= self.max_retries {
            return self.max_delay;
        }

        let delay_ms =
            self.initial_delay.as_millis() as f64 * self.backoff_multiplier.powi(attempt as i32);

        let delay = Duration::from_millis(delay_ms as u64).min(self.max_delay);

        if self.enable_jitter {
            let jitter = fastrand::f64() * 0.5; // 0 to 50% jitter
            let jittered_delay = delay.as_millis() as f64 * (1.0 + jitter);
            Duration::from_millis(jittered_delay as u64)
        } else {
            delay
        }
    }

    /// Check if status code is retryable
    #[inline]
    pub fn is_retryable_status(&self, status_code: u16) -> bool {
        self.retryable_status_codes.contains(&status_code)
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self::optimized()
    }
}

/// Response caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable response caching
    pub enabled: bool,
    /// Cache size limit (number of entries)
    pub max_entries: usize,
    /// Cache entry TTL
    pub ttl: Duration,
    /// Enable cache for streaming responses
    pub cache_streaming: bool,
    /// Cache key strategy
    pub key_strategy: CacheKeyStrategy,
}

/// Cache key generation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CacheKeyStrategy {
    /// Hash of request parameters
    RequestHash = 0,
    /// Hash of request content only
    ContentHash = 1,
    /// Custom key generation
    Custom = 2,
}

impl CacheConfig {
    /// Create optimized cache configuration
    #[inline]
    pub fn optimized() -> Self {
        Self {
            enabled: true,
            max_entries: 10000,
            ttl: Duration::from_secs(3600), // 1 hour
            cache_streaming: false,         // Don't cache streaming by default
            key_strategy: CacheKeyStrategy::RequestHash,
        }
    }

    /// Create disabled cache configuration
    #[inline]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            max_entries: 0,
            ttl: Duration::from_secs(0),
            cache_streaming: false,
            key_strategy: CacheKeyStrategy::RequestHash,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self::optimized()
    }
}

/// Token usage tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTrackingConfig {
    /// Enable token tracking
    pub enabled: bool,
    /// Track input tokens
    pub track_input_tokens: bool,
    /// Track output tokens
    pub track_output_tokens: bool,
    /// Track total cost
    pub track_cost: bool,
    /// Atomic counters for tracking
    #[serde(skip)]
    pub input_tokens: Arc<CachePadded<AtomicU64>>,
    #[serde(skip)]
    pub output_tokens: Arc<CachePadded<AtomicU64>>,
    #[serde(skip)]
    pub total_requests: Arc<CachePadded<AtomicU64>>,
}

impl TokenTrackingConfig {
    /// Create new token tracking configuration
    #[inline]
    pub fn new() -> Self {
        Self {
            enabled: true,
            track_input_tokens: true,
            track_output_tokens: true,
            track_cost: true,
            input_tokens: Arc::new(CachePadded::new(AtomicU64::new(0))),
            output_tokens: Arc::new(CachePadded::new(AtomicU64::new(0))),
            total_requests: Arc::new(CachePadded::new(AtomicU64::new(0))),
        }
    }

    /// Record token usage
    #[inline]
    pub fn record_usage(&self, input_tokens: u64, output_tokens: u64) {
        if self.enabled {
            if self.track_input_tokens {
                self.input_tokens.fetch_add(input_tokens, Ordering::Relaxed);
            }
            if self.track_output_tokens {
                self.output_tokens
                    .fetch_add(output_tokens, Ordering::Relaxed);
            }
            self.total_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get current token usage
    #[inline]
    pub fn current_usage(&self) -> (u64, u64, u64) {
        (
            self.input_tokens.load(Ordering::Relaxed),
            self.output_tokens.load(Ordering::Relaxed),
            self.total_requests.load(Ordering::Relaxed),
        )
    }

    /// Reset counters
    #[inline]
    pub fn reset(&self) {
        self.input_tokens.store(0, Ordering::Relaxed);
        self.output_tokens.store(0, Ordering::Relaxed);
        self.total_requests.store(0, Ordering::Relaxed);
    }
}

impl Default for TokenTrackingConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl LLMConfig {
    /// Create new LLM configuration
    pub fn new(provider: LLMProvider, model_name: impl Into<Arc<str>>) -> MemoryResult<Self> {
        let model_config = ModelConfig::new(model_name)?;
        let endpoint_config = EndpointConfig::new(provider.default_api_base());

        Ok(Self {
            provider,
            model_config,
            endpoint_config,
            streaming_config: if provider.supports_streaming() {
                StreamingConfig::optimized()
            } else {
                StreamingConfig::minimal()
            },
            rate_limit_config: RateLimitConfig::new(),
            retry_config: RetryConfig::optimized(),
            cache_config: CacheConfig::optimized(),
            token_tracking: TokenTrackingConfig::new(),
        })
    }

    /// Set API key
    #[inline]
    pub fn with_api_key(mut self, api_key: impl Into<Arc<str>>) -> Self {
        self.endpoint_config = self.endpoint_config.with_api_key(api_key);
        self
    }

    /// Set model configuration
    #[inline]
    pub fn with_model_config(mut self, config: ModelConfig) -> Self {
        self.model_config = config;
        self
    }

    /// Set streaming configuration
    #[inline]
    pub fn with_streaming_config(mut self, config: StreamingConfig) -> Self {
        self.streaming_config = config;
        self
    }

    /// Enable or disable streaming
    #[inline]
    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.streaming_config.enable_streaming = enabled && self.provider.supports_streaming();
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> MemoryResult<()> {
        // Check if streaming is enabled for non-streaming providers
        if self.streaming_config.enable_streaming && !self.provider.supports_streaming() {
            return Err(MemoryError::validation(format!(
                "Provider {} does not support streaming",
                self.provider
            )));
        }

        // Validate API key is provided for providers that require it
        if matches!(
            self.provider,
            LLMProvider::OpenAI
                | LLMProvider::Anthropic
                | LLMProvider::Google
                | LLMProvider::Meta
                | LLMProvider::Mistral
                | LLMProvider::Cohere
        ) && self.endpoint_config.api_key.is_none()
        {
            return Err(MemoryError::validation(format!(
                "API key is required for provider {}",
                self.provider
            )));
        }

        // Validate model name is not empty
        if self.model_config.model_name.is_empty() {
            return Err(MemoryError::validation("Model name cannot be empty"));
        }

        Ok(())
    }

    /// Check if configuration supports HTTP3
    #[inline]
    pub fn supports_http3(&self) -> bool {
        // HTTP3 is supported for all HTTPS endpoints
        self.endpoint_config.api_base.starts_with("https://")
    }

    /// Get optimal configuration for provider
    pub fn optimal_for_provider(provider: LLMProvider) -> MemoryResult<Self> {
        let mut config = Self::new(provider, provider.default_model())?;

        // Optimize based on provider characteristics
        match provider {
            LLMProvider::OpenAI => {
                config.model_config = config
                    .model_config
                    .with_temperature(0.7)?
                    .with_max_tokens(4096)?;
            }
            LLMProvider::Anthropic => {
                config.model_config = config
                    .model_config
                    .with_temperature(0.5)?
                    .with_max_tokens(8192)?;
            }
            LLMProvider::Google => {
                config.model_config = config
                    .model_config
                    .with_temperature(0.3)?
                    .with_max_tokens(2048)?;
            }
            LLMProvider::Ollama => {
                config.streaming_config = StreamingConfig::optimized();
                config.endpoint_config.timeout = Duration::from_secs(120); // Longer timeout for local models
            }
            _ => {}
        }

        Ok(config)
    }
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self::optimal_for_provider(LLMProvider::OpenAI)
            .expect("Default LLM configuration should be valid")
    }
}
