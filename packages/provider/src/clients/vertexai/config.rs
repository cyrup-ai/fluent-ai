//! Zero-allocation configuration management for VertexAI client
//!
//! Provides hot-swappable configuration with `arc-swap` and fixed-size storage.

use crate::clients::vertexai::{VertexAIError, VertexAIResult, ProjectId, Region, VertexString};
use arrayvec::ArrayString;
use arc_swap::{ArcSwap, Guard};
use atomic_counter::{AtomicCounter, RelaxedCounter};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

/// Global configuration instance with hot-swappable updates
static VERTEXAI_CONFIG: LazyLock<ArcSwap<VertexAIConfig>> = LazyLock::new(|| {
    ArcSwap::from_pointee(VertexAIConfig::default())
});

/// Configuration validation counter
static CONFIG_VALIDATIONS: RelaxedCounter = RelaxedCounter::new(0);

/// Configuration update counter
static CONFIG_UPDATES: RelaxedCounter = RelaxedCounter::new(0);

/// Zero-allocation VertexAI client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexAIConfig {
    /// Google Cloud project ID (fixed allocation)
    pub project_id: ProjectId,
    
    /// Google Cloud region (zero allocation)
    pub region: Region,
    
    /// Service account email (fixed allocation)
    pub service_account_email: VertexString,
    
    /// Service account private key (truncated for security)
    pub private_key_id: ArrayString<64>,
    
    /// Base endpoint URL (zero allocation)
    pub base_url: &'static str,
    
    /// Request timeout configuration
    pub timeout: TimeoutConfig,
    
    /// Retry configuration  
    pub retry: RetryConfig,
    
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    
    /// Authentication configuration
    pub auth: AuthConfig,
    
    /// Model-specific configurations
    pub model_defaults: ModelDefaults,
    
    /// Performance tuning parameters
    pub performance: PerformanceConfig,
}

/// Timeout configuration with specific limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,
    
    /// Streaming timeout in milliseconds
    pub streaming_timeout_ms: u64,
    
    /// Token refresh timeout in milliseconds
    pub token_refresh_timeout_ms: u64,
}

/// Exponential backoff retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    
    /// Initial backoff delay in milliseconds
    pub initial_backoff_ms: u64,
    
    /// Maximum backoff delay in milliseconds
    pub max_backoff_ms: u64,
    
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
    
    /// Jitter factor to randomize backoff (0.0 to 1.0)
    pub jitter_factor: f64,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per minute limit
    pub requests_per_minute: u32,
    
    /// Tokens per minute limit
    pub tokens_per_minute: u32,
    
    /// Burst capacity for short spikes
    pub burst_capacity: u32,
    
    /// Rate limit enforcement enabled
    pub enabled: bool,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Token refresh margin in seconds before expiry
    pub token_refresh_margin_seconds: u64,
    
    /// Maximum token lifetime in seconds
    pub max_token_lifetime_seconds: u64,
    
    /// OAuth2 scope for Vertex AI
    pub oauth_scope: &'static str,
    
    /// JWT algorithm (always RS256 for Google)
    pub jwt_algorithm: &'static str,
    
    /// Token endpoint URL
    pub token_endpoint: &'static str,
}

/// Model default configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefaults {
    /// Default temperature for generation
    pub temperature: f32,
    
    /// Default top-p for nucleus sampling
    pub top_p: f32,
    
    /// Default top-k for token filtering
    pub top_k: u32,
    
    /// Default max output tokens
    pub max_output_tokens: u32,
    
    /// Default candidate count
    pub candidate_count: u32,
    
    /// Stop sequences (fixed allocation)
    pub stop_sequences: [ArrayString<32>; 4],
    
    /// Number of stop sequences used
    pub stop_sequences_count: usize,
}

/// Performance configuration for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Connection pool size
    pub connection_pool_size: usize,
    
    /// Request buffer size in bytes
    pub request_buffer_size: usize,
    
    /// Response buffer size in bytes  
    pub response_buffer_size: usize,
    
    /// Enable HTTP/3 QUIC transport
    pub enable_http3: bool,
    
    /// Enable compression
    pub enable_compression: bool,
    
    /// Keep-alive timeout in seconds
    pub keep_alive_timeout_seconds: u64,
    
    /// Enable metrics collection
    pub enable_metrics: bool,
}

impl Default for VertexAIConfig {
    fn default() -> Self {
        Self {
            project_id: ProjectId::new(),
            region: "us-central1",
            service_account_email: VertexString::new(),
            private_key_id: ArrayString::new(),
            base_url: "https://us-central1-aiplatform.googleapis.com",
            timeout: TimeoutConfig::default(),
            retry: RetryConfig::default(),
            rate_limit: RateLimitConfig::default(),
            auth: AuthConfig::default(),
            model_defaults: ModelDefaults::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            connection_timeout_ms: 10_000,
            request_timeout_ms: 30_000,
            streaming_timeout_ms: 300_000,
            token_refresh_timeout_ms: 5_000,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 1_000,
            max_backoff_ms: 60_000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 300,
            tokens_per_minute: 300_000,
            burst_capacity: 10,
            enabled: true,
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            token_refresh_margin_seconds: 300,
            max_token_lifetime_seconds: 3600,
            oauth_scope: "https://www.googleapis.com/auth/cloud-platform",
            jwt_algorithm: "RS256",
            token_endpoint: "https://oauth2.googleapis.com/token",
        }
    }
}

impl Default for ModelDefaults {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            max_output_tokens: 1024,
            candidate_count: 1,
            stop_sequences: [ArrayString::new(); 4],
            stop_sequences_count: 0,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            connection_pool_size: 10,
            request_buffer_size: 8192,
            response_buffer_size: 16384,
            enable_http3: true,
            enable_compression: true,
            keep_alive_timeout_seconds: 90,
            enable_metrics: true,
        }
    }
}

impl VertexAIConfig {
    /// Create new configuration with project and region
    pub fn new(project_id: &str, region: Region) -> VertexAIResult<Self> {
        CONFIG_VALIDATIONS.inc();
        
        let project_id = ProjectId::from(project_id).map_err(|_| {
            VertexAIError::Config {
                parameter: "project_id".to_string(),
                issue: "Project ID too long, maximum 64 characters".to_string(),
            }
        })?;
        
        let mut config = Self::default();
        config.project_id = project_id;
        config.region = region;
        config.base_url = Self::build_base_url(region);
        
        config.validate()?;
        Ok(config)
    }
    
    /// Create configuration with service account JSON
    pub fn with_service_account(
        project_id: &str,
        region: Region,
        service_account_email: &str,
        private_key_id: &str,
    ) -> VertexAIResult<Self> {
        let mut config = Self::new(project_id, region)?;
        
        config.service_account_email = VertexString::from(service_account_email).map_err(|_| {
            VertexAIError::Config {
                parameter: "service_account_email".to_string(),
                issue: "Service account email too long, maximum 256 characters".to_string(),
            }
        })?;
        
        config.private_key_id = ArrayString::from(private_key_id).map_err(|_| {
            VertexAIError::Config {
                parameter: "private_key_id".to_string(),
                issue: "Private key ID too long, maximum 64 characters".to_string(),
            }
        })?;
        
        config.validate()?;
        Ok(config)
    }
    
    /// Build base URL from region
    fn build_base_url(region: Region) -> &'static str {
        match region {
            "us-central1" => "https://us-central1-aiplatform.googleapis.com",
            "us-east1" => "https://us-east1-aiplatform.googleapis.com",
            "us-west1" => "https://us-west1-aiplatform.googleapis.com",
            "europe-west1" => "https://europe-west1-aiplatform.googleapis.com",
            "europe-west4" => "https://europe-west4-aiplatform.googleapis.com",
            "asia-northeast1" => "https://asia-northeast1-aiplatform.googleapis.com",
            "asia-southeast1" => "https://asia-southeast1-aiplatform.googleapis.com",
            _ => "https://us-central1-aiplatform.googleapis.com", // Default fallback
        }
    }
    
    /// Validate configuration completeness and correctness
    pub fn validate(&self) -> VertexAIResult<()> {
        if self.project_id.is_empty() {
            return Err(VertexAIError::Config {
                parameter: "project_id".to_string(),
                issue: "Project ID cannot be empty".to_string(),
            });
        }
        
        if self.timeout.request_timeout_ms == 0 {
            return Err(VertexAIError::Config {
                parameter: "request_timeout_ms".to_string(),
                issue: "Request timeout must be greater than 0".to_string(),
            });
        }
        
        if self.retry.max_attempts == 0 {
            return Err(VertexAIError::Config {
                parameter: "max_attempts".to_string(),
                issue: "Max retry attempts must be greater than 0".to_string(),
            });
        }
        
        if !(0.0..=1.0).contains(&self.retry.jitter_factor) {
            return Err(VertexAIError::Config {
                parameter: "jitter_factor".to_string(),
                issue: "Jitter factor must be between 0.0 and 1.0".to_string(),
            });
        }
        
        if self.model_defaults.temperature < 0.0 || self.model_defaults.temperature > 2.0 {
            return Err(VertexAIError::Config {
                parameter: "temperature".to_string(),
                issue: "Temperature must be between 0.0 and 2.0".to_string(),
            });
        }
        
        if !(0.0..=1.0).contains(&self.model_defaults.top_p) {
            return Err(VertexAIError::Config {
                parameter: "top_p".to_string(),
                issue: "Top-p must be between 0.0 and 1.0".to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Get timeout as Duration
    pub fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.timeout.request_timeout_ms)
    }
    
    /// Get streaming timeout as Duration
    pub fn streaming_timeout(&self) -> Duration {
        Duration::from_millis(self.timeout.streaming_timeout_ms)
    }
    
    /// Get connection timeout as Duration
    pub fn connection_timeout(&self) -> Duration {
        Duration::from_millis(self.timeout.connection_timeout_ms)
    }
}

/// Global configuration management functions
impl VertexAIConfig {
    /// Get current global configuration (zero allocation)
    pub fn global() -> Guard<Arc<VertexAIConfig>> {
        VERTEXAI_CONFIG.load()
    }
    
    /// Update global configuration atomically
    pub fn update_global(config: VertexAIConfig) -> VertexAIResult<()> {
        config.validate()?;
        VERTEXAI_CONFIG.store(Arc::new(config));
        CONFIG_UPDATES.inc();
        Ok(())
    }
    
    /// Get configuration statistics
    pub fn stats() -> (usize, usize) {
        (CONFIG_VALIDATIONS.get(), CONFIG_UPDATES.get())
    }
}