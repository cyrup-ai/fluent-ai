//! Model configuration types and utilities
//!
//! This module provides configuration structures for AI model settings,
//! performance tuning, and retry policies with zero-allocation patterns.

use std::collections::HashMap;
use std::sync::Arc;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};

/// Configuration for AI model settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name/identifier
    pub model_name: Arc<str>,
    /// Model provider
    pub provider: Arc<str>,
    /// API endpoint URL
    pub endpoint: Option<String>,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Model parameters
    pub parameters: ModelParameters,
    /// Retry configuration
    pub retry_config: ModelRetryConfig,
    /// Performance configuration
    pub performance_config: ModelPerformanceConfig,
    /// Custom headers
    pub custom_headers: HashMap<String, String>,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Enable streaming responses
    pub enable_streaming: bool,
    /// Model version
    pub version: Option<String>,
    /// Temperature setting
    pub temperature: f32,
    /// Maximum tokens
    pub max_tokens: Option<usize>,
    /// Top-p sampling
    pub top_p: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<usize>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Model metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Model parameters for fine-tuning behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Context window size
    pub context_window: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Number of attention heads
    pub attention_heads: Option<usize>,
    /// Hidden layer size
    pub hidden_size: Option<usize>,
    /// Number of layers
    pub num_layers: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Embedding dimension
    pub embedding_dim: Option<usize>,
    /// Dropout rate
    pub dropout_rate: Option<f32>,
    /// Learning rate
    pub learning_rate: Option<f32>,
    /// Weight decay
    pub weight_decay: Option<f32>,
    /// Gradient clipping
    pub gradient_clip: Option<f32>,
    /// Custom parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

/// Retry configuration for model requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRetryConfig {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Base delay between retries (milliseconds)
    pub base_delay_ms: u64,
    /// Maximum delay between retries (milliseconds)
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Enable jitter in retry delays
    pub enable_jitter: bool,
    /// Retry on specific error codes
    pub retry_on_codes: Vec<u16>,
    /// Retry on timeout
    pub retry_on_timeout: bool,
    /// Retry on rate limit
    pub retry_on_rate_limit: bool,
}

/// Performance configuration for model optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceConfig {
    /// Enable response caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable request batching
    pub enable_batching: bool,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Connection pool size
    pub connection_pool_size: usize,
    /// Keep-alive timeout
    pub keep_alive_timeout_seconds: u64,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionType,
    /// Memory limit in bytes
    pub memory_limit_bytes: usize,
    /// CPU thread limit
    pub cpu_thread_limit: Option<usize>,
    /// GPU memory limit
    pub gpu_memory_limit_bytes: Option<usize>,
    /// Enable quantization
    pub enable_quantization: bool,
    /// Quantization bits
    pub quantization_bits: Option<u8>,
}

/// Compression type for model responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// Deflate compression
    Deflate,
    /// Brotli compression
    Brotli,
    /// LZ4 compression
    Lz4,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_name: Arc::from("default"),
            provider: Arc::from("openai"),
            endpoint: None,
            api_key: None,
            parameters: ModelParameters::default(),
            retry_config: ModelRetryConfig::default(),
            performance_config: ModelPerformanceConfig::default(),
            custom_headers: HashMap::new(),
            timeout_seconds: 30,
            enable_streaming: true,
            version: None,
            temperature: 0.7,
            max_tokens: Some(2048),
            top_p: Some(0.9),
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            context_window: 4096,
            batch_size: 1,
            attention_heads: None,
            hidden_size: None,
            num_layers: None,
            vocab_size: None,
            embedding_dim: None,
            dropout_rate: None,
            learning_rate: None,
            weight_decay: None,
            gradient_clip: None,
            custom_params: HashMap::new(),
        }
    }
}

impl Default for ModelRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            enable_jitter: true,
            retry_on_codes: vec![429, 500, 502, 503, 504],
            retry_on_timeout: true,
            retry_on_rate_limit: true,
        }
    }
}

impl Default for ModelPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl_seconds: 3600,
            enable_batching: false,
            batch_timeout_ms: 100,
            connection_pool_size: 10,
            keep_alive_timeout_seconds: 60,
            enable_compression: true,
            compression_algorithm: CompressionType::Gzip,
            memory_limit_bytes: 1024 * 1024 * 1024, // 1GB
            cpu_thread_limit: None,
            gpu_memory_limit_bytes: None,
            enable_quantization: false,
            quantization_bits: None,
        }
    }
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(model_name: Arc<str>, provider: Arc<str>) -> Self {
        Self {
            model_name,
            provider,
            ..Default::default()
        }
    }

    /// Validate the model configuration (streaming)
    pub fn validate(&self) -> AsyncStream<ValidationResult> {
        let config = self.clone();
        
        AsyncStream::with_channel(move |sender| {
            let mut errors = Vec::new();
            let mut warnings = Vec::new();

            // Validate model name
            if config.model_name.is_empty() {
                errors.push("Model name cannot be empty".to_string());
            }

            // Validate provider
            if config.provider.is_empty() {
                errors.push("Provider cannot be empty".to_string());
            }

            // Validate temperature
            if config.temperature < 0.0 || config.temperature > 2.0 {
                warnings.push("Temperature should be between 0.0 and 2.0".to_string());
            }

            // Validate max_tokens
            if let Some(max_tokens) = config.max_tokens {
                if max_tokens == 0 {
                    errors.push("Max tokens must be greater than 0".to_string());
                }
                if max_tokens > config.parameters.context_window {
                    warnings.push("Max tokens exceeds context window".to_string());
                }
            }

            // Validate timeout
            if config.timeout_seconds == 0 {
                errors.push("Timeout must be greater than 0".to_string());
            }

            let result = ValidationResult {
                is_valid: errors.is_empty(),
                errors,
                warnings,
                config_hash: Self::calculate_hash(&config),
            };

            let _ = sender.send(result);
        })
    }

    /// Calculate configuration hash for caching
    fn calculate_hash(config: &ModelConfig) -> String {
        // Simple hash calculation (would use proper hashing in production)
        format!("{:?}", config).chars().fold(0u64, |acc, c| {
            acc.wrapping_mul(31).wrapping_add(c as u64)
        }).to_string()
    }

    /// Update configuration with new values (streaming)
    pub fn update(&mut self, updates: ConfigUpdate) -> AsyncStream<UpdateResult> {
        AsyncStream::with_channel(move |sender| {
            let mut changes = Vec::new();

            // Apply updates and track changes
            if let Some(temperature) = updates.temperature {
                changes.push(format!("Temperature: {} -> {}", self.temperature, temperature));
                self.temperature = temperature;
            }

            if let Some(max_tokens) = updates.max_tokens {
                changes.push(format!("Max tokens: {:?} -> {:?}", self.max_tokens, max_tokens));
                self.max_tokens = max_tokens;
            }

            if let Some(timeout) = updates.timeout_seconds {
                changes.push(format!("Timeout: {} -> {}", self.timeout_seconds, timeout));
                self.timeout_seconds = timeout;
            }

            let result = UpdateResult {
                success: true,
                changes_applied: changes.len(),
                changes,
                updated_at: chrono::Utc::now(),
            };

            let _ = sender.send(result);
        })
    }

    /// Get configuration summary
    pub fn summary(&self) -> ConfigSummary {
        ConfigSummary {
            model_name: self.model_name.clone(),
            provider: self.provider.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            streaming_enabled: self.enable_streaming,
            cache_enabled: self.performance_config.enable_caching,
            retry_count: self.retry_config.max_retries,
            timeout_seconds: self.timeout_seconds,
        }
    }
}

/// Result of configuration validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether configuration is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Configuration hash
    pub config_hash: String,
}

/// Configuration update parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigUpdate {
    /// New temperature value
    pub temperature: Option<f32>,
    /// New max tokens value
    pub max_tokens: Option<Option<usize>>,
    /// New timeout value
    pub timeout_seconds: Option<u64>,
    /// New streaming setting
    pub enable_streaming: Option<bool>,
    /// New caching setting
    pub enable_caching: Option<bool>,
}

/// Result of configuration update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateResult {
    /// Whether update was successful
    pub success: bool,
    /// Number of changes applied
    pub changes_applied: usize,
    /// List of changes made
    pub changes: Vec<String>,
    /// Update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Configuration summary for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSummary {
    /// Model name
    pub model_name: Arc<str>,
    /// Provider name
    pub provider: Arc<str>,
    /// Temperature setting
    pub temperature: f32,
    /// Max tokens setting
    pub max_tokens: Option<usize>,
    /// Streaming enabled
    pub streaming_enabled: bool,
    /// Cache enabled
    pub cache_enabled: bool,
    /// Retry count
    pub retry_count: usize,
    /// Timeout in seconds
    pub timeout_seconds: u64,
}

impl Default for CompressionType {
    fn default() -> Self {
        CompressionType::Gzip
    }
}