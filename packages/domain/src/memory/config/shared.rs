//! Shared configuration types for memory system
//!
//! This module provides canonical definitions of configuration types that are
//! used across multiple memory subsystems to eliminate duplication.

use std::sync::Arc;
use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Canonical retry configuration with exponential backoff and jitter
/// 
/// This is the single source of truth for retry configuration across all memory subsystems.
/// Combines fields from both database and LLM retry configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Enable retries (from LLM config)
    pub enabled: bool,
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Initial retry delay
    pub initial_delay: Duration,
    /// Maximum retry delay
    pub max_delay: Duration,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Enable jitter to prevent thundering herd
    pub enable_jitter: bool,
    /// Retryable HTTP status codes (from LLM config)
    pub retryable_status_codes: Vec<u16>,
}

impl RetryConfig {
    /// Create default retry configuration with exponential backoff
    #[inline]
    pub fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            enable_jitter: true,
            retryable_status_codes: vec![429, 500, 502, 503, 504],
        }
    }

    /// Create optimized retry configuration for high-performance scenarios
    #[inline]
    pub fn optimized() -> Self {
        Self {
            enabled: true,
            max_retries: 5,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 1.5,
            enable_jitter: true,
            retryable_status_codes: vec![429, 500, 502, 503, 504, 408, 409, 423, 424],
        }
    }

    /// Create aggressive retry configuration for critical operations
    #[inline]
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            max_retries: 10,
            initial_delay: Duration::from_millis(25),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            enable_jitter: true,
            retryable_status_codes: vec![429, 500, 502, 503, 504, 408, 409, 423, 424, 425, 510, 511],
        }
    }

    /// Create minimal retry configuration for testing
    #[inline]
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            max_retries: 1,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            backoff_multiplier: 1.0,
            enable_jitter: false,
            retryable_status_codes: vec![429, 500, 502, 503],
        }
    }

    /// Create disabled retry configuration
    #[inline]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            max_retries: 0,
            initial_delay: Duration::from_millis(0),
            max_delay: Duration::from_millis(0),
            backoff_multiplier: 1.0,
            enable_jitter: false,
            retryable_status_codes: vec![],
        }
    }

    /// Check if a status code is retryable
    #[inline]
    pub fn is_retryable_status(&self, status_code: u16) -> bool {
        self.enabled && self.retryable_status_codes.contains(&status_code)
    }

    /// Calculate delay for a given retry attempt with optional jitter
    #[inline]
    pub fn calculate_delay(&self, attempt: usize) -> Duration {
        if !self.enabled || attempt == 0 {
            return Duration::from_millis(0);
        }

        let base_delay = self.initial_delay.as_millis() as f64;
        let multiplier = self.backoff_multiplier.powi(attempt.saturating_sub(1) as i32);
        let mut delay_ms = base_delay * multiplier;

        // Apply maximum delay limit
        let max_delay_ms = self.max_delay.as_millis() as f64;
        if delay_ms > max_delay_ms {
            delay_ms = max_delay_ms;
        }

        // Apply jitter if enabled (±25% variation)
        if self.enable_jitter {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            
            let mut hasher = DefaultHasher::new();
            attempt.hash(&mut hasher);
            let hash = hasher.finish();
            
            // Convert hash to a value between 0.75 and 1.25
            let jitter = 0.75 + (hash % 500) as f64 / 1000.0;
            delay_ms *= jitter;
        }

        Duration::from_millis(delay_ms as u64)
    }

    /// Validate the retry configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_retries > 100 {
            return Err("Maximum retries cannot exceed 100".to_string());
        }

        if self.initial_delay > self.max_delay {
            return Err("Initial delay cannot be greater than maximum delay".to_string());
        }

        if self.backoff_multiplier <= 0.0 {
            return Err("Backoff multiplier must be positive".to_string());
        }

        if self.backoff_multiplier > 10.0 {
            return Err("Backoff multiplier cannot exceed 10.0".to_string());
        }

        Ok(())
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self::default()
    }
}

/// Canonical embedding configuration
/// 
/// This is the single source of truth for embedding configuration across memory subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding model type
    pub model_type: EmbeddingModelType,
    /// Model name or identifier
    pub model_name: Arc<str>,
    /// Expected embedding dimension
    pub dimension: usize,
    /// Normalization strategy for embeddings
    pub normalization: NormalizationStrategy,
    /// Enable caching of embeddings
    pub enable_caching: bool,
    /// Cache size for embeddings
    pub cache_size: usize,
    /// Cache time-to-live
    pub cache_ttl: Duration,
    /// Enable compression of cached embeddings
    pub enable_compression: bool,
}

/// Embedding model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EmbeddingModelType {
    /// OpenAI embedding models
    OpenAI = 0,
    /// Sentence transformers
    SentenceTransformers = 1,
    /// Cohere embeddings
    Cohere = 2,
    /// Custom model
    Custom = 3,
}

/// Normalization strategies for embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum NormalizationStrategy {
    /// No normalization
    None = 0,
    /// L2 normalization (unit vectors)
    L2 = 1,
    /// L1 normalization
    L1 = 2,
    /// Min-max normalization
    MinMax = 3,
}

impl EmbeddingConfig {
    /// Create default embedding configuration
    #[inline]
    pub fn default() -> Self {
        Self {
            model_type: EmbeddingModelType::OpenAI,
            model_name: Arc::from("text-embedding-3-small"),
            dimension: 1536,
            normalization: NormalizationStrategy::L2,
            enable_caching: true,
            cache_size: 10_000,
            cache_ttl: Duration::from_secs(3600),
            enable_compression: false,
        }
    }

    /// Create high-performance embedding configuration
    #[inline]
    pub fn high_performance() -> Self {
        Self {
            model_type: EmbeddingModelType::OpenAI,
            model_name: Arc::from("text-embedding-3-large"),
            dimension: 3072,
            normalization: NormalizationStrategy::L2,
            enable_caching: true,
            cache_size: 50_000,
            cache_ttl: Duration::from_secs(7200),
            enable_compression: true,
        }
    }

    /// Create compact embedding configuration for memory-constrained environments
    #[inline]
    pub fn compact() -> Self {
        Self {
            model_type: EmbeddingModelType::SentenceTransformers,
            model_name: Arc::from("all-MiniLM-L6-v2"),
            dimension: 384,
            normalization: NormalizationStrategy::L2,
            enable_caching: true,
            cache_size: 1_000,
            cache_ttl: Duration::from_secs(1800),
            enable_compression: true,
        }
    }

    /// Validate the embedding configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.dimension == 0 {
            return Err("Embedding dimension must be greater than 0".to_string());
        }

        if self.dimension > 10_000 {
            return Err("Embedding dimension cannot exceed 10,000".to_string());
        }

        if self.cache_size == 0 && self.enable_caching {
            return Err("Cache size must be greater than 0 when caching is enabled".to_string());
        }

        if self.model_name.is_empty() {
            return Err("Model name cannot be empty".to_string());
        }

        Ok(())
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self::default()
    }
}