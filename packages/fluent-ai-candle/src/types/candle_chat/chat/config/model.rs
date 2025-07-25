//! Model configuration for chat interactions
//!
//! This module defines model-specific settings including provider selection,
//! model parameters, performance tuning, and behavior customization.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Model validator for configuration validation
#[derive(Debug, Clone)]
pub struct ModelValidator;

impl ModelValidator {
    /// Validate model configuration
    pub fn validate(&self, config: &ModelConfig) -> Result<(), String> {
        if config.provider.is_empty() {
            return Err("Provider cannot be empty".to_string());
        }
        if config.model_name.is_empty() {
            return Err("Model name cannot be empty".to_string());
        }
        if config.temperature < 0.0 || config.temperature > 2.0 {
            return Err("Temperature must be between 0.0 and 2.0".to_string());
        }
        Ok(())
    }
}

/// Duration serialization helper
mod duration_secs {
    use super::*;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

/// Model configuration for chat interactions
///
/// This configuration defines model-specific settings including provider selection,
/// model parameters, performance tuning, and behavior customization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model provider (e.g., "openai", "anthropic", "mistral", "gemini")
    pub provider: Arc<str>,
    /// Model name/identifier
    pub model_name: Arc<str>,
    /// Model version or variant
    pub model_version: Option<Arc<str>>,
    /// Temperature for response randomness (0.0 to 2.0)
    pub temperature: f32,
    /// Maximum tokens in response
    pub max_tokens: Option<u32>,
    /// Top-p nucleus sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    pub top_k: Option<u32>,
    /// Frequency penalty (-2.0 to 2.0)
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (-2.0 to 2.0)
    pub presence_penalty: Option<f32>,
    /// Stop sequences
    pub stop_sequences: Vec<Arc<str>>,
    /// System prompt/instructions
    pub system_prompt: Option<Arc<str>>,
    /// Enable function calling
    pub enable_functions: bool,
    /// Function calling mode ("auto", "none", "required")
    pub function_mode: Arc<str>,
    /// Model-specific parameters
    pub custom_parameters: HashMap<Arc<str>, serde_json::Value>,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Retry configuration
    pub retry_config: ModelRetryConfig,
    /// Performance settings
    pub performance: ModelPerformanceConfig,
}

/// Model retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Base delay between retries in milliseconds
    pub base_delay_ms: u64,
    /// Maximum delay between retries in milliseconds
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f32,
    /// Enable jitter to avoid thundering herd
    pub enable_jitter: bool,
}

/// Model performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceConfig {
    /// Enable response streaming
    pub enable_streaming: bool,
    /// Request batch size for bulk operations
    pub batch_size: u32,
    /// Connection pool size
    pub connection_pool_size: u32,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Keep-alive timeout in milliseconds
    #[serde(with = "duration_secs")]
    pub keep_alive_timeout: Duration,
    /// Enable response caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Maximum cache size in MB
    pub max_cache_size_mb: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: Arc::from("openai"),
            model_name: Arc::from("gpt-4"),
            model_version: None,
            temperature: 0.7,
            max_tokens: Some(2048),
            top_p: Some(0.9),
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: Vec::new(),
            system_prompt: None,
            enable_functions: false,
            function_mode: Arc::from("auto"),
            custom_parameters: HashMap::new(),
            timeout_ms: 30000, // 30 seconds
            retry_config: ModelRetryConfig::default(),
            performance: ModelPerformanceConfig::default(),
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
        }
    }
}

impl Default for ModelPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            batch_size: 1,
            connection_pool_size: 10,
            connection_timeout_ms: 5000,
            keep_alive_timeout: Duration::from_secs(30),
            enable_caching: true,
            cache_ttl_secs: 300, // 5 minutes
            max_cache_size_mb: 100,
        }
    }
}

impl ModelConfig {
    /// Create a new model configuration with provider and model name
    pub fn new(provider: impl Into<Arc<str>>, model_name: impl Into<Arc<str>>) -> Self {
        Self {
            provider: provider.into(),
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Set temperature with validation
    pub fn with_temperature(mut self, temperature: f32) -> Result<Self, String> {
        if !(0.0..=2.0).contains(&temperature) {
            return Err(format!(
                "Temperature must be between 0.0 and 2.0, got: {}",
                temperature
            ));
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Set max tokens with validation
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Result<Self, String> {
        if max_tokens == 0 || max_tokens > 32768 {
            return Err(format!(
                "Max tokens must be between 1 and 32768, got: {}",
                max_tokens
            ));
        }
        self.max_tokens = Some(max_tokens);
        Ok(self)
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<Arc<str>>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add stop sequence
    pub fn add_stop_sequence(mut self, sequence: impl Into<Arc<str>>) -> Self {
        self.stop_sequences.push(sequence.into());
        self
    }

    /// Set custom parameter
    pub fn set_custom_parameter(
        mut self,
        key: impl Into<Arc<str>>,
        value: serde_json::Value,
    ) -> Self {
        self.custom_parameters.insert(key.into(), value);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.provider.is_empty() {
            errors.push("Provider cannot be empty".to_string());
        }

        if self.model_name.is_empty() {
            errors.push("Model name cannot be empty".to_string());
        }

        if !(0.0..=2.0).contains(&self.temperature) {
            errors.push(format!(
                "Temperature must be between 0.0 and 2.0, got: {}",
                self.temperature
            ));
        }

        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 || max_tokens > 32768 {
                errors.push(format!(
                    "Max tokens must be between 1 and 32768, got: {}",
                    max_tokens
                ));
            }
        }

        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                errors.push(format!(
                    "Top-p must be between 0.0 and 1.0, got: {}",
                    top_p
                ));
            }
        }

        if let Some(frequency_penalty) = self.frequency_penalty {
            if !(-2.0..=2.0).contains(&frequency_penalty) {
                errors.push(format!(
                    "Frequency penalty must be between -2.0 and 2.0, got: {}",
                    frequency_penalty
                ));
            }
        }

        if let Some(presence_penalty) = self.presence_penalty {
            if !(-2.0..=2.0).contains(&presence_penalty) {
                errors.push(format!(
                    "Presence penalty must be between -2.0 and 2.0, got: {}",
                    presence_penalty
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get estimated tokens per minute based on configuration
    pub fn estimated_tokens_per_minute(&self) -> u32 {
        let base_rate = match self.provider.as_ref() {
            "openai" => 60000,
            "anthropic" => 50000,
            "mistral" => 40000,
            "gemini" => 45000,
            _ => 30000,
        };

        // Adjust based on temperature (higher temp = slower)
        let temp_adjustment = 1.0 - (self.temperature * 0.2);
        (base_rate as f32 * temp_adjustment) as u32
    }
}

