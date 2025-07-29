//! LLM configuration for memory system integration
//!
//! Zero-allocation configuration for LLM providers with blazing-fast performance

use serde::{Deserialize, Serialize};

/// LLM provider enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LLMProvider {
    /// OpenAI GPT models
    OpenAI,
    /// Anthropic Claude models  
    Anthropic,
    /// Google Gemini models
    Gemini,
    /// Local/self-hosted models
    Local,
    /// Other providers
    Other}

/// LLM configuration for memory system operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Provider to use
    pub provider: LLMProvider,
    /// Model name/identifier
    pub model: String,
    /// API endpoint (optional for non-standard endpoints)
    pub endpoint: Option<String>,
    /// API key (optional for local models)
    pub api_key: Option<String>,
    /// Maximum tokens for responses
    pub max_tokens: u32,
    /// Temperature for generation
    pub temperature: f64,
    /// Enable streaming responses
    pub streaming: bool,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Maximum retries on failure
    pub max_retries: u32}

impl LLMConfig {
    /// Create new LLM configuration
    pub fn new(provider: LLMProvider, model: impl Into<String>) -> Result<Self, LLMConfigError> {
        let model = model.into();

        if model.is_empty() {
            return Err(LLMConfigError::InvalidModel(
                "Model name cannot be empty".to_string(),
            ));
        }

        Ok(Self {
            provider,
            model,
            endpoint: None,
            api_key: None,
            max_tokens: 4096,
            temperature: 0.7,
            streaming: false,
            timeout_ms: 30000,
            max_retries: 3})
    }

    /// Enable streaming responses
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Set API endpoint
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Set API key
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set maximum tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set timeout
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set max retries
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), LLMConfigError> {
        if self.model.is_empty() {
            return Err(LLMConfigError::InvalidModel(
                "Model name cannot be empty".to_string(),
            ));
        }

        if self.max_tokens == 0 {
            return Err(LLMConfigError::InvalidParameter(
                "max_tokens must be greater than 0".to_string(),
            ));
        }

        if !(0.0..=2.0).contains(&self.temperature) {
            return Err(LLMConfigError::InvalidParameter(
                "temperature must be between 0.0 and 2.0".to_string(),
            ));
        }

        if self.timeout_ms == 0 {
            return Err(LLMConfigError::InvalidParameter(
                "timeout_ms must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: LLMProvider::OpenAI,
            model: "gpt-4".to_string(),
            endpoint: None,
            api_key: None,
            max_tokens: 4096,
            temperature: 0.7,
            streaming: false,
            timeout_ms: 30000,
            max_retries: 3}
    }
}

/// LLM configuration error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum LLMConfigError {
    /// Invalid model name or configuration
    #[error("Invalid model: {0}")]
    InvalidModel(String),

    /// Invalid parameter value or format
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Configuration validation failed: {0}")]
    ValidationError(String)}

impl LLMProvider {
    /// Get default endpoint for provider
    pub fn default_endpoint(&self) -> Option<&'static str> {
        match self {
            Self::OpenAI => Some("https://api.openai.com/v1"),
            Self::Anthropic => Some("https://api.anthropic.com/v1"),
            Self::Gemini => Some("https://generativelanguage.googleapis.com/v1"),
            Self::Local | Self::Other => None}
    }

    /// Check if provider requires API key
    pub fn requires_api_key(&self) -> bool {
        match self {
            Self::Local => false,
            _ => true}
    }

    /// Get common models for provider
    pub fn common_models(&self) -> &'static [&'static str] {
        match self {
            Self::OpenAI => &["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            Self::Anthropic => &["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            Self::Gemini => &["gemini-1.5-pro", "gemini-1.5-flash"],
            Self::Local | Self::Other => &[]}
    }
}
