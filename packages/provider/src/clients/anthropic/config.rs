//! Zero-allocation Anthropic client configuration
//!
//! This module provides blazing-fast configuration management with zero allocations
//! after construction and no locking requirements.

use super::error::{AnthropicError, AnthropicResult};
use std::borrow::Cow;

/// Default Anthropic API base URL
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Default request timeout in seconds
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 300;

/// Default maximum retry attempts
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default API version
pub const DEFAULT_API_VERSION: &str = "2023-06-01";

/// Default user agent
pub const DEFAULT_USER_AGENT: &str = "fluent-ai-provider/0.1.0";

/// Immutable Anthropic client configuration
///
/// This configuration is validated during construction and remains immutable
/// throughout its lifetime, ensuring thread safety without locking.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key for authentication
    pub(crate) api_key: String,
    /// Base URL for the Anthropic API
    pub(crate) base_url: String,
    /// Request timeout in seconds
    pub(crate) timeout_seconds: u64,
    /// Maximum retry attempts
    pub(crate) max_retries: u32,
    /// API version header
    pub(crate) api_version: String,
    /// User agent string
    pub(crate) user_agent: String,
}

impl AnthropicConfig {
    /// Create a new configuration with the given API key
    #[inline]
    pub fn new(api_key: impl Into<String>) -> AnthropicResult<Self> {
        let api_key = api_key.into();
        Self::validate_api_key(&api_key)?;
        
        Ok(Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
            max_retries: DEFAULT_MAX_RETRIES,
            api_version: DEFAULT_API_VERSION.to_string(),
            user_agent: DEFAULT_USER_AGENT.to_string(),
        })
    }

    /// Create a configuration builder
    #[inline]
    pub fn builder(api_key: impl Into<String>) -> AnthropicConfigBuilder {
        AnthropicConfigBuilder::new(api_key)
    }

    /// Get the API key
    #[inline]
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Get the base URL
    #[inline]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get the timeout in seconds
    #[inline]
    pub fn timeout_seconds(&self) -> u64 {
        self.timeout_seconds
    }

    /// Get the maximum retry attempts
    #[inline]
    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Get the API version
    #[inline]
    pub fn api_version(&self) -> &str {
        &self.api_version
    }

    /// Get the user agent
    #[inline]
    pub fn user_agent(&self) -> &str {
        &self.user_agent
    }

    /// Validate API key format
    #[inline]
    fn validate_api_key(api_key: &str) -> AnthropicResult<()> {
        if api_key.is_empty() {
            return Err(AnthropicError::ConfigurationError {
                message: "API key cannot be empty".to_string(),
            });
        }
        
        if api_key.len() < 10 {
            return Err(AnthropicError::ConfigurationError {
                message: "API key appears to be too short".to_string(),
            });
        }
        
        if !api_key.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err(AnthropicError::ConfigurationError {
                message: "API key contains invalid characters".to_string(),
            });
        }
        
        Ok(())
    }

    /// Validate base URL format
    #[inline]
    fn validate_base_url(url: &str) -> AnthropicResult<()> {
        if url.is_empty() {
            return Err(AnthropicError::ConfigurationError {
                message: "Base URL cannot be empty".to_string(),
            });
        }
        
        if !url.starts_with("https://") && !url.starts_with("http://") {
            return Err(AnthropicError::ConfigurationError {
                message: "Base URL must start with http:// or https://".to_string(),
            });
        }
        
        Ok(())
    }

    /// Validate timeout value
    #[inline]
    fn validate_timeout(timeout: u64) -> AnthropicResult<()> {
        if timeout == 0 {
            return Err(AnthropicError::ConfigurationError {
                message: "Timeout cannot be zero".to_string(),
            });
        }
        
        if timeout > 3600 {
            return Err(AnthropicError::ConfigurationError {
                message: "Timeout cannot exceed 1 hour (3600 seconds)".to_string(),
            });
        }
        
        Ok(())
    }

    /// Validate retry count
    #[inline]
    fn validate_retries(retries: u32) -> AnthropicResult<()> {
        if retries > 10 {
            return Err(AnthropicError::ConfigurationError {
                message: "Maximum retries cannot exceed 10".to_string(),
            });
        }
        
        Ok(())
    }
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        // This will panic if the default API key is invalid, but that's intentional
        // as it indicates a programming error in the defaults
        Self::new("default-api-key-placeholder").unwrap_or_else(|_| {
            // Fallback construction for safety, though this should never happen
            Self {
                api_key: "default-api-key-placeholder".to_string(),
                base_url: DEFAULT_BASE_URL.to_string(),
                timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
                max_retries: DEFAULT_MAX_RETRIES,
                api_version: DEFAULT_API_VERSION.to_string(),
                user_agent: DEFAULT_USER_AGENT.to_string(),
            }
        })
    }
}

/// Builder for creating Anthropic configuration with validation
#[derive(Debug)]
pub struct AnthropicConfigBuilder {
    api_key: String,
    base_url: Option<String>,
    timeout_seconds: Option<u64>,
    max_retries: Option<u32>,
    api_version: Option<String>,
    user_agent: Option<String>,
}

impl AnthropicConfigBuilder {
    /// Create a new builder with the required API key
    #[inline]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            timeout_seconds: None,
            max_retries: None,
            api_version: None,
            user_agent: None,
        }
    }

    /// Set the base URL
    #[inline]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the timeout in seconds
    #[inline]
    pub fn timeout_seconds(mut self, timeout: u64) -> Self {
        self.timeout_seconds = Some(timeout);
        self
    }

    /// Set the maximum retry attempts
    #[inline]
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Set the API version
    #[inline]
    pub fn api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Set the user agent
    #[inline]
    pub fn user_agent(mut self, agent: impl Into<String>) -> Self {
        self.user_agent = Some(agent.into());
        self
    }

    /// Build the configuration with validation
    pub fn build(self) -> AnthropicResult<AnthropicConfig> {
        // Validate API key
        AnthropicConfig::validate_api_key(&self.api_key)?;
        
        // Handle base URL
        let base_url = match self.base_url {
            Some(url) => {
                AnthropicConfig::validate_base_url(&url)?;
                url
            }
            None => DEFAULT_BASE_URL.to_string(),
        };
        
        // Handle timeout
        let timeout_seconds = match self.timeout_seconds {
            Some(timeout) => {
                AnthropicConfig::validate_timeout(timeout)?;
                timeout
            }
            None => DEFAULT_TIMEOUT_SECONDS,
        };
        
        // Handle retries
        let max_retries = match self.max_retries {
            Some(retries) => {
                AnthropicConfig::validate_retries(retries)?;
                retries
            }
            None => DEFAULT_MAX_RETRIES,
        };
        
        // Handle API version
        let api_version = self.api_version.unwrap_or_else(|| DEFAULT_API_VERSION.to_string());
        
        // Handle user agent
        let user_agent = self.user_agent.unwrap_or_else(|| DEFAULT_USER_AGENT.to_string());
        
        Ok(AnthropicConfig {
            api_key: self.api_key,
            base_url,
            timeout_seconds,
            max_retries,
            api_version,
            user_agent,
        })
    }
}

/// Create configuration from environment variables
pub fn from_env() -> AnthropicResult<AnthropicConfig> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| AnthropicError::ConfigurationError {
            message: "ANTHROPIC_API_KEY environment variable not set".to_string(),
        })?;
    
    let mut builder = AnthropicConfig::builder(api_key);
    
    // Optional environment variables
    if let Ok(base_url) = std::env::var("ANTHROPIC_BASE_URL") {
        builder = builder.base_url(base_url);
    }
    
    if let Ok(timeout) = std::env::var("ANTHROPIC_TIMEOUT_SECONDS") {
        let timeout: u64 = timeout.parse()
            .map_err(|_| AnthropicError::ConfigurationError {
                message: "Invalid ANTHROPIC_TIMEOUT_SECONDS value".to_string(),
            })?;
        builder = builder.timeout_seconds(timeout);
    }
    
    if let Ok(retries) = std::env::var("ANTHROPIC_MAX_RETRIES") {
        let retries: u32 = retries.parse()
            .map_err(|_| AnthropicError::ConfigurationError {
                message: "Invalid ANTHROPIC_MAX_RETRIES value".to_string(),
            })?;
        builder = builder.max_retries(retries);
    }
    
    if let Ok(version) = std::env::var("ANTHROPIC_API_VERSION") {
        builder = builder.api_version(version);
    }
    
    if let Ok(agent) = std::env::var("ANTHROPIC_USER_AGENT") {
        builder = builder.user_agent(agent);
    }
    
    builder.build()
}

/// Convenience function to create a configuration with just an API key
#[inline]
pub fn with_api_key(api_key: impl Into<String>) -> AnthropicResult<AnthropicConfig> {
    AnthropicConfig::new(api_key)
}