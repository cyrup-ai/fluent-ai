//! AI21 Labs client with zero-allocation OpenAI-compatible architecture
//!
//! Provides high-performance AI21 integration supporting Jamba model family:
//! - Jamba-1.5-Large: Production-ready hybrid Mamba-Transformer model
//! - Jamba-1.5-Mini: Efficient model optimized for speed and cost
//! - J2-Ultra: Legacy ultra-capable model for complex tasks
//! - J2-Mid: Legacy mid-tier model for balanced performance
//!
//! Features:
//! - Zero allocation OpenAI-compatible architecture with shared connection pool
//! - Lock-free concurrent design with atomic counters
//! - Streaming chat completions with incremental JSON parsing
//! - Bearer token authentication with hot-swappable API keys
//! - Circuit breaker fault tolerance with exponential backoff
//! - SIMD-optimized operations where applicable
//!
//! ```
//! use fluent_ai_provider::clients::ai21::{AI21Client, AI21CompletionBuilder};
//! 
//! let client = AI21Client::new("your-api-key".to_string())?;
//! 
//! // Chat completion
//! client.completion_model("jamba-1.5-large")?
//!     .system_prompt("You are a helpful assistant")
//!     .temperature(0.7)
//!     .prompt("Hello world");
//! ```

use crate::completion_provider::{CompletionProvider, CompletionError};
use crate::client::{CompletionClient, ProviderClient};

// Re-export all public types with zero allocation
pub use client::AI21Client;
pub use completion::AI21CompletionBuilder;
pub use streaming::AI21Stream;
pub use error::{AI21Error, Result};

// Internal modules
mod client;
mod completion;
mod error;
mod streaming;

/// AI21 Labs model constants using &'static str for zero allocation
pub mod models {
    /// Jamba-1.5-Large: Production-ready hybrid Mamba-Transformer model
    pub const JAMBA_1_5_LARGE: &str = "jamba-1.5-large";
    
    /// Jamba-1.5-Mini: Efficient model optimized for speed and cost
    pub const JAMBA_1_5_MINI: &str = "jamba-1.5-mini";
    
    /// J2-Ultra: Legacy ultra-capable model for complex tasks
    pub const J2_ULTRA: &str = "j2-ultra";
    
    /// J2-Mid: Legacy mid-tier model for balanced performance
    pub const J2_MID: &str = "j2-mid";
    
    /// All supported models for validation
    pub const ALL_MODELS: &[&str] = &[
        JAMBA_1_5_LARGE,
        JAMBA_1_5_MINI,
        J2_ULTRA,
        J2_MID,
    ];
    
    /// Jamba models (current generation)
    pub const JAMBA_MODELS: &[&str] = &[
        JAMBA_1_5_LARGE,
        JAMBA_1_5_MINI,
    ];
    
    /// J2 models (legacy generation)
    pub const J2_MODELS: &[&str] = &[
        J2_ULTRA,
        J2_MID,
    ];
}

/// Configuration constants for AI21 client
pub mod config {
    use std::time::Duration;
    
    /// Default API endpoint for AI21 completions
    pub const DEFAULT_ENDPOINT: &str = "https://api.ai21.com/studio/v1/chat/completions";
    
    /// Default timeout for API requests
    pub const DEFAULT_TIMEOUT_SECS: u64 = 30;
    
    /// Maximum retry attempts for transient failures
    pub const MAX_RETRIES: u32 = 3;
    
    /// Default maximum tokens for completions
    pub const DEFAULT_MAX_TOKENS: u32 = 2048;
    
    /// Maximum context length for Jamba models
    pub const JAMBA_MAX_CONTEXT: u32 = 256_000;
    
    /// Maximum context length for J2 models
    pub const J2_MAX_CONTEXT: u32 = 8_192;
    
    /// Default temperature for completions
    pub const DEFAULT_TEMPERATURE: f32 = 0.7;
    
    /// API key validation regex pattern
    pub const API_KEY_PATTERN: &str = r"^[a-zA-Z0-9_-]{32,}$";
    
    /// Bearer token prefix for authentication
    pub const BEARER_PREFIX: &str = "Bearer ";
    
    /// Circuit breaker failure threshold
    pub const CIRCUIT_BREAKER_THRESHOLD: u32 = 5;
    
    /// Circuit breaker timeout duration
    pub const CIRCUIT_BREAKER_TIMEOUT: Duration = Duration::from_secs(60);
}

/// Model classification and validation functions with compile-time optimization
impl models {
    /// Check if model is a Jamba model (current generation)
    #[inline]
    pub const fn is_jamba_model(model: &str) -> bool {
        matches!(model, models::JAMBA_1_5_LARGE | models::JAMBA_1_5_MINI)
    }
    
    /// Check if model is a J2 model (legacy generation)
    #[inline]
    pub const fn is_j2_model(model: &str) -> bool {
        matches!(model, models::J2_ULTRA | models::J2_MID)
    }
    
    /// Check if model is supported by AI21
    #[inline]
    pub const fn is_supported_model(model: &str) -> bool {
        Self::is_jamba_model(model) || Self::is_j2_model(model)
    }
    
    /// Get maximum context length for a model
    #[inline]
    pub const fn context_length(model: &str) -> u32 {
        match model {
            models::JAMBA_1_5_LARGE | models::JAMBA_1_5_MINI => config::JAMBA_MAX_CONTEXT,
            models::J2_ULTRA | models::J2_MID => config::J2_MAX_CONTEXT,
            _ => 0,
        }
    }
    
    /// Get model generation (Jamba or J2)
    #[inline]
    pub const fn model_generation(model: &str) -> Option<&'static str> {
        if Self::is_jamba_model(model) {
            Some("jamba")
        } else if Self::is_j2_model(model) {
            Some("j2")
        } else {
            None
        }
    }
    
    /// Get model tier (large, mini, ultra, mid)
    #[inline]
    pub const fn model_tier(model: &str) -> Option<&'static str> {
        match model {
            models::JAMBA_1_5_LARGE => Some("large"),
            models::JAMBA_1_5_MINI => Some("mini"),
            models::J2_ULTRA => Some("ultra"),
            models::J2_MID => Some("mid"),
            _ => None,
        }
    }
    
    /// Check if model supports streaming
    #[inline]
    pub const fn supports_streaming(model: &str) -> bool {
        // All AI21 models support streaming
        Self::is_supported_model(model)
    }
    
    /// Check if model supports tools/function calling
    #[inline]
    pub const fn supports_tools(model: &str) -> bool {
        // Jamba models support tools, J2 models do not
        Self::is_jamba_model(model)
    }
    
    /// Get recommended temperature range for model
    #[inline]
    pub const fn temperature_range(model: &str) -> (f32, f32) {
        match model {
            models::JAMBA_1_5_LARGE | models::JAMBA_1_5_MINI => (0.0, 2.0),
            models::J2_ULTRA | models::J2_MID => (0.0, 1.0),
            _ => (0.0, 1.0),
        }
    }
}

/// Metadata constants for AI21 models
pub mod metadata {
    /// Typical latency for AI21 completions (milliseconds)
    pub const TYPICAL_COMPLETION_LATENCY_MS: u64 = 1500;
    
    /// Typical streaming chunk latency (milliseconds)
    pub const TYPICAL_STREAMING_LATENCY_MS: u64 = 50;
    
    /// User agent for AI21 API requests
    pub const USER_AGENT: &str = "fluent-ai-provider/0.1.0 (ai21)";
    
    /// Content type for JSON requests
    pub const CONTENT_TYPE: &str = "application/json";
    
    /// Accept header for JSON responses
    pub const ACCEPT: &str = "application/json";
}

/// Utility functions for AI21 client operations
pub mod utils {
    use super::*;
    
    /// Get user agent string for AI21 requests
    #[inline]
    pub const fn user_agent() -> &'static str {
        metadata::USER_AGENT
    }
    
    /// Validate API key format
    #[inline]
    pub fn validate_api_key(api_key: &str) -> Result<(), arrayvec::ArrayString<64>> {
        if api_key.is_empty() {
            return Err(arrayvec::ArrayString::from("API key cannot be empty").unwrap_or_default());
        }
        
        if api_key.len() < 32 {
            return Err(arrayvec::ArrayString::from("API key too short").unwrap_or_default());
        }
        
        if api_key.len() > 128 {
            return Err(arrayvec::ArrayString::from("API key too long").unwrap_or_default());
        }
        
        // Check for valid characters (alphanumeric, underscore, hyphen)
        if !api_key.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(arrayvec::ArrayString::from("API key contains invalid characters").unwrap_or_default());
        }
        
        Ok(())
    }
    
    /// Calculate optimal timeout for model
    #[inline]
    pub const fn optimal_timeout_ms(model: &str) -> u64 {
        match model {
            models::JAMBA_1_5_LARGE | models::J2_ULTRA => 45_000, // Large models need more time
            models::JAMBA_1_5_MINI | models::J2_MID => 30_000,   // Smaller models are faster
            _ => 30_000,
        }
    }
    
    /// Get model pricing tier for cost optimization
    #[inline]
    pub const fn pricing_tier(model: &str) -> &'static str {
        match model {
            models::JAMBA_1_5_LARGE | models::J2_ULTRA => "premium",
            models::JAMBA_1_5_MINI | models::J2_MID => "standard",
            _ => "unknown",
        }
    }
}

/// Endpoint configuration for AI21 API
pub mod endpoints {
    /// Chat completions endpoint
    pub const CHAT_COMPLETIONS: &str = "https://api.ai21.com/studio/v1/chat/completions";
    
    /// Model information endpoint
    pub const MODELS: &str = "https://api.ai21.com/studio/v1/models";
    
    /// Health check endpoint
    pub const HEALTH: &str = "https://api.ai21.com/studio/v1/health";
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_classification() {
        assert!(models::is_jamba_model(models::JAMBA_1_5_LARGE));
        assert!(models::is_jamba_model(models::JAMBA_1_5_MINI));
        assert!(!models::is_jamba_model(models::J2_ULTRA));
        assert!(!models::is_jamba_model(models::J2_MID));
        
        assert!(models::is_j2_model(models::J2_ULTRA));
        assert!(models::is_j2_model(models::J2_MID));
        assert!(!models::is_j2_model(models::JAMBA_1_5_LARGE));
        assert!(!models::is_j2_model(models::JAMBA_1_5_MINI));
    }
    
    #[test]
    fn test_model_validation() {
        assert!(models::is_supported_model(models::JAMBA_1_5_LARGE));
        assert!(models::is_supported_model(models::JAMBA_1_5_MINI));
        assert!(models::is_supported_model(models::J2_ULTRA));
        assert!(models::is_supported_model(models::J2_MID));
        assert!(!models::is_supported_model("invalid-model"));
    }
    
    #[test]
    fn test_context_length() {
        assert_eq!(models::context_length(models::JAMBA_1_5_LARGE), config::JAMBA_MAX_CONTEXT);
        assert_eq!(models::context_length(models::JAMBA_1_5_MINI), config::JAMBA_MAX_CONTEXT);
        assert_eq!(models::context_length(models::J2_ULTRA), config::J2_MAX_CONTEXT);
        assert_eq!(models::context_length(models::J2_MID), config::J2_MAX_CONTEXT);
        assert_eq!(models::context_length("invalid-model"), 0);
    }
    
    #[test]
    fn test_api_key_validation() {
        assert!(utils::validate_api_key("valid_api_key_123456789012345678901234567890").is_ok());
        assert!(utils::validate_api_key("").is_err());
        assert!(utils::validate_api_key("short").is_err());
        assert!(utils::validate_api_key("invalid@key").is_err());
    }
}