//! DeepSeek client with clean completion builder integration
//!
//! Provides factory methods that return clean completion builders with ModelInfo defaults:
//! ```
//! let client = DeepSeekClient::new(api_key).await?;
//! client.completion_model("deepseek-chat")
//!     .system_prompt("You are helpful")
//!     .temperature(0.8)
//!     .prompt("Hello world")
//! ```

use crate::completion_provider::{CompletionProvider, CompletionError};
use super::completion::DeepSeekCompletionBuilder;
use fluent_ai_http3::{HttpClient, HttpConfig};

/// DeepSeek client providing clean completion builder factory methods
#[derive(Clone)]
pub struct DeepSeekClient {
    api_key: String,
}

impl DeepSeekClient {
    /// Create new DeepSeek client with API key
    pub fn new(api_key: String) -> Result<Self, CompletionError> {
        if api_key.is_empty() {
            return Err(CompletionError::AuthError);
        }
        
        Ok(Self { api_key })
    }
    
    /// Create completion builder for specific model with ModelInfo defaults loaded
    pub fn completion_model(&self, model_name: &'static str) -> Result<DeepSeekCompletionBuilder, CompletionError> {
        DeepSeekCompletionBuilder::new(self.api_key.clone(), model_name)
    }
    
    /// Get API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

/// DeepSeek provider for enumeration and discovery
pub struct DeepSeekProvider;

impl DeepSeekProvider {
    /// Create new DeepSeek provider instance
    pub fn new() -> Self {
        Self
    }
    
    /// Get provider name
    pub const fn name() -> &'static str {
        "deepseek"
    }
    
    /// Get available models (compile-time constant)
    pub const fn models() -> &'static [&'static str] {
        &[
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-v3",
            "deepseek-r1",
        ]
    }
}

impl Default for DeepSeekProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_client_creation() {
        let client = DeepSeekClient::new("test-key".to_string());
        assert!(client.is_ok());
        
        let client = client.unwrap();
        assert_eq!(client.api_key(), "test-key");
    }
    
    #[test]
    fn test_client_creation_empty_key() {
        let client = DeepSeekClient::new("".to_string());
        assert!(matches!(client, Err(CompletionError::AuthError)));
    }
    
    #[test]
    fn test_completion_model_factory() {
        let client = DeepSeekClient::new("test-key".to_string()).unwrap();
        let builder = client.completion_model("deepseek-chat");
        assert!(builder.is_ok());
    }
    
    #[test]
    fn test_provider() {
        let provider = DeepSeekProvider::new();
        assert_eq!(DeepSeekProvider::name(), "deepseek");
        assert!(!DeepSeekProvider::models().is_empty());
    }
}