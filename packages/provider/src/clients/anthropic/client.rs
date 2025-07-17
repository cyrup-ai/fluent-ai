//! Anthropic client with clean completion builder integration
//!
//! Provides factory methods that return clean completion builders with ModelInfo defaults:
//! ```
//! let client = AnthropicClient::new(api_key).await?;
//! client.completion_model("claude-3-5-sonnet-20241022")
//!     .system_prompt("You are helpful")
//!     .temperature(0.8)
//!     .prompt("Hello world")
//! ```

use crate::completion_provider::{CompletionProvider, CompletionError};
use super::completion::AnthropicCompletionBuilder;
use fluent_ai_http3::{HttpClient, HttpConfig};

/// Anthropic client providing clean completion builder factory methods
#[derive(Clone)]
pub struct AnthropicClient {
    api_key: String,
}

impl AnthropicClient {
    /// Create new Anthropic client with API key
    pub fn new(api_key: String) -> Result<Self, CompletionError> {
        if api_key.is_empty() {
            return Err(CompletionError::AuthError);
        }
        
        Ok(Self { api_key })
    }
    
    /// Create completion builder for specific model with ModelInfo defaults loaded
    pub fn completion_model(&self, model_name: &'static str) -> Result<AnthropicCompletionBuilder, CompletionError> {
        AnthropicCompletionBuilder::new(self.api_key.clone(), model_name)
    }
    
    /// Get API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

/// Anthropic provider for enumeration and discovery
pub struct AnthropicProvider;

impl AnthropicProvider {
    /// Create new Anthropic provider instance
    pub fn new() -> Self {
        Self
    }
    
    /// Get provider name
    pub const fn name() -> &'static str {
        "anthropic"
    }
    
    /// Get available models (compile-time constant)
    pub const fn models() -> &'static [&'static str] {
        &[
            // Claude 4 models (newest and most powerful)
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            
            // Claude 3.7 models
            "claude-3-7-sonnet-20250219",
            
            // Claude 3.5 models 
            "claude-3-5-sonnet-20241022",    // v2 (latest)
            "claude-3-5-sonnet-20240620",    // v1 (original)
            "claude-3-5-haiku-20241022",
            
            // Claude 3 models
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_client_creation() {
        let client = AnthropicClient::new("test-key".to_string());
        assert!(client.is_ok());
        
        let client = client.unwrap();
        assert_eq!(client.api_key(), "test-key");
    }
    
    #[test]
    fn test_client_creation_empty_key() {
        let client = AnthropicClient::new("".to_string());
        assert!(matches!(client, Err(CompletionError::AuthError)));
    }
    
    #[test]
    fn test_completion_model_factory() {
        let client = AnthropicClient::new("test-key".to_string()).unwrap();
        let builder = client.completion_model("claude-3-5-sonnet-20241022");
        assert!(builder.is_ok());
    }
    
    #[test]
    fn test_provider() {
        let provider = AnthropicProvider::new();
        assert_eq!(AnthropicProvider::name(), "anthropic");
        assert!(!AnthropicProvider::models().is_empty());
    }
}