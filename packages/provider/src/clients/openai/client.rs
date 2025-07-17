//! OpenAI client with clean completion builder integration
//!
//! Provides factory methods that return clean completion builders with ModelInfo defaults:
//! ```
//! let client = OpenAIClient::new(api_key).await?;
//! client.completion_model("gpt-4o")
//!     .system_prompt("You are helpful")
//!     .temperature(0.8)
//!     .prompt("Hello world")
//! ```

use crate::completion_provider::{CompletionProvider, CompletionError};
use super::completion::OpenAICompletionBuilder;
use fluent_ai_http3::{HttpClient, HttpConfig};

/// OpenAI client providing clean completion builder factory methods
#[derive(Clone)]
pub struct OpenAIClient {
    api_key: String,
}

impl OpenAIClient {
    /// Create new OpenAI client with API key
    pub fn new(api_key: String) -> Result<Self, CompletionError> {
        if api_key.is_empty() {
            return Err(CompletionError::AuthError);
        }
        
        Ok(Self { api_key })
    }
    
    /// Create completion builder for specific model with ModelInfo defaults loaded
    pub fn completion_model(&self, model_name: &'static str) -> Result<OpenAICompletionBuilder, CompletionError> {
        OpenAICompletionBuilder::new(self.api_key.clone(), model_name)
    }
    
    /// Get API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

/// OpenAI provider for enumeration and discovery
pub struct OpenAIProvider;

impl OpenAIProvider {
    /// Create new OpenAI provider instance
    pub fn new() -> Self {
        Self
    }
    
    /// Get provider name
    pub const fn name() -> &'static str {
        "openai"
    }
    
    /// Get available models (compile-time constant)
    pub const fn models() -> &'static [&'static str] {
        &[
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]
    }
}

impl Default for OpenAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_client_creation() {
        let client = OpenAIClient::new("test-key".to_string());
        assert!(client.is_ok());
        
        let client = client.expect("Failed to create openai client in test");
        assert_eq!(client.api_key(), "test-key");
    }
    
    #[test]
    fn test_client_creation_empty_key() {
        let client = OpenAIClient::new("".to_string());
        assert!(matches!(client, Err(CompletionError::AuthError)));
    }
    
    #[test]
    fn test_completion_model_factory() {
        let client = OpenAIClient::new("test-key".to_string()).expect("Failed to create openai client in test");
        let builder = client.completion_model("gpt-4o");
        assert!(builder.is_ok());
    }
    
    #[test]
    fn test_provider() {
        let provider = OpenAIProvider::new();
        assert_eq!(OpenAIProvider::name(), "openai");
        assert!(!OpenAIProvider::models().is_empty());
    }
}