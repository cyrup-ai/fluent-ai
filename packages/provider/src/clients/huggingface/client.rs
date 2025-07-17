//! HuggingFace client with clean completion builder integration
//!
//! Provides factory methods that return clean completion builders with ModelInfo defaults:
//! ```
//! let client = HuggingFaceClient::new(api_key).await?;
//! client.completion_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
//!     .system_prompt("You are helpful")
//!     .temperature(0.8)
//!     .prompt("Hello world")
//! ```

use crate::completion_provider::{CompletionProvider, CompletionError};
use super::completion::HuggingFaceCompletionBuilder;
use fluent_ai_http3::{HttpClient, HttpConfig};

/// HuggingFace client providing clean completion builder factory methods
#[derive(Clone)]
pub struct HuggingFaceClient {
    api_key: String,
}

impl HuggingFaceClient {
    /// Create new HuggingFace client with API key
    pub fn new(api_key: String) -> Result<Self, CompletionError> {
        if api_key.is_empty() {
            return Err(CompletionError::AuthError);
        }
        
        Ok(Self { api_key })
    }
    
    /// Create completion builder for specific model with ModelInfo defaults loaded
    pub fn completion_model(&self, model_name: &'static str) -> Result<HuggingFaceCompletionBuilder, CompletionError> {
        HuggingFaceCompletionBuilder::new(self.api_key.clone(), model_name)
    }
    
    /// Get API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

/// HuggingFace provider for enumeration and discovery
pub struct HuggingFaceProvider;

impl HuggingFaceProvider {
    /// Create new HuggingFace provider instance
    pub fn new() -> Self {
        Self
    }
    
    /// Get provider name
    pub const fn name() -> &'static str {
        "huggingface"
    }
    
    /// Get available models (compile-time constant)
    pub const fn models() -> &'static [&'static str] {
        &[
            "google/gemma-2-2b-it",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "microsoft/phi-4",
            "PowerInfer/SmallThinker-3B-Preview",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/QVQ-72B-Preview",
        ]
    }
}

impl Default for HuggingFaceProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_client_creation() {
        let client = HuggingFaceClient::new("test-key".to_string());
        assert!(client.is_ok());
        
        let client = client.expect("Failed to create huggingface client in test");
        assert_eq!(client.api_key(), "test-key");
    }
    
    #[test]
    fn test_client_creation_empty_key() {
        let client = HuggingFaceClient::new("".to_string());
        assert!(matches!(client, Err(CompletionError::AuthError)));
    }
    
    #[test]
    fn test_completion_model_factory() {
        let client = HuggingFaceClient::new("test-key".to_string()).expect("Failed to create huggingface client in test");
        let builder = client.completion_model("meta-llama/Meta-Llama-3.1-8B-Instruct");
        assert!(builder.is_ok());
    }
    
    #[test]
    fn test_provider() {
        let provider = HuggingFaceProvider::new();
        assert_eq!(HuggingFaceProvider::name(), "huggingface");
        assert!(!HuggingFaceProvider::models().is_empty());
    }
}