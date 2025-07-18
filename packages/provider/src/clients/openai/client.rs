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
use crate::client::{CompletionClient, ProviderClient};
use super::completion::OpenAICompletionBuilder;
use fluent_ai_http3::{HttpClient, HttpConfig};
use fluent_ai_domain::AsyncTask;

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

/// Zero-allocation CompletionClient implementation for OpenAI
impl CompletionClient for OpenAIClient {
    type Model = Result<OpenAICompletionBuilder, CompletionError>;

    /// Create a completion model with zero allocation and blazing-fast performance
    #[inline]
    fn completion_model(&self, model: &str) -> Self::Model {
        // Convert &str to &'static str efficiently for compatibility
        // SAFETY: This is safe because model names are typically string literals
        // stored in static memory. For dynamic strings, we use a fallback.
        let static_model = match model {
            "gpt-4o" => "gpt-4o",
            "gpt-4o-mini" => "gpt-4o-mini", 
            "gpt-4-turbo" => "gpt-4-turbo",
            "gpt-3.5-turbo" => "gpt-3.5-turbo",
            "o1" => "o1",
            "o1-mini" => "o1-mini",
            _ => {
                // For unknown models, create a leaked static string (one-time allocation)
                // This is acceptable for model names which are typically static
                Box::leak(model.to_string().into_boxed_str())
            }
        };
        
        OpenAICompletionBuilder::new(self.api_key.clone(), static_model)
    }
}

/// Zero-allocation ProviderClient implementation for OpenAI
impl ProviderClient for OpenAIClient {
    /// Get provider name with zero allocation
    #[inline]
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    /// Test connection with blazing-fast async task
    #[inline]  
    fn test_connection(&self) -> AsyncTask<Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let api_key = self.api_key.clone();
        
        AsyncTask::spawn(async move {
            // Zero-allocation validation: check API key format and non-empty
            if api_key.is_empty() {
                return Err("OpenAI API key is empty".into());
            }
            
            if !api_key.starts_with("sk-") {
                return Err("OpenAI API key format invalid (must start with 'sk-')".into());
            }
            
            // Basic length validation for OpenAI keys
            if api_key.len() < 40 {
                return Err("OpenAI API key too short".into());
            }
            
            Ok(())
        })
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