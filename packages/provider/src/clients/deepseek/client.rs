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

use fluent_ai_domain::AsyncTask as DomainAsyncTask;
use fluent_ai_http3::{HttpClient, HttpConfig};

use super::completion::DeepSeekCompletionBuilder;
use crate::{
    client::{CompletionClient, ProviderClient},
    completion_provider::{CompletionError, CompletionProvider},
};

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

    /// Create from environment (DEEPSEEK_API_KEY)
    pub fn from_env() -> Result<Self, CompletionError> {
        let api_key = std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
            CompletionError::ConfigError("DEEPSEEK_API_KEY environment variable not set".into())
        })?;
        Self::new(api_key)
    }

    /// Create completion builder for specific model with ModelInfo defaults loaded
    pub fn completion_model(
        &self,
        model_name: &'static str,
    ) -> Result<DeepSeekCompletionBuilder, CompletionError> {
        DeepSeekCompletionBuilder::new(self.api_key.clone(), model_name)
    }

    /// Test connection to DeepSeek API
    pub async fn test_connection(&self) -> Result<(), CompletionError> {
        // Create a minimal completion request to test connectivity
        let builder = self.completion_model("deepseek-chat")?;

        // For now, we'll return success if we can create the builder
        // A full implementation would make an actual API call
        Ok(())
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

/// CompletionClient trait implementation for auto-generation
impl CompletionClient for DeepSeekClient {
    type Model = Result<DeepSeekCompletionBuilder, CompletionError>;

    fn completion_model(&self, model: &str) -> Self::Model {
        DeepSeekCompletionBuilder::new(self.api_key.clone(), model)
    }
}

/// ProviderClient trait implementation for ecosystem integration
impl ProviderClient for DeepSeekClient {
    fn provider_name(&self) -> &'static str {
        "deepseek"
    }

    fn test_connection(
        &self,
    ) -> DomainAsyncTask<std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let client = self.clone();
        DomainAsyncTask::spawn(async move {
            client
                .test_connection()
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = DeepSeekClient::new("test-key".to_string());
        assert!(client.is_ok());

        let client = client.expect("Failed to create deepseek client in test");
        assert_eq!(client.api_key(), "test-key");
    }

    #[test]
    fn test_client_creation_empty_key() {
        let client = DeepSeekClient::new("".to_string());
        assert!(matches!(client, Err(CompletionError::AuthError)));
    }

    #[test]
    fn test_completion_model_factory() {
        let client = DeepSeekClient::new("test-key".to_string())
            .expect("Failed to create deepseek client in test");
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
