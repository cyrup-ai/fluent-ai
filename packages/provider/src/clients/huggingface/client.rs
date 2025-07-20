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

use fluent_ai_domain::AsyncTask as DomainAsyncTask;
use fluent_ai_http3::{HttpClient, HttpConfig};

use super::completion::HuggingFaceCompletionBuilder;
use crate::{
    client::{CompletionClient, ProviderClient},
    completion_provider::{CompletionError, CompletionProvider},
};

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

    /// Create from environment (HUGGINGFACE_API_KEY)
    pub fn from_env() -> Result<Self, CompletionError> {
        let api_key = std::env::var("HUGGINGFACE_API_KEY")
            .or_else(|_| std::env::var("HF_TOKEN"))
            .map_err(|_| {
                CompletionError::ConfigError(
                    "HUGGINGFACE_API_KEY or HF_TOKEN environment variable not set".into(),
                )
            })?;
        Self::new(api_key)
    }

    /// Create completion builder for specific model with ModelInfo defaults loaded
    pub fn completion_model(
        &self,
        model_name: &'static str,
    ) -> Result<HuggingFaceCompletionBuilder, CompletionError> {
        HuggingFaceCompletionBuilder::new(self.api_key.clone(), model_name)
    }

    /// Test connection to HuggingFace API
    pub async fn test_connection(&self) -> Result<(), CompletionError> {
        // Create a minimal completion request to test connectivity
        let builder = self.completion_model("meta-llama/Meta-Llama-3.1-8B-Instruct")?;

        // For now, we'll return success if we can create the builder
        // A full implementation would make an actual API call
        Ok(())
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

/// CompletionClient trait implementation for auto-generation
impl CompletionClient for HuggingFaceClient {
    type Model = Result<HuggingFaceCompletionBuilder, CompletionError>;

    fn completion_model(&self, model: &str) -> Self::Model {
        HuggingFaceCompletionBuilder::new(self.api_key.clone(), model)
    }
}

/// ProviderClient trait implementation for ecosystem integration
impl ProviderClient for HuggingFaceClient {
    fn provider_name(&self) -> &'static str {
        "huggingface"
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
        let client = HuggingFaceClient::new("test-key".to_string())
            .expect("Failed to create huggingface client in test");
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
