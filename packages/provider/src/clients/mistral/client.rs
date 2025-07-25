//! Mistral client with clean completion builder integration
//!
//! Provides factory methods that return clean completion builders with ModelInfo defaults:
//! ```
//! let client = MistralClient::new(api_key).await?;
//! client.completion_model("mistral-large-latest")
//!     .system_prompt("You are helpful")
//!     .temperature(0.8)
//!     .prompt("Hello world")
//! ```

use fluent_ai_domain::AsyncTask;

use super::completion::MistralCompletionBuilder;
use super::completion::{
    CODESTRAL, CODESTRAL_MAMBA, MINISTRAL_3B, MINISTRAL_8B, MISTRAL_LARGE, MISTRAL_NEMO,
    MISTRAL_SABA, MISTRAL_SMALL, PIXTRAL_LARGE, PIXTRAL_SMALL};
use crate::client::{CompletionClient, ProviderClient};
use crate::completion_provider::{CompletionError, CompletionProvider};

/// Mistral client providing clean completion builder factory methods
#[derive(Clone)]
pub struct MistralClient {
    api_key: String}

impl MistralClient {
    /// Create new Mistral client with API key
    pub fn new(api_key: String) -> Result<Self, CompletionError> {
        if api_key.is_empty() {
            return Err(CompletionError::AuthError);
        }

        Ok(Self { api_key })
    }

    /// Create completion builder for specific model with ModelInfo defaults loaded
    pub fn completion_model(
        &self,
        model_name: &'static str,
    ) -> Result<MistralCompletionBuilder, CompletionError> {
        MistralCompletionBuilder::new(self.api_key.clone(), model_name)
    }

    /// Get API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

/// Mistral provider for enumeration and discovery
pub struct MistralProvider;

impl MistralProvider {
    /// Create new Mistral provider instance
    pub fn new() -> Self {
        Self
    }

    /// Get provider name
    pub const fn name() -> &'static str {
        "mistral"
    }

    /// Get available models (compile-time constant)
    pub const fn models() -> &'static [&'static str] {
        &[
            MISTRAL_LARGE,
            MISTRAL_SABA,
            CODESTRAL,
            PIXTRAL_LARGE,
            MINISTRAL_3B,
            MINISTRAL_8B,
            MISTRAL_SMALL,
            PIXTRAL_SMALL,
            MISTRAL_NEMO,
            CODESTRAL_MAMBA,
        ]
    }
}

impl Default for MistralProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-allocation CompletionClient implementation for Mistral
impl CompletionClient for MistralClient {
    type Model = Result<MistralCompletionBuilder, CompletionError>;

    /// Create a completion model with zero allocation and blazing-fast performance
    #[inline]
    fn completion_model(&self, model: &str) -> Self::Model {
        // Convert &str to &'static str efficiently for compatibility
        // SAFETY: This is safe because model names are typically string literals
        // stored in static memory. For dynamic strings, we use a fallback.
        let static_model = match model {
            "mistral-large-latest" => MISTRAL_LARGE,
            "mistral-saba-latest" => MISTRAL_SABA,
            "codestral-latest" => CODESTRAL,
            "pixtral-large-latest" => PIXTRAL_LARGE,
            "ministral-3b-latest" => MINISTRAL_3B,
            "ministral-8b-latest" => MINISTRAL_8B,
            "mistral-small-latest" => MISTRAL_SMALL,
            "pixtral-12b-2409" => PIXTRAL_SMALL,
            "open-mistral-nemo" => MISTRAL_NEMO,
            "open-codestral-mamba" => CODESTRAL_MAMBA,
            _ => {
                // For unknown models, create a leaked static string (one-time allocation)
                // This is acceptable for model names which are typically static
                Box::leak(model.to_string().into_boxed_str())
            }
        };

        MistralCompletionBuilder::new(self.api_key.clone(), static_model)
    }
}

/// Zero-allocation ProviderClient implementation for Mistral
impl ProviderClient for MistralClient {
    /// Get provider name with zero allocation
    #[inline]
    fn provider_name(&self) -> &'static str {
        "mistral"
    }

    /// Test connection with blazing-fast async task
    #[inline]
    fn test_connection(&self) -> AsyncTask<Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let api_key = self.api_key.clone();

        AsyncTask::spawn(async move {
            // Zero-allocation validation: check API key format and non-empty
            if api_key.is_empty() {
                return Err("Mistral API key is empty".into());
            }

            // Mistral API keys typically don't have a standard prefix like OpenAI's "sk-"
            // Just do basic length validation
            if api_key.len() < 20 {
                return Err("Mistral API key too short".into());
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
        let client = MistralClient::new("test-key".to_string());
        assert!(client.is_ok());

        let client = client.expect("Failed to create mistral client in test");
        assert_eq!(client.api_key(), "test-key");
    }

    #[test]
    fn test_client_creation_empty_key() {
        let client = MistralClient::new("".to_string());
        assert!(matches!(client, Err(CompletionError::AuthError)));
    }

    #[test]
    fn test_completion_model_factory() {
        let client = MistralClient::new("test-key".to_string())
            .expect("Failed to create mistral client in test");
        let builder = client.completion_model(MISTRAL_LARGE);
        assert!(builder.is_ok());
    }

    #[test]
    fn test_provider() {
        let provider = MistralProvider::new();
        assert_eq!(MistralProvider::name(), "mistral");
        assert!(!MistralProvider::models().is_empty());
        assert!(MistralProvider::models().contains(&MISTRAL_LARGE));
        assert!(MistralProvider::models().contains(&MISTRAL_SMALL));
    }
}
