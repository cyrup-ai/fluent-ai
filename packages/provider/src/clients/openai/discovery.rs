//! OpenAI model discovery and registration

use std::sync::Arc;

use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, instrument, trace, warn};

use super::{client::OpenAIClient, error::OpenAIError};
use crate::discovery::{DiscoveryError, DiscoveryResult, ProviderModelDiscovery};
use model_info::{discovery::Provider, ModelInfo, ModelInfoBuilder};
use fluent_ai_domain::model::{
    error::ModelError,
    registry::{ModelRegistry, RegisteredModel},
    traits::Model};

/// OpenAI model discovery implementation
#[derive(Debug, Clone)]
pub struct OpenAIDiscovery {
    client: Arc<OpenAIClient>,
    supported_models: &'static [&'static str]}

impl OpenAIDiscovery {
    /// Create a new OpenAI model discovery instance
    pub fn new(client: OpenAIClient) -> Self {
        static SUPPORTED_MODELS: &[&str] = &[
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-3.5-turbo-instruct",
        ];

        Self {
            client: Arc::new(client),
            supported_models: SUPPORTED_MODELS}
    }

    /// Get model information for a specific model using model-info package
    #[instrument(skip(self))]
    fn get_model_info(&self, model_name: &'static str) -> Option<ModelInfo> {
        // Create model info using the new model-info package architecture
        ModelInfoBuilder::new()
            .provider_name("openai")
            .name(model_name)
            .max_input_tokens(match model_name {
                "gpt-4o" | "gpt-4o-mini" => 128000,
                "gpt-4-turbo" => 128000,
                "gpt-4" => 8192,
                "gpt-3.5-turbo" => 16385,
                "gpt-3.5-turbo-instruct" => 4096,
                _ => 4096,
            })
            .max_output_tokens(match model_name {
                "gpt-4o" | "gpt-4o-mini" => 16384,
                "gpt-4-turbo" => 4096,
                "gpt-4" => 4096,
                "gpt-3.5-turbo" => 4096,
                "gpt-3.5-turbo-instruct" => 4096,
                _ => 4096,
            })
            .with_function_calling(match model_name {
                "gpt-4o" | "gpt-4o-mini" | "gpt-4-turbo" | "gpt-4" | "gpt-3.5-turbo" => true,
                _ => false,
            })
            .with_vision(match model_name {
                "gpt-4o" | "gpt-4o-mini" | "gpt-4-turbo" | "gpt-4" => true,
                _ => false,
            })
            .with_streaming(true) // All OpenAI models support streaming
            .build()
            .ok()
    }
}

impl ProviderModelDiscovery for OpenAIDiscovery {
    fn discover_models(&self) -> DiscoveryResult<Vec<String>> {
        Ok(self.supported_models.iter().map(|&s| s.to_string()).collect())
    }

    fn get_model_info(&self, model_name: &str) -> DiscoveryResult<ModelInfo> {
        // Implementation for getting model info
        if self.supported_models.contains(&model_name) {
            // Use the internal get_model_info method
            self.get_model_info(model_name)
                .ok_or_else(|| DiscoveryError::ModelNotFound(model_name.to_string()))
        } else {
            Err(DiscoveryError::ModelNotFound(model_name.to_string()))
        }
    }

    fn is_model_available(&self, model_name: &str) -> bool {
        self.supported_models.contains(&model_name)
    }
}

// Additional implementation for OpenAI-specific functionality
impl OpenAIDiscovery {
    pub fn provider_name(&self) -> &'static str {
        "openai"
    }

    pub async fn discover_and_register(&self) -> DiscoveryResult<()> {
        info!("Starting OpenAI model discovery");
        let registry = ModelRegistry::global();

        for &model_name in self.supported_models {
            if let Some(model_info) = self.get_model_info(model_name) {
                // Create a new model instance for each model
                let model = OpenAIModel {
                    info: model_info,
                    client: self.client.clone()};

                // Register the model
                if let Err(e) = registry.register("openai", model) {
                    error!("Failed to register OpenAI model {}: {}", model_name, e);
                    return Err(DiscoveryError::RegistrationFailed(e.to_string()));
                }

                debug!("Registered OpenAI model: {}", model_name);
            } else {
                warn!("Skipping unsupported OpenAI model: {}", model_name);
            }
        }

        info!("Completed OpenAI model discovery");
        Ok(())
    }

    fn supported_models(&self) -> &'static [&'static str] {
        self.supported_models
    }
}

/// OpenAI model implementation
#[derive(Debug, Clone)]
struct OpenAIModel {
    info: ModelInfo,
    client: Arc<OpenAIClient>}

impl Model for OpenAIModel {
    fn info(&self) -> &'static ModelInfo {
        // SAFETY: ModelInfo is 'static and we ensure it's never dropped
        // while references to it exist through the registry
        unsafe { std::mem::transmute(&self.info) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::traits::Model;

    #[tokio::test]
    async fn test_openai_discovery() {
        let client = OpenAIClient::new("test-api-key".to_string()).unwrap();
        let discovery = OpenAIDiscovery::new(client);

        // Test provider name
        assert_eq!(discovery.provider_name(), "openai");

        // Test supported models
        let supported_models = discovery.supported_models();
        assert!(!supported_models.is_empty());
        assert!(supported_models.contains(&&"gpt-4o"));

        // Test model info retrieval
        let model_info = discovery.get_model_info("gpt-4o").unwrap();
        assert_eq!(model_info.name(), "gpt-4o");
        assert_eq!(model_info.provider(), "openai");
        assert!(model_info.has_streaming()); // All models support streaming

        // Test model registration
        let result = discovery.discover_and_register().await;
        assert!(
            result.is_ok(),
            "Failed to discover and register models: {:?}",
            result.err()
        );

        // Verify models were registered
        let registry = ModelRegistry::global();
        for model_name in supported_models {
            assert!(
                registry
                    .get_model::<OpenAIModel>("openai", model_name)
                    .is_ok(),
                "Model {} was not registered",
                model_name
            );
        }
    }

    #[test]
    fn test_get_model_info() {
        let client = OpenAIClient::new("test-api-key".to_string()).unwrap();
        let discovery = OpenAIDiscovery::new(client);

        // Test with supported model
        let model_info = discovery.get_model_info("gpt-4o").unwrap();
        assert_eq!(model_info.name(), "gpt-4o");
        assert_eq!(model_info.provider(), "openai");
        assert!(model_info.has_streaming());
        assert!(model_info.has_function_calling());
        assert!(model_info.has_vision());

        // Test with unsupported model
        assert!(discovery.get_model_info("unsupported-model").is_none());
    }
}
