//! OpenAI model discovery and registration

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, instrument, trace, warn};

use super::super::discovery::{DiscoveryError, DiscoveryResult, ProviderModelDiscovery};
use super::{
    client::OpenAIClient,
    error::OpenAIError,
    model_info::{
        get_model_config, model_name_from_variant, model_supports_audio, model_supports_tools,
        model_supports_vision,
    },
};
use crate::model::{
    error::ModelError,
    info::{ModelCapability, ModelInfo, ModelInfoBuilder},
    registry::{ModelRegistry, RegisteredModel},
    traits::Model,
};

/// OpenAI model discovery implementation
#[derive(Debug, Clone)]
pub struct OpenAIDiscovery {
    client: Arc<OpenAIClient>,
    supported_models: &'static [&'static str],
}

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
            supported_models: SUPPORTED_MODELS,
        }
    }

    /// Get model information for a specific model
    #[instrument(skip(self))]
    fn get_model_info(&self, model_name: &'static str) -> Option<ModelInfo> {
        let config = get_model_config(model_name)?;

        let mut capabilities = vec![ModelCapability::TextGeneration];

        if config.supports_tools {
            capabilities.push(ModelCapability::FunctionCalling);
        }

        if config.supports_vision {
            capabilities.push(ModelCapability::Vision);
        }

        if config.supports_audio {
            capabilities.push(ModelCapability::Audio);
        }

        Some(
            ModelInfoBuilder::new(model_name, "openai")
                .with_display_name(model_name)
                .with_max_input_tokens(config.context_length)
                .with_max_output_tokens(config.max_tokens)
                .with_capabilities(capabilities)
                .with_parameter("temperature", config.temperature as f64)
                .with_parameter("top_p", config.top_p as f64)
                .with_parameter("frequency_penalty", config.frequency_penalty as f64)
                .with_parameter("presence_penalty", config.presence_penalty as f64)
                .with_metadata("supports_tools", config.supports_tools.to_string())
                .with_metadata("supports_vision", config.supports_vision.to_string())
                .with_metadata("supports_audio", config.supports_audio.to_string())
                .build()
                .expect("Failed to build model info"),
        )
    }
}

#[async_trait]
impl ProviderModelDiscovery for OpenAIDiscovery {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    async fn discover_and_register(&self) -> DiscoveryResult<()> {
        info!("Starting OpenAI model discovery");
        let registry = ModelRegistry::global();

        for &model_name in self.supported_models {
            if let Some(model_info) = self.get_model_info(model_name) {
                // Create a new model instance for each model
                let model = OpenAIModel {
                    info: model_info,
                    client: self.client.clone(),
                };

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
    client: Arc<OpenAIClient>,
}

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
        assert!(model_info.has_capability(ModelCapability::TextGeneration));

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
        assert!(model_info.has_capability(ModelCapability::TextGeneration));
        assert!(model_info.has_capability(ModelCapability::FunctionCalling));
        assert!(model_info.has_capability(ModelCapability::Vision));

        // Test with unsupported model
        assert!(discovery.get_model_info("unsupported-model").is_none());
    }
}
