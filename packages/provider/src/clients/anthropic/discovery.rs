//! Anthropic model discovery and registration

use std::sync::Arc;

use async_trait::async_trait;
use fluent_ai_domain::model::{
    capabilities::Capability,
    error::ModelError,
    info::{ModelInfo, ModelInfoBuilder},
    registry::ModelRegistry,
    traits::Model};
use tracing::{debug, error, info, instrument, warn};

use super::client::AnthropicClient;
use crate::discovery::{DiscoveryError, DiscoveryResult, ProviderModelDiscovery};

/// Anthropic model discovery implementation
#[derive(Debug, Clone)]
pub struct AnthropicDiscovery {
    client: Arc<AnthropicClient>,
    supported_models: &'static [&'static str]}

impl AnthropicDiscovery {
    /// Create a new Anthropic model discovery instance
    pub fn new(api_key: String) -> Result<Self, ModelError> {
        let client = AnthropicClient::new(api_key).map_err(|e| {
            ModelError::ProviderError(format!("Failed to create Anthropic client: {}", e))
        })?;

        // List of supported models with their capabilities
        static SUPPORTED_MODELS: &[&str] = &[
            // Claude 4 models (newest and most powerful)
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            // Claude 3.7 models
            "claude-3-7-sonnet-20250219",
            // Claude 3.5 models
            "claude-3-5-sonnet-20241022", // v2 (latest)
            "claude-3-5-sonnet-20240620", // v1 (original)
            "claude-3-5-haiku-20241022",
            // Claude 3 models
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ];

        Ok(Self {
            client: Arc::new(client),
            supported_models: SUPPORTED_MODELS})
    }

    /// Get model information for a specific model
    #[instrument(skip(self))]
    fn get_model_info(&self, model_name: &'static str) -> Option<ModelInfo> {
        // Determine model capabilities based on model name patterns
        let mut capabilities = vec![Capability::TextGeneration];

        // All Claude 3+ models support function calling
        if model_name.contains("claude-3")
            || model_name.contains("claude-opus-4")
            || model_name.contains("claude-sonnet-4")
        {
            capabilities.push(Capability::FunctionCalling);
        }

        // Determine context length and other model-specific properties
        let (context_length, max_output_tokens) = if model_name.contains("opus") {
            (200_000, 8_192) // Opus models have larger context
        } else if model_name.contains("sonnet") {
            (200_000, 8_192) // Sonnet models also have large context
        } else if model_name.contains("haiku") {
            (100_000, 4_096) // Haiku models are more limited
        } else {
            (100_000, 4_096) // Default values for other models
        };

        Some(
            ModelInfoBuilder::new(model_name, "anthropic")
                .with_display_name(model_name)
                .with_max_input_tokens(context_length)
                .with_max_output_tokens(max_output_tokens)
                .with_capabilities(capabilities)
                .with_parameter("temperature", 0.7)
                .with_parameter("top_p", 0.9)
                .with_metadata("supports_tools", "true")
                .with_metadata("supports_vision", "false")
                .with_metadata("supports_audio", "false")
                .build()
                .expect("Failed to build model info"),
        )
    }
}

impl ProviderModelDiscovery for AnthropicDiscovery {
    fn discover_models(&self) -> DiscoveryResult<Vec<String>> {
        Ok(self.supported_models.iter().map(|s| s.to_string()).collect())
    }

    fn get_model_info(&self, model_name: &str) -> DiscoveryResult<ModelInfo> {
        // Implementation for getting model info
        todo!("Implement get_model_info for Anthropic models")
    }

    fn is_model_available(&self, model_name: &str) -> bool {
        self.supported_models.contains(&model_name)
    }
}

// Legacy implementation for backwards compatibility
impl AnthropicDiscovery {
    pub fn provider_name(&self) -> &'static str {
        "anthropic"
    }

    pub async fn discover_and_register(&self) -> DiscoveryResult<()> {
        info!("Starting Anthropic model discovery");
        let registry = ModelRegistry::global();

        for &model_name in self.supported_models {
            if let Some(model_info) = self.get_model_info(model_name) {
                // Create a new model instance for each model
                let model = AnthropicModel {
                    info: model_info,
                    client: self.client.clone()};

                // Register the model
                if let Err(e) = registry.register("anthropic", model) {
                    error!("Failed to register Anthropic model {}: {}", model_name, e);
                    return Err(DiscoveryError::RegistrationFailed(e.to_string()));
                }

                debug!("Registered Anthropic model: {}", model_name);
            } else {
                warn!("Skipping unsupported Anthropic model: {}", model_name);
            }
        }

        info!("Completed Anthropic model discovery");
        Ok(())
    }

    pub fn supported_models(&self) -> &'static [&'static str] {
        self.supported_models
    }
}

/// Anthropic model implementation
#[derive(Debug, Clone)]
struct AnthropicModel {
    info: ModelInfo,
    client: Arc<AnthropicClient>}

impl Model for AnthropicModel {
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
    async fn test_anthropic_discovery() {
        // Create a discovery instance with a dummy API key for testing
        let discovery = AnthropicDiscovery::new("test-api-key".to_string())
            .expect("Failed to create AnthropicDiscovery in test");

        // Test provider name
        assert_eq!(discovery.provider_name(), "anthropic");

        // Test supported models
        let supported_models = discovery.supported_models();
        assert!(!supported_models.is_empty());
        assert!(supported_models.contains(&&"claude-3-5-sonnet-20241022"));

        // Test model info retrieval
        let model_info = discovery
            .get_model_info("claude-3-5-sonnet-20241022")
            .expect("Failed to get model info for claude-3-5-sonnet-20241022 in test");
        assert_eq!(model_info.name(), "claude-3-5-sonnet-20241022");
        assert_eq!(model_info.provider(), "anthropic");
        assert!(model_info.has_capability(Capability::TextGeneration));
        assert!(model_info.has_capability(Capability::FunctionCalling));

        // Test model registration (in-memory only for tests)
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
                    .get_model::<AnthropicModel>("anthropic", model_name)
                    .is_ok(),
                "Model {} was not registered",
                model_name
            );
        }
    }

    #[test]
    fn test_get_model_info() {
        let discovery = AnthropicDiscovery::new("test-api-key".to_string())
            .expect("Failed to create AnthropicDiscovery in test_get_model_info");

        // Test with supported model
        let model_info = discovery
            .get_model_info("claude-3-5-sonnet-20241022")
            .expect(
                "Failed to get model info for claude-3-5-sonnet-20241022 in test_get_model_info",
            );
        assert_eq!(model_info.name(), "claude-3-5-sonnet-20241022");
        assert_eq!(model_info.provider(), "anthropic");
        assert!(model_info.has_capability(Capability::TextGeneration));
        assert!(model_info.has_capability(Capability::FunctionCalling));

        // Test with unsupported model
        assert!(discovery.get_model_info("unsupported-model").is_none());
    }
}
