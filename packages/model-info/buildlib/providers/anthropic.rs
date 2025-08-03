use super::{ModelData, ProviderBuilder, ProcessProviderResult};
use serde::{Deserialize, Serialize};

/// Anthropic provider implementation with legitimate hardcoded data
/// This is the ONLY provider that legitimately uses static data because
/// Anthropic/Claude does not provide a public models list API endpoint
pub struct AnthropicProvider;

// Placeholder type since Anthropic doesn't have an API endpoint
#[derive(Deserialize, Serialize, Default)]
pub struct AnthropicModelsResponse;

impl ProviderBuilder for AnthropicProvider {
    type ListResponse = AnthropicModelsResponse; // Placeholder - no API
    type GetResponse = AnthropicModelsResponse;

    fn provider_name(&self) -> &'static str {
        "anthropic"
    }
    
    fn base_url(&self) -> &'static str {
        "https://api.anthropic.com" // Not used since no public models API
    }
    
    fn api_key_env_var(&self) -> Option<&'static str> {
        // No API key needed since there's no API endpoint
        None
    }

    fn response_to_models(&self, _response: Self::ListResponse) -> Vec<ModelData> {
        // Not used since Anthropic has no API
        Vec::new()
    }

    fn model_to_data(&self, _model: &Self::GetResponse) -> ModelData {
        // Not used since Anthropic has no API - placeholder implementation
        ("unknown".to_string(), 0, 0.0, 0.0, false, None)
    }

    // Custom process() implementation for Anthropic's static models
    fn process(&self) -> ProcessProviderResult {
        // Get static models - ONLY legitimate use since Anthropic has no public /v1/models API
        let models = match self.static_models() {
            Some(models) => models,
            None => {
                return ProcessProviderResult {
                    success: false,
                    status: format!("No static models defined for {}", self.provider_name()),
                };
            }
        };

        if models.is_empty() {
            return ProcessProviderResult {
                success: false,
                status: format!("No models found for {}", self.provider_name()),
            };
        }

        // Generate code using syn
        match self.generate_code(&models) {
            Ok((_enum_code, _impl_code)) => ProcessProviderResult {
                success: true,
                status: format!("Successfully processed {} static models for {}", models.len(), self.provider_name()),
            },
            Err(e) => ProcessProviderResult {
                success: false,
                status: format!("Code generation failed for {}: {}", self.provider_name(), e),
            },
        }
    }

    fn static_models(&self) -> Option<Vec<ModelData>> {
        // ONLY legitimate use of static model data - Anthropic has no public /v1/models API
        // User-specified claude-4 models only
        Some(vec![
            (
                "claude-4-sonnet".to_string(),
                200000,
                3.0,
                15.0,
                false,
                None,
            ),
            (
                "claude-4-sonnet-thinking".to_string(),
                200000,
                3.0,
                15.0,
                true,
                Some(1.0),
            ),
            (
                "claude-4-opus".to_string(),
                200000,
                15.0,
                75.0,
                false,
                None,
            ),
            (
                "claude-4-opus-thinking".to_string(),
                200000,
                15.0,
                75.0,
                true,
                Some(1.0),
            ),
        ])
    }
}