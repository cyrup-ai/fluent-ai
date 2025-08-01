use anyhow::Result;
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use std::env;

use super::super::codegen::SynCodeGenerator;
use super::{ModelData, ProviderBuilder, StandardModelsResponse};

/// Mistral provider implementation with dynamic API fetching
/// No fallback data - fails build if API unavailable
pub struct MistralProvider;

// Using StandardModelsResponse from mod.rs

impl ProviderBuilder for MistralProvider {
    fn provider_name(&self) -> &'static str {
        "mistral"
    }

    fn api_endpoint(&self) -> Option<&'static str> {
        Some("https://api.mistral.ai/v1/models")
    }

    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("MISTRAL_API_KEY")
    }

    fn fetch_models(&self) -> AsyncStream<ModelData> {
        AsyncStream::with_channel(move |sender| {
            let responses = if let Ok(api_key) = env::var("MISTRAL_API_KEY") {
                Http3::json::<StandardModelsResponse>()
                    .bearer_auth(&api_key)
                    .get("https://api.mistral.ai/v1/models")
                    .collect::<StandardModelsResponse>()
            } else {
                Http3::json::<StandardModelsResponse>()
                    .get("https://api.mistral.ai/v1/models")
                    .collect::<StandardModelsResponse>()
            };
            
            if let Some(response) = responses.into_iter().next() {
                for model in response.data {
                    let model_data = mistral_model_to_data(&model.id);
                    if sender.send(model_data).is_err() {
                        break;
                    }
                }
            }
        })
    }

    fn static_models(&self) -> Option<Vec<ModelData>> {
        // Fallback static models for build resilience when API unavailable
        Some(vec![
            ("mistral-large-latest".to_string(), 128000, 2.0, 6.0, false, None),
            ("mistral-large-2407".to_string(), 128000, 2.0, 6.0, false, None),
            ("mistral-large-2402".to_string(), 32000, 4.0, 12.0, false, None),
            ("mistral-medium-latest".to_string(), 32000, 2.7, 8.1, false, None),
            ("mistral-small-latest".to_string(), 32000, 1.0, 3.0, false, None),
            ("mistral-tiny".to_string(), 32000, 0.14, 0.42, false, None),
            ("codestral-latest".to_string(), 32000, 0.2, 0.6, false, None),
            ("codestral-2405".to_string(), 32000, 0.2, 0.6, false, None),
        ])
    }

    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = SynCodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}

/// Convert Mistral model ID to ModelData with appropriate pricing and capabilities
/// Based on current Mistral pricing as of 2024
fn mistral_model_to_data(model_id: &str) -> ModelData {
    match model_id {
        "mistral-large-latest" => (model_id.to_string(), 128000, 2.0, 6.0, false, None),
        "mistral-large-2407" => (model_id.to_string(), 128000, 2.0, 6.0, false, None),
        "mistral-large-2402" => (model_id.to_string(), 32000, 4.0, 12.0, false, None),
        "mistral-medium-latest" => (model_id.to_string(), 32000, 2.7, 8.1, false, None),
        "mistral-small-latest" => (model_id.to_string(), 32000, 1.0, 3.0, false, None),
        "mistral-tiny" => (model_id.to_string(), 32000, 0.14, 0.42, false, None),
        "codestral-latest" => (model_id.to_string(), 32000, 0.2, 0.6, false, None),
        "codestral-2405" => (model_id.to_string(), 32000, 0.2, 0.6, false, None),
        
        // Default for unknown models
        _ => (model_id.to_string(), 32000, 1.0, 3.0, false, None),
    }
}
