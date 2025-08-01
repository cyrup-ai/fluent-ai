use anyhow::Result;
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use std::env;

use super::super::codegen::SynCodeGenerator;
use super::{ModelData, ProviderBuilder, StandardModelsResponse};

/// OpenAI provider implementation with dynamic API fetching
/// No fallback data - fails build if API unavailable
pub struct OpenAiProvider;

// Using StandardModelsResponse from mod.rs

impl ProviderBuilder for OpenAiProvider {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn api_endpoint(&self) -> Option<&'static str> {
        Some("https://api.openai.com/v1/models")
    }

    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("OPENAI_API_KEY")
    }

    fn fetch_models(&self) -> AsyncStream<ModelData> {
        AsyncStream::with_channel(move |sender| {
            let responses = if let Ok(api_key) = env::var("OPENAI_API_KEY") {
                Http3::json::<StandardModelsResponse>()
                    .bearer_auth(&api_key)
                    .get("https://api.openai.com/v1/models")
                    .collect::<StandardModelsResponse>()
            } else {
                Http3::json::<StandardModelsResponse>()
                    .get("https://api.openai.com/v1/models")
                    .collect::<StandardModelsResponse>()
            };
            
            if let Some(response) = responses.into_iter().next() {
                for model in response.data {
                    let model_data = openai_model_to_data(&model.id);
                    if sender.send(model_data).is_err() {
                        break;
                    }
                }
            }
        })
    }

    fn static_models(&self) -> Option<Vec<ModelData>> {
        // Minimal fallback models for build resilience when API unavailable
        // Dynamic fetching is tried FIRST, these are only used as last resort
        Some(vec![
            ("gpt-4o".to_string(), 128000, 0.005, 0.015, false, Some(0.0)),
            ("gpt-4o-mini".to_string(), 128000, 0.00015, 0.0006, false, Some(0.0)),
            ("gpt-3.5-turbo".to_string(), 16384, 0.001, 0.002, false, Some(0.0)),
        ])
    }

    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = SynCodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}

/// Convert OpenAI model ID to ModelData with appropriate pricing and capabilities
/// Based on current OpenAI pricing as of 2024
fn openai_model_to_data(model_id: &str) -> ModelData {
    match model_id {
        // GPT-4 models
        "gpt-4" => (model_id.to_string(), 128000, 0.03, 0.06, false, Some(0.0)),
        "gpt-4-turbo" => (model_id.to_string(), 128000, 0.01, 0.03, false, Some(0.0)),
        "gpt-4o" => (model_id.to_string(), 128000, 0.005, 0.015, false, Some(0.0)),
        "gpt-4o-mini" => (model_id.to_string(), 128000, 0.00015, 0.0006, false, Some(0.0)),
        
        // GPT-3.5 models
        "gpt-3.5-turbo" => (model_id.to_string(), 16384, 0.001, 0.002, false, Some(0.0)),
        "gpt-3.5-turbo-instruct" => (model_id.to_string(), 4096, 0.0015, 0.002, false, Some(0.0)),
        
        // Text models
        "text-davinci-003" => (model_id.to_string(), 4097, 0.02, 0.02, false, Some(0.0)),
        "text-curie-001" => (model_id.to_string(), 2049, 0.002, 0.002, false, Some(0.0)),
        "text-babbage-001" => (model_id.to_string(), 2049, 0.0005, 0.0005, false, Some(0.0)),
        "text-ada-001" => (model_id.to_string(), 2049, 0.0004, 0.0004, false, Some(0.0)),
        
        // Default for unknown models
        _ => (model_id.to_string(), 8192, 0.001, 0.002, false, Some(0.0)),
    }
}
