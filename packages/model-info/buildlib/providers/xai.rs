use super::{ModelData, ProviderBuilder, StandardModelsResponse};
use super::super::codegen::SynCodeGenerator;
use anyhow::Result;
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use std::env;

/// X.AI provider implementation with dynamic API fetching
/// Uses the official X.AI API endpoint as documented at https://docs.x.ai/docs/api-reference#list-models
pub struct XaiProvider;

// Using StandardModelsResponse from mod.rs

impl ProviderBuilder for XaiProvider {
    fn provider_name(&self) -> &'static str {
        "xai"
    }
    
    fn api_endpoint(&self) -> Option<&'static str> {
        Some("https://api.x.ai/v1/models")
    }
    
    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("XAI_API_KEY")
    }
    
    fn fetch_models(&self) -> AsyncStream<ModelData> {
        AsyncStream::with_channel(move |sender| {
            let responses = if let Ok(api_key) = env::var("XAI_API_KEY") {
                Http3::json::<StandardModelsResponse>()
                    .bearer_auth(&api_key)
                    .get("https://api.x.ai/v1/models")
                    .collect::<StandardModelsResponse>()
            } else {
                Http3::json::<StandardModelsResponse>()
                    .get("https://api.x.ai/v1/models")
                    .collect::<StandardModelsResponse>()
            };
            
            if let Some(response) = responses.into_iter().next() {
                for model in response.data {
                    let model_data = xai_model_to_data(&model.id);
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
            ("grok-2".to_string(), 131072, 2.0, 10.0, true, Some(1.0)),
            ("grok-2-mini".to_string(), 131072, 0.3, 0.5, true, None),
        ])
    }
    
    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = SynCodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}

/// Convert X.AI model ID to ModelData with appropriate pricing and capabilities
/// Based on current X.AI pricing as of 2024
fn xai_model_to_data(model_id: &str) -> ModelData {
    match model_id {
        "grok-beta" => (model_id.to_string(), 131072, 5.0, 15.0, true, Some(1.0)),
        "grok-2" => (model_id.to_string(), 131072, 2.0, 10.0, true, Some(1.0)),
        "grok-2-mini" => (model_id.to_string(), 131072, 0.3, 0.5, true, None),
        
        // Default for unknown models
        _ => (model_id.to_string(), 131072, 2.0, 10.0, true, None),
    }
}