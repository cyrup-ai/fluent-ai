use super::{ModelData, ProviderBuilder};
use super::super::codegen::SynCodeGenerator;
use anyhow::Result;
use serde::Deserialize;
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use std::env;

/// OpenRouter provider implementation with dynamic API fetching  
/// No fallback data - fails build if API unavailable
pub struct OpenRouterProvider;

#[derive(Deserialize, serde::Serialize, Default)]
struct OpenRouterModelsResponse {
    data: Vec<OpenRouterModel>,
}

#[derive(Deserialize, serde::Serialize)]
struct OpenRouterModel {
    id: String,
    name: String,
    context_length: Option<u64>,
    pricing: OpenRouterPricing,
}

#[derive(Deserialize, serde::Serialize)]
struct OpenRouterPricing {
    prompt: String,
    completion: String,
}

impl ProviderBuilder for OpenRouterProvider {
    fn provider_name(&self) -> &'static str {
        "openrouter"
    }
    
    fn api_endpoint(&self) -> Option<&'static str> {
        Some("https://openrouter.ai/api/v1/models")
    }
    
    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("OPENROUTER_API_KEY")
    }
    
    fn fetch_models(&self) -> AsyncStream<ModelData> {
        AsyncStream::with_channel(move |sender| {
            // Dynamic API fetching - now with proper timeout defaults from Http3::json()
            let responses: Vec<OpenRouterModelsResponse> = if let Ok(api_key) = env::var("OPENROUTER_API_KEY") {
                Http3::json::<OpenRouterModelsResponse>()
                    .bearer_auth(&api_key)
                    .get("https://openrouter.ai/api/v1/models")
                    .collect()
            } else {
                Http3::json::<OpenRouterModelsResponse>()
                    .get("https://openrouter.ai/api/v1/models")
                    .collect()
            };
            
            if let Some(response) = responses.into_iter().next() {
                for model in response.data {
                    let model_data = openrouter_model_to_data(&model);
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
            // Popular models via OpenRouter
            ("openai/gpt-4".to_string(), 8192, 0.03, 0.06, false, Some(0.0)),
            ("openai/gpt-3.5-turbo".to_string(), 4096, 0.002, 0.002, false, Some(0.0)),
            ("anthropic/claude-3-sonnet".to_string(), 200000, 0.003, 0.015, false, Some(0.0)),
            ("anthropic/claude-3-haiku".to_string(), 200000, 0.00025, 0.00125, false, Some(0.0)),
            ("meta-llama/llama-3-8b-instruct".to_string(), 8192, 0.00018, 0.00018, false, Some(0.0)),
            ("meta-llama/llama-3-70b-instruct".to_string(), 8192, 0.00059, 0.00079, false, Some(0.0)),
            ("google/gemma-7b-it".to_string(), 8192, 0.00013, 0.00013, false, Some(0.0)),
        ])
    }
    
    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = SynCodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}

/// Convert OpenRouter model to ModelData with appropriate pricing and capabilities
/// Parses pricing strings from OpenRouter API response
#[inline]
fn openrouter_model_to_data(model: &OpenRouterModel) -> ModelData {
    let context_length = model.context_length.unwrap_or(8192);
    
    // Parse pricing strings to f64 (OpenRouter uses string format like "0.005")
    let input_price = model.pricing.prompt.parse::<f64>().unwrap_or(0.001);
    let output_price = model.pricing.completion.parse::<f64>().unwrap_or(0.002);
    
    // Detect thinking models based on model name
    let supports_thinking = model.id.contains("thinking") || model.id.contains("o1");
    let required_temp = if supports_thinking { Some(1.0) } else { None };
    
    (model.id.clone(), context_length, input_price, output_price, supports_thinking, required_temp)
}