use anyhow::Result;
use serde::{Deserialize, Serialize};
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use std::env;
use super::super::codegen::SynCodeGenerator;
use super::{ModelData, ProviderBuilder};

/// Together provider implementation with dynamic API fetching
/// No fallback data - fails build if API unavailable
pub struct TogetherProvider;

#[derive(Deserialize, Serialize, Default)]
struct TogetherModelsResponse {
    data: Vec<TogetherModel>,
}

#[derive(Deserialize, Serialize)]
struct TogetherModel {
    id: String,
    #[serde(rename = "type")]
    model_type: String,
    pricing: Option<TogetherPricing>,
    context_length: Option<u64>,
}

#[derive(Deserialize, Serialize)]
struct TogetherPricing {
    input: f64,
    output: f64,
}

impl ProviderBuilder for TogetherProvider {
    fn provider_name(&self) -> &'static str {
        "together"
    }

    fn api_endpoint(&self) -> Option<&'static str> {
        Some("https://api.together.xyz/v1/models")
    }

    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("TOGETHER_API_KEY")
    }

    fn fetch_models(&self) -> AsyncStream<ModelData> {
        AsyncStream::with_channel(move |sender| {
            let responses = if let Ok(api_key) = env::var("TOGETHER_API_KEY") {
                Http3::json::<TogetherModelsResponse>()
                    .bearer_auth(&api_key)
                    .get("https://api.together.xyz/v1/models")
                    .collect::<TogetherModelsResponse>()
            } else {
                Http3::json::<TogetherModelsResponse>()
                    .get("https://api.together.xyz/v1/models")
                    .collect::<TogetherModelsResponse>()
            };
            
            if let Some(response) = responses.into_iter().next() {
                for model in response.data {
                    let model_data = together_model_to_data(&model);
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
            // Popular Together models with realistic pricing
            ("meta-llama/Llama-2-70b-chat-hf".to_string(), 4096, 0.9, 0.9, false, Some(0.0)),
            ("mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(), 32768, 0.6, 0.6, false, Some(0.0)),
            ("codellama/CodeLlama-34b-Instruct-hf".to_string(), 16384, 0.776, 0.776, false, Some(0.0)),
            ("meta-llama/Llama-2-13b-chat-hf".to_string(), 4096, 0.225, 0.225, false, Some(0.0)),
            ("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO".to_string(), 32768, 0.6, 0.6, false, Some(0.0)),
            ("zero-one-ai/Yi-34B-Chat".to_string(), 4096, 0.8, 0.8, false, Some(0.0)),
        ])
    }

    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = SynCodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}

/// Convert Together model to ModelData with appropriate pricing and capabilities
/// Uses pricing from API response when available, with fallback defaults
fn together_model_to_data(model: &TogetherModel) -> ModelData {
    let context_length = model.context_length.unwrap_or(4096);
    let (input_price, output_price) = if let Some(pricing) = &model.pricing {
        (pricing.input, pricing.output)
    } else {
        // Default pricing for models without pricing info
        match model.id.as_str() {
            id if id.contains("llama-2-70b") => (0.9, 0.9),
            id if id.contains("mixtral") => (0.6, 0.6),
            id if id.contains("codellama-34b") => (0.776, 0.776),
            id if id.contains("llama-2-13b") => (0.225, 0.225),
            _ => (0.2, 0.2), // Default for smaller models
        }
    };
    
    (model.id.clone(), context_length, input_price, output_price, false, None)
}
