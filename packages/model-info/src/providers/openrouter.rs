use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use serde::Deserialize;
use std::env;

#[derive(Deserialize, Default)]
pub struct OpenRouterModelsResponse {
    pub data: Vec<OpenRouterModelData>,
}

#[derive(Deserialize, Default)]
pub struct OpenRouterModelData {
    pub id: String,
    pub name: String,
    pub created: u64,
    pub context_length: Option<u64>,
    pub pricing: OpenRouterPricingData,
}

#[derive(Deserialize, Default)]
pub struct OpenRouterPricingData {
    pub prompt: String,
    pub completion: String,
}

#[derive(Clone)]
pub struct OpenRouterProvider;

impl ProviderTrait for OpenRouterProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let key = env::var("OPENROUTER_API_KEY")
                .expect("OPENROUTER_API_KEY environment variable is required");
            
            let response = Http3::json()
                .api_key(&key)
                .get("https://openrouter.ai/api/v1/models")
                .collect::<OpenRouterModelsResponse>();
                
            let model_data = response.data.into_iter()
                .find(|m| m.id == model_name);
                
            let model_info = if let Some(data) = model_data {
                adapt_openrouter_to_model_info(&data)
            } else {
                panic!("Model '{}' not found in OpenRouter API response", model_name);
            };
            
            let _ = sender.send(model_info);
        })
    }
}

fn adapt_openrouter_to_model_info(data: &OpenRouterModelData) -> ModelInfo {
    let context_length = data.context_length.unwrap_or(4096);
    
    // Parse pricing strings to floats (they come as strings like "0.000001")
    let pricing_input = data.pricing.prompt.parse::<f64>().unwrap_or(0.0) * 1_000_000.0; // Convert to per 1M tokens
    let pricing_output = data.pricing.completion.parse::<f64>().unwrap_or(0.0) * 1_000_000.0;
    
    // Determine if it's a thinking model based on name
    let is_thinking = data.id.contains("o1") || data.id.contains("reasoning");
    let required_temperature = if is_thinking { Some(1.0) } else { None };
    
    ModelInfo {
        name: data.id.clone(),
        max_context: context_length,
        pricing_input,
        pricing_output,
        is_thinking,
        required_temperature,
    }
}