use anyhow::{anyhow, Context, Result};
use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_http3::Http3;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct OpenRouterModelsResponse {
    pub data: Vec<OpenRouterModelData>,
}

#[derive(Deserialize)]
pub struct OpenRouterModelData {
    pub id: String,
    pub context_length: u64,
    pub pricing: OpenRouterPricingData,
    pub description: String,
}

#[derive(Deserialize)]
pub struct OpenRouterPricingData {
    pub prompt: String,
    pub completion: String,
}

pub struct OpenRouterProvider;

impl ProviderTrait for OpenRouterProvider {
    async fn get_model_info(&self, model: &str) -> Result<ModelInfo> {
        let response = Http3::json()
            .get("https://openrouter.ai/api/v1/models")
            .collect::<OpenRouterModelsResponse>()
            .await
            .context("Failed to fetch OpenRouter models")?;
            
        let model_data = response.data.into_iter()
            .find(|m| m.id == model)
            .ok_or(anyhow!("Model {} not found", model))?;
            
        Ok(adapt_openrouter_to_model_info(&model_data))
    }
}

fn adapt_openrouter_to_model_info(data: &OpenRouterModelData) -> ModelInfo {
    let pricing_input = data.pricing.prompt.parse::<f64>().unwrap_or(0.0) * 1_000_000.0;
    let pricing_output = data.pricing.completion.parse::<f64>().unwrap_or(0.0) * 1_000_000.0;
    let is_thinking = data.description.to_lowercase().contains("reasoning") 
        || data.id.contains("o1") 
        || data.id.contains("o3");
    let required_temperature = if is_thinking { Some(1.0) } else { None };
    ModelInfo {
        name: data.id.clone(),
        max_context: data.context_length,
        pricing_input,
        pricing_output,
        is_thinking,
        required_temperature,
    }
}