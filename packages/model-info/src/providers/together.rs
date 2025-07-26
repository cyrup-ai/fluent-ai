use anyhow::{anyhow, Context, Result};
use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_http3::Http3;
use serde::Deserialize;
use std::env;

#[derive(Deserialize)]
pub struct TogetherModelData {
    pub id: String,
    pub context_length: u64,
    pub pricing: TogetherPricingData,
    pub description: String,
}

#[derive(Deserialize)]
pub struct TogetherPricingData {
    pub input: f64,
    pub output: f64,
}

pub struct TogetherProvider;

impl ProviderTrait for TogetherProvider {
    async fn get_model_info(&self, model: &str) -> Result<ModelInfo> {
        let mut headers = std::collections::HashMap::new();
        if let Ok(key) = env::var("TOGETHER_API_KEY") {
            headers.insert("Authorization".to_string(), format!("Bearer {}", key));
        }
        
        let response = Http3::json()
            .headers(|| headers)
            .get("https://api.together.ai/v1/models")
            .collect::<Vec<TogetherModelData>>()
            .await
            .context("Failed to fetch Together models")?;
            
        let model_data = response.into_iter()
            .find(|m| m.id == model)
            .ok_or(anyhow!("Model {} not found", model))?;
            
        Ok(adapt_together_to_model_info(&model_data))
    }
}

fn adapt_together_to_model_info(data: &TogetherModelData) -> ModelInfo {
    let is_thinking = data.description.to_lowercase().contains("reasoning") || data.id.contains("reasoning");
    let required_temperature = if is_thinking { Some(1.0) } else { None };
    ModelInfo {
        name: data.id.clone(),
        max_context: data.context_length,
        pricing_input: data.pricing.input,
        pricing_output: data.pricing.output,
        is_thinking,
        required_temperature,
    }
}