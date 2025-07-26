use anyhow::{anyhow, Context, Result};
use crate::common::{ModelInfo, ProviderTrait};
use hashbrown::HashMap;
use fluent_ai_http3::Http3;
use serde::Deserialize;
use std::env;
use std::sync::OnceLock;

#[derive(Deserialize)]
pub struct OpenAiModelsResponse {
    pub object: String,
    pub data: Vec<OpenAiModelData>,
}

#[derive(Deserialize)]
pub struct OpenAiModelData {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

pub struct OpenAiProvider;

impl ProviderTrait for OpenAiProvider {
    async fn get_model_info(&self, model: &str) -> Result<ModelInfo> {
        let key = env::var("OPENAI_API_KEY").map_err(|_| anyhow!("OPENAI_API_KEY not set"))?;
        
        let response = Http3::json()
            .api_key(&key)
            .get("https://api.openai.com/v1/models")
            .collect::<OpenAiModelsResponse>()
            .await
            .context("Failed to fetch OpenAI models")?;
            
        let model_data = response.data.into_iter()
            .find(|m| m.id == model)
            .ok_or(anyhow!("Model {} not found", model))?;
            
        Ok(adapt_openai_to_model_info(&model_data))
    }
}

fn adapt_openai_to_model_info(data: &OpenAiModelData) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u64, f64, f64, bool, Option<f64>)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("gpt-4.1", (128000, 2.0, 8.0, false, None));
        m.insert("gpt-4.1-mini", (128000, 0.4, 1.6, false, None));
        m.insert("o3", (200000, 3.0, 12.0, true, Some(1.0)));
        m.insert("o4-mini", (128000, 1.1, 4.4, true, Some(1.0)));
        m.insert("gpt-4o", (128000, 5.0, 15.0, false, None));
        m.insert("gpt-4o-mini", (128000, 0.15, 0.6, false, None));
        m
    });
    let (max_context, pricing_input, pricing_output, is_thinking, required_temperature) = 
        map.get(data.id.as_str()).copied().unwrap_or((8192, 0.0, 0.0, false, None));
    ModelInfo {
        name: data.id.clone(),
        max_context,
        pricing_input,
        pricing_output,
        is_thinking,
        required_temperature,
    }
}