use anyhow::{anyhow, Context, Result};
use crate::common::{ModelInfo, ProviderTrait};
use hashbrown::HashMap;
use fluent_ai_http3::Http3;
use serde::Deserialize;
use std::env;
use std::sync::OnceLock;

#[derive(Deserialize)]
pub struct XaiModelsResponse {
    pub object: String,
    pub data: Vec<XaiModelData>,
}

#[derive(Deserialize)]
pub struct XaiModelData {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

pub struct XaiProvider;

impl ProviderTrait for XaiProvider {
    async fn get_model_info(&self, model: &str) -> Result<ModelInfo> {
        let key = env::var("XAI_API_KEY").map_err(|_| anyhow!("XAI_API_KEY not set"))?;
        
        let response = Http3::json()
            .api_key(&key)
            .get("https://api.x.ai/v1/models")
            .collect::<XaiModelsResponse>()
            .await
            .context("Failed to fetch xAI models")?;
            
        let model_data = response.data.into_iter()
            .find(|m| m.id == model)
            .ok_or(anyhow!("Model {} not found", model))?;
            
        Ok(adapt_xai_to_model_info(&model_data))
    }
}

fn adapt_xai_to_model_info(data: &XaiModelData) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u64, f64, f64, bool, Option<f64>)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("grok-4", (256000, 3.0, 15.0, true, Some(1.0)));
        m.insert("grok-3", (131072, 3.0, 15.0, true, Some(1.0)));
        m.insert("grok-3-mini", (131072, 0.3, 0.5, true, None));
        m
    });
    let (max_context, pricing_input, pricing_output, is_thinking, required_temperature) = 
        map.get(data.id.as_str()).copied().unwrap_or((8192, 0.0, 0.0, true, None));
    ModelInfo {
        name: data.id.clone(),
        max_context,
        pricing_input,
        pricing_output,
        is_thinking,
        required_temperature,
    }
}