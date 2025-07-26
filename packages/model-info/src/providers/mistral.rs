use anyhow::{anyhow, Context, Result};
use crate::common::{ModelInfo, ProviderTrait};
use hashbrown::HashMap;
use fluent_ai_http3::Http3;
use serde::Deserialize;
use std::env;
use std::sync::OnceLock;

#[derive(Deserialize)]
pub struct MistralModelsResponse {
    pub object: String,
    pub data: Vec<MistralModelData>,
}

#[derive(Deserialize)]
pub struct MistralModelData {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

pub struct MistralProvider;

impl ProviderTrait for MistralProvider {
    async fn get_model_info(&self, model: &str) -> Result<ModelInfo> {
        let key = env::var("MISTRAL_API_KEY").map_err(|_| anyhow!("MISTRAL_API_KEY not set"))?;
        
        let response = Http3::json()
            .api_key(&key)
            .get("https://api.mistral.ai/v1/models")
            .collect::<MistralModelsResponse>()
            .await
            .context("Failed to fetch Mistral models")?;
            
        let model_data = response.data.into_iter()
            .find(|m| m.id == model)
            .ok_or(anyhow!("Model {} not found", model))?;
            
        Ok(adapt_mistral_to_model_info(&model_data))
    }
}

fn adapt_mistral_to_model_info(data: &MistralModelData) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u64, f64, f64, bool, Option<f64>)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("mistral-large-2407", (128000, 8.0, 24.0, false, None));
        m.insert("mistral-large-2312", (32000, 3.0, 9.0, false, None));
        m.insert("mistral-small-2409", (128000, 0.2, 0.6, false, None));
        m.insert("mistral-nemo-2407", (128000, 0.2, 0.6, false, None));
        m.insert("open-mistral-nemo", (128000, 0.3, 1.0, false, None));
        m.insert("codestral-2405", (32000, 0.8, 2.5, false, None));
        m.insert("mistral-embed", (8192, 0.1, 0.1, false, None));
        m.insert("mistral-tiny", (32000, 0.25, 0.75, false, None));
        m.insert("mistral-small", (32000, 2.0, 6.0, false, None));
        m.insert("mistral-medium", (32000, 8.0, 24.0, false, None));
        m.insert("mistral-large", (32000, 20.0, 60.0, false, None));
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