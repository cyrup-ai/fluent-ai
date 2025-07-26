use anyhow::{anyhow, Context, Result};
use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_http3::Http3;
use serde_json::Value;
use std::env;

pub struct HuggingFaceProvider;

impl ProviderTrait for HuggingFaceProvider {
    async fn get_model_info(&self, model: &str) -> Result<ModelInfo> {
        let mut headers = std::collections::HashMap::new();
        if let Ok(key) = env::var("HF_API_KEY") {
            headers.insert("Authorization".to_string(), format!("Bearer {}", key));
        }
        
        let response = Http3::json()
            .headers(|| headers)
            .get(&format!("https://api.huggingface.co/models/{}", model))
            .collect::<Value>()
            .await
            .context("Failed to fetch HuggingFace model")?;
            
        if response["id"].as_str().unwrap_or("") != model {
            return Err(anyhow!("Model {} not found", model));
        }
        
        Ok(adapt_huggingface_to_model_info(&response))
    }
}

fn adapt_huggingface_to_model_info(data: &Value) -> ModelInfo {
    let name = data["id"].as_str().unwrap_or("unknown").to_string();
    let max_context = data["config"]["max_position_embeddings"]
        .as_u64()
        .unwrap_or(data["max_length"].as_u64().unwrap_or(8192));
    let pricing_input = 0.0;
    let pricing_output = 0.0;
    let is_thinking = data["tags"]
        .as_array()
        .map(|t| t.iter().any(|tag| tag.as_str().unwrap_or("").contains("reasoning")))
        .unwrap_or(false);
    let required_temperature = if is_thinking { Some(1.0) } else { None };
    ModelInfo {
        name,
        max_context,
        pricing_input,
        pricing_output,
        is_thinking,
        required_temperature,
    }
}