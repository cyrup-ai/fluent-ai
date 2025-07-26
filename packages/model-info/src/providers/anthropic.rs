use anyhow::{anyhow, Context, Result};
use crate::common::{ModelInfo, ProviderTrait};
use hashbrown::HashMap;
use fluent_ai_http3::Http3;
use serde_json::Value;
use std::env;
use std::sync::OnceLock;

pub struct AnthropicProvider;

impl ProviderTrait for AnthropicProvider {
    async fn get_model_info(&self, model: &str) -> Result<ModelInfo> {
        let key = env::var("ANTHROPIC_API_KEY").map_err(|_| anyhow!("ANTHROPIC_API_KEY not set"))?;
        
        let mut headers = std::collections::HashMap::new();
        headers.insert("Authorization".to_string(), format!("Bearer {}", key));
        headers.insert("x-api-key".to_string(), key);
        headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());
        
        let response = Http3::json()
            .headers(|| headers)
            .get("https://api.anthropic.com/v1/models")
            .collect::<Value>()
            .await
            .context("Failed to fetch Anthropic models")?;
            
        let models = response["data"].as_array()
            .ok_or(anyhow!("Invalid response format"))?;
            
        let found = models.iter().find(|m| m["id"].as_str() == Some(model));
        if found.is_none() {
            return Err(anyhow!("Model {} not found", model));
        }
        
        Ok(adapt_anthropic_to_model_info(model))
    }
}

fn adapt_anthropic_to_model_info(model: &str) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u64, f64, f64, bool, Option<f64>)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("claude-3-5-sonnet-20240620", (200000, 3.0, 15.0, false, None));
        m.insert("claude-3-haiku-20240307", (200000, 0.25, 1.25, false, None));
        m.insert("claude-3-opus-20240229", (200000, 15.0, 75.0, false, None));
        m
    });
    let (max_context, pricing_input, pricing_output, is_thinking, required_temperature) = 
        map.get(model).copied().unwrap_or((200000, 0.0, 0.0, false, None));
    ModelInfo {
        name: model.to_string(),
        max_context,
        pricing_input,
        pricing_output,
        is_thinking,
        required_temperature,
    }
}