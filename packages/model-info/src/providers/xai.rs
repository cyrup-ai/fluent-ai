use crate::common::{ModelInfo, ProviderTrait};
use hashbrown::HashMap;
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use serde::Deserialize;
use std::env;
use std::sync::OnceLock;

#[derive(Deserialize, Default)]
pub struct XaiModelsResponse {
    pub object: String,
    pub data: Vec<XaiModelData>,
}

#[derive(Deserialize, Default)]
pub struct XaiModelData {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Clone)]
pub struct XaiProvider;

impl ProviderTrait for XaiProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let key = env::var("XAI_API_KEY")
                .expect("XAI_API_KEY environment variable is required");
            
            let response = Http3::json()
                .api_key(&key)
                .get("https://api.x.ai/v1/models")
                .collect::<XaiModelsResponse>();
                
            let model_data = response.data.into_iter()
                .find(|m| m.id == model_name);
                
            let model_info = if let Some(data) = model_data {
                adapt_xai_to_model_info(&data)
            } else {
                panic!("Model '{}' not found in XAI API response", model_name);
            };
            
            let _ = sender.send(model_info);
        })
    }
}

fn adapt_xai_to_model_info(data: &XaiModelData) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u64, f64, f64, bool, Option<f64>)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("grok-4", (256000, 3.0, 15.0, true, Some(1.0)));
        m.insert("grok-3", (131072, 3.0, 15.0, true, Some(1.0)));
        m.insert("grok-3-mini", (131072, 0.3, 0.5, true, None));
        m.insert("grok-beta", (131072, 5.0, 15.0, true, Some(1.0)));
        m.insert("grok-2", (131072, 2.0, 10.0, true, Some(1.0)));
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