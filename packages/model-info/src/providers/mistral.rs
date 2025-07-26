use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use serde::Deserialize;

#[derive(Deserialize, Default)]
pub struct MistralModelsResponse {
    pub data: Vec<MistralModelData>,
}

#[derive(Deserialize, Default)]
pub struct MistralModelData {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Clone)]
pub struct MistralProvider;

impl ProviderTrait for MistralProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_mistral_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }
}

fn adapt_mistral_to_model_info(model: &str) -> ModelInfo {
    use std::sync::OnceLock;
    use hashbrown::HashMap;
    
    static MAP: OnceLock<HashMap<&'static str, (u64, f64, f64, bool, Option<f64>)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("mistral-large-latest", (32768, 3.0, 9.0, false, None));
        m.insert("mistral-medium", (32768, 2.7, 8.1, false, None));
        m.insert("mistral-small", (32768, 1.0, 3.0, false, None));
        m.insert("mistral-tiny", (32768, 0.25, 0.75, false, None));
        m.insert("codestral-latest", (32768, 1.0, 3.0, false, None));
        m.insert("mistral-embed", (8192, 0.1, 0.0, false, None));
        m
    });
    
    let (max_context, pricing_input, pricing_output, is_thinking, required_temperature) = 
        map.get(model).copied().unwrap_or((32768, 0.0, 0.0, false, None));
    
    ModelInfo {
        name: model.to_string(),
        max_context,
        pricing_input,
        pricing_output,
        is_thinking,
        required_temperature,
    }
}