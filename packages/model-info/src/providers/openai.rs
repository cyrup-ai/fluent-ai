use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use serde::Deserialize;

#[derive(Deserialize, Default)]
pub struct OpenAiModelsResponse {
    pub data: Vec<OpenAiModelData>,
}

#[derive(Deserialize, Default)]
pub struct OpenAiModelData {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Clone)]
pub struct OpenAiProvider;

impl ProviderTrait for OpenAiProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_openai_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }
}


fn adapt_openai_to_model_info(model: &str) -> ModelInfo {
    use std::sync::OnceLock;
    use hashbrown::HashMap;
    
    static MAP: OnceLock<HashMap<&'static str, (u64, f64, f64, bool, Option<f64>)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("gpt-4", (8192, 30.0, 60.0, false, None));
        m.insert("gpt-4-1106-preview", (128000, 10.0, 30.0, false, None));
        m.insert("gpt-4-turbo", (128000, 10.0, 30.0, false, None));
        m.insert("gpt-4o", (128000, 5.0, 15.0, false, None));
        m.insert("gpt-4o-mini", (128000, 0.15, 0.6, false, None));
        m.insert("gpt-3.5-turbo", (16385, 0.5, 1.5, false, None));
        m.insert("gpt-3.5-turbo-1106", (16385, 1.0, 2.0, false, None));
        m.insert("o1-preview", (128000, 15.0, 60.0, true, Some(1.0)));
        m.insert("o1-mini", (128000, 3.0, 12.0, true, Some(1.0)));
        m
    });
    
    let (max_context, pricing_input, pricing_output, is_thinking, required_temperature) = 
        map.get(model).copied().unwrap_or((4096, 0.0, 0.0, false, None));
    
    ModelInfo {
        name: model.to_string(),
        max_context,
        pricing_input,
        pricing_output,
        is_thinking,
        required_temperature,
    }
}