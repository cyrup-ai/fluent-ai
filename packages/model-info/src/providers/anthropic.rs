use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use hashbrown::HashMap;
use std::sync::OnceLock;

#[derive(Clone)]
pub struct AnthropicProvider;

impl ProviderTrait for AnthropicProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_anthropic_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }
}

fn adapt_anthropic_to_model_info(model: &str) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u64, f64, f64, bool, Option<f64>)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("claude-3-5-sonnet-20240620", (200000, 3.0, 15.0, false, None));
        m.insert("claude-3-haiku-20240307", (200000, 0.25, 1.25, false, None));
        m.insert("claude-3-opus-20240229", (200000, 15.0, 75.0, false, None));
        m.insert("claude-3-sonnet-20240229", (200000, 3.0, 15.0, false, None));
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