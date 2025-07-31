use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use hashbrown::HashMap;
use std::sync::OnceLock;

#[derive(Clone)]
pub struct OllamaProvider;

impl ProviderTrait for OllamaProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_ollama_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }
    
    fn list_models(&self) -> AsyncStream<ModelInfo> {
        AsyncStream::with_channel(move |sender| {
            let models = vec!["llama3.2", "llama3.1", "gemma2", "phi3", "qwen2.5"];
            
            for model in models {
                let model_info = adapt_ollama_to_model_info(model);
                let _ = sender.send(model_info);
            }
        })
    }
    
    fn provider_name(&self) -> &'static str {
        "ollama"
    }
}

fn adapt_ollama_to_model_info(model: &str) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u32, u32, f64, f64, bool, bool, bool, bool, bool)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // (max_input, max_output, input_price, output_price, vision, function_calling, streaming, embeddings, thinking)
        // Local models - no pricing
        m.insert("llama3.2", (128000, 4096, 0.0, 0.0, false, true, true, false, false));
        m.insert("llama3.1", (128000, 4096, 0.0, 0.0, false, true, true, false, false));
        m.insert("gemma2", (8192, 4096, 0.0, 0.0, false, true, true, false, false));
        m.insert("phi3", (4096, 4096, 0.0, 0.0, false, true, true, false, false));
        m.insert("qwen2.5", (32768, 8192, 0.0, 0.0, false, true, true, false, false));
        m
    });
    
    let (max_input, max_output, pricing_input, pricing_output, supports_vision, supports_function_calling, _supports_streaming, supports_embeddings, supports_thinking) = 
        map.get(model).copied().unwrap_or((4096, 4096, 0.0, 0.0, false, true, true, false, false));
    
    ModelInfo {
        provider_name: "ollama",
        name: Box::leak(model.to_string().into_boxed_str()),
        max_input_tokens: std::num::NonZeroU32::new(max_input),
        max_output_tokens: std::num::NonZeroU32::new(max_output),
        input_price: Some(pricing_input),
        output_price: Some(pricing_output),
        supports_vision,
        supports_function_calling,
        supports_embeddings,
        requires_max_tokens: false,
        supports_thinking,
        optimal_thinking_budget: if supports_thinking { Some(16000) } else { None },
        system_prompt_prefix: None,
        real_name: None,
        model_type: None,
        patch: None,
        required_temperature: None,
    }
}