use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use hashbrown::HashMap;
use std::sync::OnceLock;

#[derive(Clone)]
pub struct GroqProvider;

impl ProviderTrait for GroqProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_groq_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }
    
    fn list_models(&self) -> AsyncStream<ModelInfo> {
        AsyncStream::with_channel(move |sender| {
            let models = vec!["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"];
            
            for model in models {
                let model_info = adapt_groq_to_model_info(model);
                let _ = sender.send(model_info);
            }
        })
    }
    
    fn provider_name(&self) -> &'static str {
        "groq"
    }
}

fn adapt_groq_to_model_info(model: &str) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u32, u32, f64, f64, bool, bool, bool, bool, bool)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // (max_input, max_output, input_price, output_price, vision, function_calling, streaming, embeddings, thinking)
        m.insert("llama-3.1-70b-versatile", (131072, 32768, 0.59, 0.79, false, true, true, false, false));
        m.insert("llama-3.1-8b-instant", (131072, 32768, 0.05, 0.08, false, true, true, false, false));
        m.insert("mixtral-8x7b-32768", (32768, 32768, 0.24, 0.24, false, true, true, false, false));
        m
    });
    
    let (max_input, max_output, pricing_input, pricing_output, supports_vision, supports_function_calling, _supports_streaming, supports_embeddings, supports_thinking) = 
        map.get(model).copied().unwrap_or((32768, 32768, 0.24, 0.24, false, true, true, false, false));
    
    ModelInfo {
        provider_name: "groq",
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
        optimal_thinking_budget: if supports_thinking { Some(32000) } else { None },
        system_prompt_prefix: None,
        real_name: None,
        model_type: None,
        patch: None,
        required_temperature: None,
    }
}