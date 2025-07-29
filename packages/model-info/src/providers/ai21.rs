use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use hashbrown::HashMap;
use std::sync::OnceLock;

#[derive(Clone)]
pub struct Ai21Provider;

impl ProviderTrait for Ai21Provider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            Box::pin(async move {
                let model_info = adapt_ai21_to_model_info(&model_name);
                let _ = sender.send(model_info).await;
                Ok(())
            })
        })
    }
    
    fn list_models(&self) -> AsyncStream<ModelInfo> {
        AsyncStream::with_channel(move |sender| {
            Box::pin(async move {
                let models = vec!["jamba-1.5-large", "jamba-1.5-mini", "j2-ultra", "j2-mid"];
                
                for model in models {
                    let model_info = adapt_ai21_to_model_info(model);
                    let _ = sender.send(model_info).await;
                }
                Ok(())
            })
        })
    }
    
    fn provider_name(&self) -> &'static str {
        "ai21"
    }
}

fn adapt_ai21_to_model_info(model: &str) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u32, u32, f64, f64, bool, bool, bool, bool, bool)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // (max_input, max_output, input_price, output_price, vision, function_calling, streaming, embeddings, thinking)
        m.insert("jamba-1.5-large", (256000, 4096, 2.0, 8.0, false, true, true, false, false));
        m.insert("jamba-1.5-mini", (256000, 4096, 0.2, 0.4, false, true, true, false, false));
        m.insert("j2-ultra", (8192, 8192, 15.0, 15.0, false, false, false, false, false));
        m.insert("j2-mid", (8192, 8192, 10.0, 10.0, false, false, false, false, false));
        m
    });
    
    let (max_input, max_output, pricing_input, pricing_output, supports_vision, supports_function_calling, supports_streaming, supports_embeddings, supports_thinking) = 
        map.get(model).copied().unwrap_or((8192, 8192, 10.0, 10.0, false, false, false, false, false));
    
    ModelInfo {
        provider_name: "ai21",
        name: Box::leak(model.to_string().into_boxed_str()),
        max_input_tokens: std::num::NonZeroU32::new(max_input),
        max_output_tokens: std::num::NonZeroU32::new(max_output),
        input_price: Some(pricing_input),
        output_price: Some(pricing_output),
        supports_vision,
        supports_function_calling,
        supports_streaming,
        supports_embeddings,
        requires_max_tokens: false,
        supports_thinking,
        optimal_thinking_budget: if supports_thinking { Some(25000) } else { None },
        system_prompt_prefix: None,
        real_name: None,
        model_type: None,
        patch: None,
        required_temperature: None,
    }
}