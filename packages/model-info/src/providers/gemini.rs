use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use hashbrown::HashMap;
use std::sync::OnceLock;

#[derive(Clone)]
pub struct GeminiProvider;

impl ProviderTrait for GeminiProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            Box::pin(async move {
                let model_info = adapt_gemini_to_model_info(&model_name);
                let _ = sender.send(model_info).await;
                Ok(())
            })
        })
    }
    
    fn list_models(&self) -> AsyncStream<ModelInfo> {
        AsyncStream::with_channel(move |sender| {
            Box::pin(async move {
                let models = vec![
                    "gemini-1.5-pro-latest",
                    "gemini-1.5-flash-latest",
                    "gemini-1.0-pro",
                    "gemini-pro-vision",
                ];
                
                for model in models {
                    let model_info = adapt_gemini_to_model_info(model);
                    let _ = sender.send(model_info).await;
                }
                Ok(())
            })
        })
    }
    
    fn provider_name(&self) -> &'static str {
        "gemini"
    }
}

fn adapt_gemini_to_model_info(model: &str) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u32, u32, f64, f64, bool, bool, bool, bool, bool)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // (max_input, max_output, input_price, output_price, vision, function_calling, streaming, embeddings, thinking)
        m.insert("gemini-1.5-pro-latest", (2097152, 8192, 1.25, 5.0, true, true, true, false, false));
        m.insert("gemini-1.5-flash-latest", (1048576, 8192, 0.075, 0.3, true, true, true, false, false));
        m.insert("gemini-1.0-pro", (32768, 2048, 0.5, 1.5, false, true, true, false, false));
        m.insert("gemini-pro-vision", (16384, 2048, 0.5, 1.5, true, true, true, false, false));
        m
    });
    
    let (max_input, max_output, pricing_input, pricing_output, supports_vision, supports_function_calling, supports_streaming, supports_embeddings, supports_thinking) = 
        map.get(model).copied().unwrap_or((32768, 2048, 0.5, 1.5, false, true, true, false, false));
    
    ModelInfo {
        provider_name: "gemini",
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
        optimal_thinking_budget: if supports_thinking { Some(40000) } else { None },
        system_prompt_prefix: None,
        real_name: None,
        model_type: None,
        patch: None,
        required_temperature: None,
    }
}