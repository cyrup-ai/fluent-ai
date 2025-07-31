use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use hashbrown::HashMap;
use std::sync::OnceLock;

#[derive(Clone)]
pub struct BedrockProvider;

impl ProviderTrait for BedrockProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_bedrock_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }
    
    fn list_models(&self) -> AsyncStream<ModelInfo> {
        AsyncStream::with_channel(move |sender| {
            let models = vec![
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "anthropic.claude-3-opus-20240229-v1:0", 
                "anthropic.claude-3-haiku-20240307-v1:0",
            ];
            
            for model in models {
                let model_info = adapt_bedrock_to_model_info(model);
                let _ = sender.send(model_info);
            }
        })
    }
    
    fn provider_name(&self) -> &'static str {
        "bedrock"
    }
}

fn adapt_bedrock_to_model_info(model: &str) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u32, u32, f64, f64, bool, bool, bool, bool, bool)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // (max_input, max_output, input_price, output_price, vision, function_calling, streaming, embeddings, thinking)
        m.insert("anthropic.claude-3-5-sonnet-20240620-v1:0", (200000, 50000, 3.0, 15.0, true, true, true, false, false));
        m.insert("anthropic.claude-3-opus-20240229-v1:0", (200000, 50000, 15.0, 75.0, true, true, true, false, false));
        m.insert("anthropic.claude-3-haiku-20240307-v1:0", (200000, 50000, 0.25, 1.25, true, true, true, false, false));
        m
    });
    
    let (max_input, max_output, pricing_input, pricing_output, supports_vision, supports_function_calling, _supports_streaming, supports_embeddings, supports_thinking) = 
        map.get(model).copied().unwrap_or((200000, 4096, 3.0, 15.0, false, true, true, false, false));
    
    ModelInfo {
        provider_name: "bedrock",
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
        optimal_thinking_budget: if supports_thinking { Some(60000) } else { None },
        system_prompt_prefix: None,
        real_name: None,
        model_type: None,
        patch: None,
        required_temperature: None,
    }
}