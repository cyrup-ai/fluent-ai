use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use serde::Deserialize;

#[derive(Deserialize, Default)]
pub struct TogetherModelData {
    pub id: String,
    pub context_length: u64,
    pub pricing: TogetherPricingData,
    pub description: String,
}

#[derive(Deserialize, Default)]
pub struct TogetherPricingData {
    pub input: f64,
    pub output: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TogetherProvider;

impl ProviderTrait for TogetherProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_together_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }
    
    fn list_models(&self) -> AsyncStream<ModelInfo> {
        use crate::generated_models::TogetherModel as Together;
        use crate::common::Model;
        
        AsyncStream::with_channel(move |sender| {
            let models = vec![
                Together::MetaLlamaLlama38bChatHf,
                Together::MistralaiMixtral8x7bInstructV01,
                Together::TogethercomputerCodellama34bInstruct,
            ];
            
            for model in models {
                let model_info = adapt_together_to_model_info(model.name());
                let _ = sender.send(model_info);
            }
        })
    }
    
    fn provider_name(&self) -> &'static str {
        "together"
    }
}

// Type alias for complex provider data tuple to improve readability
type ProviderModelData = (u32, u32, f64, f64, bool, bool, bool, bool, bool);

fn adapt_together_to_model_info(model: &str) -> ModelInfo {
    use std::sync::OnceLock;
    use hashbrown::HashMap;
    
    static MAP: OnceLock<HashMap<&'static str, ProviderModelData>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // (max_input, max_output, input_price, output_price, vision, function_calling, streaming, embeddings, thinking)
        m.insert("meta-llama/Llama-3-8b-chat-hf", (8192, 2048, 0.2, 0.2, false, false, true, false, false));
        m.insert("mistralai/Mixtral-8x7B-Instruct-v0.1", (32768, 8192, 0.27, 0.27, false, true, true, false, false));
        m.insert("togethercomputer/CodeLlama-34b-Instruct", (16384, 4096, 0.5, 0.5, false, false, true, false, false));
        m
    });
    
    let (max_input, max_output, pricing_input, pricing_output, supports_vision, supports_function_calling, _supports_streaming, supports_embeddings, supports_thinking) = 
        map.get(model).copied().unwrap_or((8192, 2048, 0.0, 0.0, false, false, true, false, false));
    
    ModelInfo {
        // Core identification
        provider_name: "together",
        name: Box::leak(model.to_string().into_boxed_str()),
        
        // Token limits
        max_input_tokens: std::num::NonZeroU32::new(max_input),
        max_output_tokens: std::num::NonZeroU32::new(max_output),
        
        // Pricing (per 1M tokens)
        input_price: Some(pricing_input),
        output_price: Some(pricing_output),
        
        // Capability flags
        supports_vision,
        supports_function_calling,
        supports_embeddings,
        requires_max_tokens: false,
        supports_thinking,
        
        // Advanced features
        optimal_thinking_budget: if supports_thinking { Some(50000) } else { None },
        system_prompt_prefix: None,
        real_name: None,
        model_type: None,
        patch: None,
        required_temperature: None,
    }
}