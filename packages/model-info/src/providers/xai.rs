use crate::common::{ModelInfo, ProviderTrait};
use hashbrown::HashMap;
use fluent_ai_async::AsyncStream;
use serde::Deserialize;
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

#[derive(Clone, Debug, PartialEq)]
pub struct XaiProvider;

impl ProviderTrait for XaiProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_xai_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }
    
    fn list_models(&self) -> AsyncStream<ModelInfo> {
        use crate::generated_models::XaiModel as Xai;
        use crate::common::Model;
        
        AsyncStream::with_channel(move |sender| {
            let models = vec![
                Xai::Grok4,
                Xai::Grok3,
                Xai::Grok3Mini,
            ];
            
            for model in models {
                let model_info = adapt_xai_to_model_info(model.name());
                let _ = sender.send(model_info);
            }
        })
    }
    
    fn provider_name(&self) -> &'static str {
        "xai"
    }
}

fn adapt_xai_to_model_info(model: &str) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, (u32, u32, f64, f64, bool, bool, bool, bool, bool, Option<f64>)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // (max_input, max_output, input_price, output_price, vision, function_calling, streaming, embeddings, thinking, required_temp)
        m.insert("grok-4", (256000, 60000, 3.0, 15.0, false, true, true, false, true, Some(1.0)));
        m.insert("grok-3", (131072, 30000, 3.0, 15.0, false, true, true, false, true, Some(1.0)));
        m.insert("grok-3-mini", (131072, 30000, 0.3, 0.5, false, true, true, false, true, None));
        m.insert("grok-beta", (131072, 30000, 5.0, 15.0, false, true, true, false, true, Some(1.0)));
        m.insert("grok-2", (131072, 30000, 2.0, 10.0, false, true, true, false, true, Some(1.0)));
        m
    });
    
    let (max_input, max_output, pricing_input, pricing_output, supports_vision, supports_function_calling, supports_streaming, supports_embeddings, supports_thinking, required_temperature) = 
        map.get(model).copied().unwrap_or((131072, 30000, 0.0, 0.0, false, true, true, false, true, None));
    
    ModelInfo {
        // Core identification
        provider_name: "xai",
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
        supports_streaming,
        supports_embeddings,
        requires_max_tokens: false,
        supports_thinking,
        
        // Advanced features
        optimal_thinking_budget: if supports_thinking { Some(100000) } else { None },
        system_prompt_prefix: None,
        real_name: None,
        model_type: None,
        patch: None,
        required_temperature,
    }
}