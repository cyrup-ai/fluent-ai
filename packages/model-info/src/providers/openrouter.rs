use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use serde::Deserialize;

#[derive(Deserialize, Default)]
pub struct OpenRouterModelsResponse {
    pub data: Vec<OpenRouterModelData>,
}

#[derive(Deserialize, Default)]
pub struct OpenRouterModelData {
    pub id: String,
    pub name: String,
    pub created: u64,
    pub context_length: Option<u64>,
    pub pricing: OpenRouterPricingData,
}

#[derive(Deserialize, Default)]
pub struct OpenRouterPricingData {
    pub prompt: String,
    pub completion: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpenRouterProvider;

impl ProviderTrait for OpenRouterProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_openrouter_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }
    
    fn list_models(&self) -> AsyncStream<ModelInfo> {
        use crate::generated_models::OpenRouterModel as OpenRouter;
        use crate::common::Model;
        
        AsyncStream::with_channel(move |sender| {
            let models = vec![
                OpenRouter::OpenaiGpt4o,
                OpenRouter::AnthropicClaude35Sonnet,
                OpenRouter::GoogleGeminiPro15,
            ];
            
            for model in models {
                let model_info = adapt_openrouter_to_model_info(model.name());
                let _ = sender.send(model_info);
            }
        })
    }
    
    fn provider_name(&self) -> &'static str {
        "openrouter"
    }
}

// Type alias for complex provider data tuple to improve readability
type ProviderModelData = (u32, u32, f64, f64, bool, bool, bool, bool, bool);

fn adapt_openrouter_to_model_info(model: &str) -> ModelInfo {
    use std::sync::OnceLock;
    use hashbrown::HashMap;
    
    static MAP: OnceLock<HashMap<&'static str, ProviderModelData>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // (max_input, max_output, input_price, output_price, vision, function_calling, streaming, embeddings, thinking)
        m.insert("openai/gpt-4o", (128000, 32000, 5.0, 15.0, true, true, true, false, false));
        m.insert("anthropic/claude-3.5-sonnet", (200000, 50000, 3.0, 15.0, true, true, true, false, false));
        m.insert("google/gemini-pro-1.5", (1000000, 250000, 0.5, 1.5, true, true, true, false, false));
        m
    });
    
    let (max_input, max_output, pricing_input, pricing_output, supports_vision, supports_function_calling, _supports_streaming, supports_embeddings, supports_thinking) = 
        map.get(model).copied().unwrap_or((128000, 32000, 0.0, 0.0, false, false, true, false, false));
    
    ModelInfo {
        // Core identification
        provider_name: "openrouter",
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
        optimal_thinking_budget: if supports_thinking { Some(75000) } else { None },
        system_prompt_prefix: None,
        real_name: None,
        model_type: None,
        patch: None,
        required_temperature: if supports_thinking { Some(1.0) } else { None },
    }
}