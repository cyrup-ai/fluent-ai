use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use serde::Deserialize;

#[derive(Deserialize, Default)]
pub struct MistralModelsResponse {
    pub data: Vec<MistralModelData>,
}

#[derive(Deserialize, Default)]
pub struct MistralModelData {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MistralProvider;

impl ProviderTrait for MistralProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_mistral_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }
    
    fn list_models(&self) -> AsyncStream<ModelInfo> {
        use fluent_ai_http3::{Http3, HttpStreamExt};
        use std::env;
        use serde::{Deserialize, Serialize};
        
        #[derive(Deserialize, Serialize, Debug)]
        struct MistralModel {
            id: String,
            object: String,
            created: Option<u64>,
            owned_by: Option<String>,
        }
        
        #[derive(Deserialize, Serialize, Debug, Default)]
        struct MistralModelsResponse {
            object: String,
            data: Vec<MistralModel>,
        }
        
        AsyncStream::with_channel(move |sender| {
            let response = if let Ok(api_key) = env::var("MISTRAL_API_KEY") {
                Http3::json()
                    .bearer_auth(&api_key)
                    .get("https://api.mistral.ai/v1/models")
                    .collect::<MistralModelsResponse>()
            } else {
                Http3::json()
                    .get("https://api.mistral.ai/v1/models")
                    .collect::<MistralModelsResponse>()
            };
            
            if let Some(models_response) = response.into_iter().next() {
                for model in models_response.data {
                    let model_info = adapt_mistral_to_model_info(&model.id);
                    if sender.send(model_info).is_err() {
                        break;
                    }
                }
            }
        })
    }
    
    fn provider_name(&self) -> &'static str {
        "mistral"
    }
}

// Type alias for complex provider data tuple to improve readability
type ProviderModelData = (u32, u32, f64, f64, bool, bool, bool, bool, bool);

fn adapt_mistral_to_model_info(model: &str) -> ModelInfo {
    use std::sync::OnceLock;
    use hashbrown::HashMap;
    
    static MAP: OnceLock<HashMap<&'static str, ProviderModelData>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // (max_input, max_output, input_price, output_price, vision, function_calling, streaming, embeddings, thinking)
        m.insert("mistral-large-2407", (128000, 32000, 8.0, 24.0, false, true, true, false, false));
        m.insert("mistral-large-2312", (32000, 8000, 3.0, 9.0, false, true, true, false, false));
        m.insert("mistral-small-2409", (128000, 32000, 0.2, 0.6, false, true, true, false, false));
        m.insert("mistral-nemo-2407", (128000, 32000, 0.2, 0.6, false, true, true, false, false));
        m.insert("open-mistral-nemo", (128000, 32000, 0.3, 1.0, false, true, true, false, false));
        m.insert("codestral-2405", (32000, 8000, 0.8, 2.5, false, false, true, false, false));
        m.insert("mistral-embed", (8192, 0, 0.1, 0.0, false, false, false, true, false));
        m.insert("mistral-tiny", (32000, 8000, 0.25, 0.75, false, false, true, false, false));
        m.insert("mistral-small", (32000, 8000, 2.0, 6.0, false, true, true, false, false));
        m.insert("mistral-medium", (32000, 8000, 8.0, 24.0, false, true, true, false, false));
        m.insert("mistral-large", (32000, 8000, 20.0, 60.0, false, true, true, false, false));
        // Current generation models with latest API endpoints
        m.insert("mistral-large-latest", (128000, 32000, 8.0, 24.0, false, true, true, false, false));
        m.insert("codestral-latest", (32000, 8000, 0.8, 2.5, false, false, true, false, false));
        m
    });
    
    let (max_input, max_output, pricing_input, pricing_output, supports_vision, supports_function_calling, _supports_streaming, supports_embeddings, supports_thinking) = 
        map.get(model).copied().unwrap_or((32000, 8000, 0.0, 0.0, false, false, true, false, false));
    
    ModelInfo {
        // Core identification
        provider_name: "mistral",
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
        model_type: if supports_embeddings { Some("embedding".to_string()) } else { None },
        patch: None,
        required_temperature: None,
    }
}