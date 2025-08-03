use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use serde::Deserialize;

#[derive(Deserialize, Default)]
pub struct HuggingFaceModelInfo {
    #[serde(rename = "_id")]
    pub id: String,
    #[serde(rename = "modelId")]
    pub model_id: String,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HuggingFaceProvider;

impl ProviderTrait for HuggingFaceProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_huggingface_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }
    
    fn list_models(&self) -> AsyncStream<ModelInfo> {
        use crate::generated_models::HuggingFaceModel as HuggingFace;
        use crate::common::Model;
        
        use fluent_ai_http3::{Http3, HttpStreamExt};
        use std::env;
        use serde::{Deserialize, Serialize};
        
        #[derive(Deserialize, Serialize, Debug)]
        struct HuggingFaceModel {
            id: String,
            #[serde(rename = "modelId")]
            model_id: Option<String>,
            pipeline_tag: Option<String>,
            library_name: Option<String>,
        }
        
        AsyncStream::with_channel(move |sender| {
            let response = if let Ok(api_key) = env::var("HUGGINGFACE_API_KEY") {
                Http3::json::<Vec<HuggingFaceModel>>()
                    .bearer_auth(&api_key)
                    .get("https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1&limit=50")
                    .collect::<Vec<HuggingFaceModel>>()
            } else {
                Http3::json::<Vec<HuggingFaceModel>>()
                    .get("https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1&limit=50")
                    .collect::<Vec<HuggingFaceModel>>()
            };
            
            if let Some(models) = response.into_iter().next() {
                for model in models {
                    let model_info = adapt_huggingface_to_model_info(&model.id);
                    if sender.send(model_info).is_err() {
                        break;
                    }
                }
            }
        })
    }
    
    fn provider_name(&self) -> &'static str {
        "huggingface"
    }
}

// Type alias for complex provider data tuple to improve readability
type ProviderModelData = (u32, u32, f64, f64, bool, bool, bool, bool, bool);

fn adapt_huggingface_to_model_info(model: &str) -> ModelInfo {
    use std::sync::OnceLock;
    use hashbrown::HashMap;
    
    static MAP: OnceLock<HashMap<&'static str, ProviderModelData>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // (max_input, max_output, input_price, output_price, vision, function_calling, streaming, embeddings, thinking)
        m.insert("meta-llama/Meta-Llama-3-8B-Instruct", (8192, 2048, 0.0, 0.0, false, false, true, false, false));
        m.insert("mistralai/Mistral-7B-Instruct-v0.3", (32768, 8192, 0.0, 0.0, false, false, true, false, false));
        m.insert("google/gemma-2-9b-it", (8192, 2048, 0.0, 0.0, false, false, true, false, false));
        m
    });
    
    let (max_input, max_output, pricing_input, pricing_output, supports_vision, supports_function_calling, _supports_streaming, supports_embeddings, supports_thinking) = 
        map.get(model).copied().unwrap_or((8192, 2048, 0.0, 0.0, false, false, true, false, false));
    
    ModelInfo {
        // Core identification
        provider_name: "huggingface",
        name: Box::leak(model.to_string().into_boxed_str()),
        
        // Token limits
        max_input_tokens: std::num::NonZeroU32::new(max_input),
        max_output_tokens: std::num::NonZeroU32::new(max_output),
        
        // Pricing (per 1M tokens) - many HF models are free
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