use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use serde::Deserialize;
use std::env;

#[derive(Deserialize, Default)]
pub struct HuggingFaceModelInfo {
    #[serde(rename = "_id")]
    pub id: String,
    #[serde(rename = "modelId")]
    pub model_id: String,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Clone)]
pub struct HuggingFaceProvider;

impl ProviderTrait for HuggingFaceProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let key = env::var("HUGGINGFACE_API_KEY")
                .expect("HUGGINGFACE_API_KEY environment variable is required");
            
            let response = Http3::json()
                .api_key(&key)
                .get(&format!("https://huggingface.co/api/models/{}", model_name))
                .collect::<HuggingFaceModelInfo>();
                
            let model_info = adapt_huggingface_to_model_info(&response);
            let _ = sender.send(model_info);
        })
    }
}

fn adapt_huggingface_to_model_info(data: &HuggingFaceModelInfo) -> ModelInfo {
    // Determine context length based on model type and tags
    let max_context = if data.tags.iter().any(|tag| tag.contains("32k") || tag.contains("32768")) {
        32768
    } else if data.tags.iter().any(|tag| tag.contains("16k") || tag.contains("16384")) {
        16384
    } else if data.tags.iter().any(|tag| tag.contains("8k") || tag.contains("8192")) {
        8192
    } else if data.tags.iter().any(|tag| tag.contains("4k") || tag.contains("4096")) {
        4096
    } else {
        2048 // Default for HuggingFace models
    };
    
    // Check for reasoning/thinking capabilities
    let is_thinking = data.tags.iter().any(|tag| 
        tag.contains("reasoning") || 
        tag.contains("cot") || 
        tag.contains("chain-of-thought")
    );
    
    let required_temperature = if is_thinking { Some(1.0) } else { None };
    
    ModelInfo {
        name: data.model_id.clone(),
        max_context,
        pricing_input: 0.0,  // Many HF models are free
        pricing_output: 0.0,
        is_thinking,
        required_temperature,
    }
}