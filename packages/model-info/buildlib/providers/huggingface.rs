use serde::Deserialize;
use anyhow::Result;
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use super::super::codegen::SynCodeGenerator;
use super::{ModelData, ProviderBuilder};


/// HuggingFace provider implementation with dynamic API fetching
/// No fallback data - fails build if API unavailable
pub struct HuggingFaceProvider;

#[derive(Deserialize, serde::Serialize, Default)]
struct HuggingFaceModel {
    id: String,
    #[serde(rename = "modelId")]
    model_id: Option<String>,
    tags: Option<Vec<String>>,
    downloads: Option<u64>,
    #[serde(rename = "createdAt")]
    created_at: Option<String>,
}

impl ProviderBuilder for HuggingFaceProvider {
    fn provider_name(&self) -> &'static str {
        "huggingface"
    }

    fn api_endpoint(&self) -> Option<&'static str> {
        Some(
            "https://huggingface.co/api/models?filter=text-generation&sort=downloads&direction=-1&limit=50",
        )
    }

    fn api_key_env_var(&self) -> Option<&'static str> {
        // HuggingFace API is public for model listing
        None
    }

    fn fetch_models(&self) -> AsyncStream<ModelData> {
        AsyncStream::with_channel(move |sender| {
            let url = "https://huggingface.co/api/models?filter=text-generation&sort=downloads&direction=-1&limit=50";
            let model_arrays: Vec<Vec<HuggingFaceModel>> = Http3::json::<Vec<HuggingFaceModel>>()
                .get(url)
                .collect::<Vec<HuggingFaceModel>>();
            
            for models in model_arrays {
                for model in models {
                    let model_data = huggingface_model_to_data(&model);
                    if sender.send(model_data).is_err() {
                        return;
                    }
                }
            }
        })
    }

    fn static_models(&self) -> Option<Vec<ModelData>> {
        // Fallback static models for build resilience when API unavailable
        Some(vec![
            ("meta-llama/Llama-2-70b-chat-hf".to_string(), 4096, 0.0, 0.0, false, None),
            ("meta-llama/Llama-2-13b-chat-hf".to_string(), 4096, 0.0, 0.0, false, None),
            ("meta-llama/Llama-2-7b-chat-hf".to_string(), 4096, 0.0, 0.0, false, None),
            ("mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(), 32768, 0.0, 0.0, false, None),
            ("mistralai/Mistral-7B-Instruct-v0.1".to_string(), 8192, 0.0, 0.0, false, None),
            ("codellama/CodeLlama-34b-Instruct-hf".to_string(), 16384, 0.0, 0.0, false, None),
            ("codellama/CodeLlama-13b-Instruct-hf".to_string(), 16384, 0.0, 0.0, false, None),
            ("codellama/CodeLlama-7b-Instruct-hf".to_string(), 16384, 0.0, 0.0, false, None),
            ("google/gemma-7b-it".to_string(), 8192, 0.0, 0.0, false, None),
            ("google/gemma-2b-it".to_string(), 8192, 0.0, 0.0, false, None),
        ])
    }

    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = SynCodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}

/// Convert HuggingFace model to ModelData with appropriate context and capabilities
/// HuggingFace models are typically free/open-source with pricing of 0.0
fn huggingface_model_to_data(model: &HuggingFaceModel) -> ModelData {
    // Determine context length based on model name/tags
    let context_length = match model.id.as_str() {
        id if id.contains("llama-2") => 4096,
        id if id.contains("mixtral") => 32768,
        id if id.contains("codellama") => 16384,
        id if id.contains("mistral") => 8192,
        id if id.contains("gemma") => 8192,
        _ => 4096, // Default context length
    };
    
    // HuggingFace models are typically free/open-source (pricing 0.0)
    // No thinking models on HuggingFace currently
    (model.id.clone(), context_length, 0.0, 0.0, false, None)
}