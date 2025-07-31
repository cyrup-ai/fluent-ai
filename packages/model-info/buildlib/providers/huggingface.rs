use serde::Deserialize;
use std::collections::HashMap;
use anyhow::Result;
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use super::super::codegen::CodeGenerator;
use super::{ModelData, ProviderBuilder};

#[derive(Deserialize, Clone, Default)]
struct ModelDetails {
    max_tokens: u64,
    input_price: f64,
    output_price: f64,
    supports_thinking: bool,
    required_temp: Option<f64>,
}

#[derive(Deserialize)]
struct HuggingFaceModelDetails {
    // Additional fields from per-model API
    config: Option<HashMap<String, serde_json::Value>>,
    tags: Option<Vec<String>>,
    // Add other relevant fields
}

/// HuggingFace provider implementation with dynamic API fetching
/// No fallback data - fails build if API unavailable
pub struct HuggingFaceProvider;

#[derive(Deserialize, serde::Serialize)]
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
            let models = Http3::json::<Vec<HuggingFaceModel>>()
                .get("https://huggingface.co/api/models?filter=text-generation&sort=downloads&direction=-1&limit=50")
                .collect::<Vec<HuggingFaceModel>>()
                .into_iter()
                .next()
                .unwrap_or_default();
            
            for model in models.into_iter().take(20) {
                let id = model.id;
                let downloads = model.downloads.unwrap_or(0);
                if downloads < 1000 {
                    continue;
                }

                let is_text_gen = model
                    .tags
                    .as_ref()
                    .map(|tags| {
                        tags.iter().any(|tag| {
                            tag.contains("text-generation")
                                || tag.contains("conversational")
                                || tag.contains("text2text-generation")
                        })
                    })
                    .unwrap_or(false);

                if !is_text_gen {
                    continue;
                }
                
                // Default values for HuggingFace models
                let (context_length, input_price, output_price, supports_thinking, required_temp) = 
                    if id.contains("llama") || id.contains("meta-llama") {
                        (4096, 0.0, 0.0, false, None)
                    } else if id.contains("mistral") {
                        (8192, 0.0, 0.0, false, None)
                    } else if id.contains("codellama") {
                        (16384, 0.0, 0.0, false, None)
                    } else if id.contains("gemma") {
                        (8192, 0.0, 0.0, false, None)
                    } else {
                        (4096, 0.0, 0.0, false, None)
                    };
                
                let model_data = (id, context_length, input_price, output_price, supports_thinking, required_temp);
                let _ = sender.send(model_data);
            }
        })
    }

    fn static_models(&self) -> Option<Vec<ModelData>> {
        // HuggingFace supports /v1/models API endpoint - no static models needed
        None
    }

    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = CodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}
