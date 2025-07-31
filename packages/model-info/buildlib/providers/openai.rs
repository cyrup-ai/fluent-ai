use anyhow::Result;
use serde::Deserialize;
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use http::{header, HeaderValue};

#[derive(Deserialize, Clone, Default)]
struct ModelDetails {
    max_tokens: u64,
    input_price: f64,
    output_price: f64,
    supports_thinking: bool,
    required_temp: Option<f64>,
}

#[derive(Deserialize)]
struct OpenAiModelDetails {
    // Fields from per-model API, if any (e.g., hypothetical)
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(default)]
    input_price: Option<f64>,
    #[serde(default)]
    output_price: Option<f64>,
    #[serde(default)]
    supports_thinking: Option<bool>,
    #[serde(default)]
    required_temp: Option<f64>,
}

use super::super::codegen::CodeGenerator;
use super::{ModelData, ProviderBuilder, StandardModelsResponse};

/// OpenAI provider implementation with dynamic API fetching
/// No fallback data - fails build if API unavailable
pub struct OpenAiProvider;

// Using StandardModelsResponse from mod.rs

impl ProviderBuilder for OpenAiProvider {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn api_endpoint(&self) -> Option<&'static str> {
        Some("https://api.openai.com/v1/models")
    }

    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("OPENAI_API_KEY")
    }

    fn fetch_models(&self) -> AsyncStream<ModelData> {
        AsyncStream::with_channel(move |sender| {
            // /v1/models endpoint is PUBLIC - no auth needed
            let response_data = Http3::json::<StandardModelsResponse>()
                .get("https://api.openai.com/v1/models")
                .collect::<StandardModelsResponse>()
                .into_iter()
                .next()
                .unwrap_or_default();
            
            // Process the collected response data using ONLY what comes from the API
            for model in response_data.data {
                let id = model.id;
                if id.contains("embedding") || id.contains("ft:") {
                    continue;
                }
                
                // Use a reasonable default context length since OpenAI API doesn't always provide it
                let context_length = if id.contains("gpt-4") {
                    128000 // GPT-4 models typically have 128k context
                } else if id.contains("gpt-3.5") {
                    16384 // GPT-3.5 models typically have 16k context  
                } else {
                    4096 // Default fallback
                };
                
                // Determine if model supports thinking based on model name
                let supports_thinking = id.to_lowercase().contains("thinking");
                
                // Set default temperature based on thinking support (user can override at runtime)
                let required_temp = if supports_thinking {
                    Some(1.0) // Thinking models default to 1.0
                } else {
                    Some(0.0) // Non-thinking models default to 0.0
                };
                
                let model_data = (
                    id,
                    context_length,
                    0.0, // Pricing not available in /v1/models endpoint
                    0.0, // Pricing not available in /v1/models endpoint  
                    supports_thinking,
                    required_temp,
                );
                let _ = sender.send(model_data);
            }
        })
    }

    fn static_models(&self) -> Option<Vec<ModelData>> {
        // OpenAI supports /v1/models API endpoint - no static models needed
        None
    }

    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = CodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}
