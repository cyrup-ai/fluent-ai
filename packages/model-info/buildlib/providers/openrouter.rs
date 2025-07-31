use super::{ModelData, ProviderBuilder, StandardModelsResponse};
use super::super::codegen::CodeGenerator;
use anyhow::{Context, Result};
use serde::Deserialize;
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use http::{header, HeaderValue};

/// OpenRouter provider implementation with dynamic API fetching  
/// No fallback data - fails build if API unavailable
pub struct OpenRouterProvider;

#[derive(Deserialize, serde::Serialize, Default)]
struct OpenRouterModelsResponse {
    data: Vec<OpenRouterModel>,
}

#[derive(Deserialize, serde::Serialize)]
struct OpenRouterModel {
    id: String,
    name: String,
    context_length: Option<u64>,
    pricing: OpenRouterPricing,
}

#[derive(Deserialize, serde::Serialize)]
struct OpenRouterPricing {
    prompt: String,
    completion: String,
}

impl ProviderBuilder for OpenRouterProvider {
    fn provider_name(&self) -> &'static str {
        "openrouter"
    }
    
    fn api_endpoint(&self) -> Option<&'static str> {
        Some("https://openrouter.ai/api/v1/models")
    }
    
    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("OPENROUTER_API_KEY")
    }
    
    fn fetch_models(&self) -> AsyncStream<ModelData> {
        AsyncStream::with_channel(move |sender| {
            let Some(api_key) = std::env::var("OPENROUTER_API_KEY").ok() else {
                return;
            };
            
            let auth_header = HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap();
            
            let response = Http3::json::<OpenRouterModelsResponse>()
                .header(header::AUTHORIZATION, auth_header)
                .get("https://openrouter.ai/api/v1/models")
                .collect::<OpenRouterModelsResponse>()
                .into_iter()
                .next()
                .unwrap_or_default();
            
            for model in response.data {
                let id = model.id;
                let context_length = model.context_length.unwrap_or(8192);
                
                // Parse pricing strings (format: "0.000001" per token)
                let input_price = model.pricing.prompt.parse::<f64>()
                    .unwrap_or(0.0) * 1_000_000.0; // Convert to per-million tokens
                let output_price = model.pricing.completion.parse::<f64>()
                    .unwrap_or(0.0) * 1_000_000.0; // Convert to per-million tokens
                
                // Determine if model supports thinking based on ID
                let supports_thinking = id.contains("o1") || id.contains("reasoning");
                let required_temp = if supports_thinking { Some(1.0) } else { None };
                
                let model_data = (id, context_length, input_price, output_price, supports_thinking, required_temp);
                let _ = sender.send(model_data);
            }
        })
    }
    
    fn static_models(&self) -> Option<Vec<ModelData>> {
        // OpenRouter supports /v1/models API endpoint - no static models needed
        None
    }
    
    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = CodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}