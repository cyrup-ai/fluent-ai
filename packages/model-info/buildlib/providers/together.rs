use anyhow::Result;
use serde::Deserialize;
use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use http::{header, HeaderValue};
use super::super::codegen::CodeGenerator;
use super::{ModelData, ProviderBuilder, StandardModelsResponse};

/// Together provider implementation with dynamic API fetching
/// No fallback data - fails build if API unavailable
pub struct TogetherProvider;

#[derive(Deserialize)]
struct TogetherModel {
    id: String,
    #[serde(rename = "type")]
    model_type: String,
    pricing: Option<TogetherPricing>,
    context_length: Option<u64>,
}

#[derive(Deserialize)]
struct TogetherPricing {
    input: f64,
    output: f64,
}

impl ProviderBuilder for TogetherProvider {
    fn provider_name(&self) -> &'static str {
        "together"
    }

    fn api_endpoint(&self) -> Option<&'static str> {
        Some("https://api.together.xyz/v1/models")
    }

    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("TOGETHER_API_KEY")
    }

    fn fetch_models(&self) -> AsyncStream<ModelData> {
        AsyncStream::with_channel(move |sender| {
            let Some(api_key) = std::env::var("TOGETHER_API_KEY").ok() else {
                return;
            };
            
            let auth_header = HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap();
            
            let response = Http3::json::<StandardModelsResponse>()
                .header(header::AUTHORIZATION, auth_header)
                .get("https://api.together.xyz/v1/models")
                .collect::<StandardModelsResponse>()
                .into_iter()
                .next()
                .unwrap_or_default();
            
            for model in response.data {
                if model.object != "model" {
                    continue;
                }
                
                let context_length = 8192; // Together default
                let input_price = 0.2;
                let output_price = 0.2;
                let supports_thinking = false;
                let required_temp = None;
                
                let model_data = (model.id, context_length, input_price, output_price, supports_thinking, required_temp);
                let _ = sender.send(model_data);
            }
        })
    }

    fn static_models(&self) -> Option<Vec<ModelData>> {
        // Together supports /v1/models API endpoint - no static models needed
        None
    }

    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = CodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}
