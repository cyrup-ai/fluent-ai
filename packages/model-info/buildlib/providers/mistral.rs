use anyhow::Result;

use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use http::{header, HeaderValue};
use super::super::codegen::CodeGenerator;
use super::{ModelData, ProviderBuilder, StandardModelsResponse};

/// Mistral provider implementation with dynamic API fetching
/// No fallback data - fails build if API unavailable
pub struct MistralProvider;

// Using StandardModelsResponse from mod.rs

impl ProviderBuilder for MistralProvider {
    fn provider_name(&self) -> &'static str {
        "mistral"
    }

    fn api_endpoint(&self) -> Option<&'static str> {
        Some("https://api.mistral.ai/v1/models")
    }

    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("MISTRAL_API_KEY")
    }

    fn fetch_models(&self) -> AsyncStream<ModelData> {
        let Some(api_key) = std::env::var("MISTRAL_API_KEY").ok() else {
            return AsyncStream::empty();
        };
        
        let auth_header = HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap();
        
        let response = Http3::json::<StandardModelsResponse>()
            .header(header::AUTHORIZATION, auth_header)
            .get("https://api.mistral.ai/v1/models")
            .collect::<StandardModelsResponse>()
            .into_iter()
            .next()
            .unwrap_or_default();
        
        AsyncStream::with_channel(move |sender| {
            for model in response.data {
            let id = model.id;
            let (context_length, input_price, output_price, supports_thinking, required_temp) = 
                match id.as_str() {
                    "mistral-large-latest" => (128000, 2.0, 6.0, false, None),
                    "mistral-large-2407" => (128000, 2.0, 6.0, false, None),
                    "mistral-large-2402" => (32000, 4.0, 12.0, false, None),
                    "mistral-medium-latest" => (32000, 2.7, 8.1, false, None),
                    "mistral-small-latest" => (32000, 1.0, 3.0, false, None),
                    "mistral-tiny" => (32000, 0.14, 0.42, false, None),
                    "mistral-embed" => (8192, 0.1, 0.0, false, None),
                    "codestral-latest" => (32000, 0.2, 0.6, false, None),
                    "codestral-2405" => (32000, 0.2, 0.6, false, None),
                    _ => {
                        if id.contains("embed") {
                            (8192, 0.1, 0.0, false, None)
                        } else if id.contains("large") {
                            (128000, 2.0, 6.0, false, None)
                        } else if id.contains("medium") {
                            (32000, 2.7, 8.1, false, None)
                        } else if id.contains("small") {
                            (32000, 1.0, 3.0, false, None)
                        } else if id.contains("codestral") {
                            (32000, 0.2, 0.6, false, None)
                        } else {
                            (32000, 1.0, 3.0, false, None)
                        }
                    }
                };
            
                let model_data = (id, context_length, input_price, output_price, supports_thinking, required_temp);
                let _ = sender.send(model_data);
            }
        })
    }

    fn static_models(&self) -> Option<Vec<ModelData>> {
        // Mistral supports /v1/models API endpoint - no static models needed
        None
    }

    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = CodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}
