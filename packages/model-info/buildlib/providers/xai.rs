use super::{ModelData, ProviderBuilder, StandardModelsResponse};
use super::super::codegen::CodeGenerator;
use anyhow::{Context, Result};

use fluent_ai_async::AsyncStream;
use fluent_ai_http3::{Http3, HttpStreamExt};
use http::{header, HeaderValue};

/// X.AI provider implementation with dynamic API fetching
/// Uses the official X.AI API endpoint as documented at https://docs.x.ai/docs/api-reference#list-models
pub struct XaiProvider;

// Using StandardModelsResponse from mod.rs

impl ProviderBuilder for XaiProvider {
    fn provider_name(&self) -> &'static str {
        "xai"
    }
    
    fn api_endpoint(&self) -> Option<&'static str> {
        Some("https://api.x.ai/v1/models")
    }
    
    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("XAI_API_KEY")
    }
    
    fn fetch_models(&self) -> AsyncStream<ModelData> {
        let Some(api_key) = std::env::var("XAI_API_KEY").ok() else {
            return AsyncStream::empty();
        };
        
        let auth_header = HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap();
        
        let response = Http3::json::<StandardModelsResponse>()
            .header(header::AUTHORIZATION, auth_header)
            .get("https://api.x.ai/v1/models")
            .collect::<StandardModelsResponse>()
            .into_iter()
            .next()
            .unwrap_or_default();
        
        AsyncStream::with_channel(move |sender| {
            for model in response.data {
                let id = model.id;
                
                // Parse X.AI model specifications based on model ID
                // All X.AI models support thinking and have required temperature
                let (context_length, input_price, output_price, supports_thinking, required_temp) = match id.as_str() {
                    "grok-beta" => (131072, 5.0, 15.0, true, Some(1.0)),
                    "grok-2" => (131072, 2.0, 10.0, true, Some(1.0)),
                    "grok-2-mini" => (131072, 0.3, 0.5, true, None),
                    _ => {
                        // For unknown X.AI models, apply consistent defaults
                        // X.AI models typically support thinking
                        if id.contains("mini") {
                            (131072, 0.3, 0.5, true, None)
                        } else if id.starts_with("grok") {
                            // Standard grok model defaults
                            (131072, 3.0, 15.0, true, Some(1.0))
                        } else {
                            // Conservative defaults for new models
                            (131072, 3.0, 15.0, true, Some(1.0))
                        }
                    }
                };
                
                let model_data = (id, context_length, input_price, output_price, supports_thinking, required_temp);
                let _ = sender.send(model_data);
            }
        })
    }
    
    fn static_models(&self) -> Option<Vec<ModelData>> {
        // XAI supports /v1/models API endpoint - no static models needed
        None
    }
    
    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = CodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}