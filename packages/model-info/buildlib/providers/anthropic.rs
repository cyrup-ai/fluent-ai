use super::{ModelData, ProviderBuilder};
use super::super::codegen::SynCodeGenerator;
use fluent_ai_async::AsyncStream;
use anyhow::Result;
use serde::Deserialize;

/// Anthropic provider implementation with legitimate hardcoded data
/// This is the ONLY provider that legitimately uses static data because
/// Anthropic/Claude does not provide a public models list API endpoint
pub struct AnthropicProvider;

// Placeholder type since Anthropic doesn't have an API endpoint
#[derive(Deserialize)]
struct AnthropicModelsResponse;

impl ProviderBuilder for AnthropicProvider {
    fn provider_name(&self) -> &'static str {
        "anthropic"
    }
    
    fn api_endpoint(&self) -> Option<&'static str> {
        // Anthropic does not provide a public models list API
        None
    }
    
    fn api_key_env_var(&self) -> Option<&'static str> {
        // No API key needed since there's no API endpoint
        None
    }
    
    fn fetch_models(&self) -> AsyncStream<ModelData> {
        // Anthropic doesn't have a models API endpoint - return empty stream
        AsyncStream::empty()
    }
    
    fn static_models(&self) -> Option<Vec<ModelData>> {
        // ONLY legitimate use of static model data - Anthropic has no public /v1/models API
        // User-specified claude-4 models only
        Some(vec![
            (
                "claude-4-sonnet".to_string(),
                200000,
                3.0,
                15.0,
                false,
                None,
            ),
            (
                "claude-4-sonnet-thinking".to_string(),
                200000,
                3.0,
                15.0,
                true,
                Some(1.0),
            ),
            (
                "claude-4-opus".to_string(),
                200000,
                15.0,
                75.0,
                false,
                None,
            ),
            (
                "claude-4-opus-thinking".to_string(),
                200000,
                15.0,
                75.0,
                true,
                Some(1.0),
            ),
        ])
    }
    
    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        let codegen = SynCodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}