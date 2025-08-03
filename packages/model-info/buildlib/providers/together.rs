use std::env;
use fluent_ai_http3::{Http3, HttpStreamExt};
use super::{ModelData, ProviderBuilder, TogetherModel, ProcessProviderResult};

/// Together provider implementation with dynamic API fetching
/// API must be available - no static data
pub struct TogetherProvider;

impl ProviderBuilder for TogetherProvider {
    type ListResponse = TogetherModel; // Http3 needs element type, not Vec
    type GetResponse = TogetherModel;

    fn provider_name(&self) -> &'static str {
        "together"
    }

    fn base_url(&self) -> &'static str {
        "https://api.together.xyz"
    }

    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("TOGETHER_API_KEY")
    }

    fn response_to_models(&self, response: Self::ListResponse) -> Vec<ModelData> {
        vec![together_model_to_data(&response)]
    }

    // Custom process() implementation for Together's direct array response format
    fn process(&self) -> ProcessProviderResult {
        // Construct full URL
        let full_url = format!("{}{}", self.base_url(), self.list_url());
        
        // Together.ai returns direct array - collect as Vec<Vec<TogetherModel>> then flatten
        let models_nested: Vec<Vec<TogetherModel>> = if let Ok(api_key) = env::var("TOGETHER_API_KEY") {
            Http3::json()
                .bearer_auth(&api_key)
                .get(&full_url)
                .collect::<Vec<TogetherModel>>()
        } else {
            Http3::json()
                .get(&full_url)
                .collect::<Vec<TogetherModel>>()
        };
        
        // Flatten nested Vec<Vec<TogetherModel>> to Vec<TogetherModel>
        let models_array: Vec<TogetherModel> = models_nested.into_iter().flatten().collect();

        // Convert to ModelData format
        let models: Vec<ModelData> = models_array
            .iter()
            .map(together_model_to_data)
            .collect();

        if models.is_empty() {
            return ProcessProviderResult {
                success: false,
                status: format!("No models found for {}", self.provider_name()),
            };
        }

        // Generate code using syn
        match self.generate_code(&models) {
            Ok((_enum_code, _impl_code)) => ProcessProviderResult {
                success: true,
                status: format!("Successfully processed {} models for {}", models.len(), self.provider_name()),
            },
            Err(e) => ProcessProviderResult {
                success: false,
                status: format!("Code generation failed for {}: {}", self.provider_name(), e),
            },
        }
    }
}

/// Convert Together model to ModelData with appropriate pricing and capabilities
/// Uses pricing from API response when available
fn together_model_to_data(model: &TogetherModel) -> ModelData {
    let context_length = model.context_length.unwrap_or(8192);
    let (input_price, output_price) = if let Some(ref pricing) = model.pricing {
        (pricing.input.unwrap_or(0.0), pricing.output.unwrap_or(0.0))
    } else {
        (0.001, 0.002) // Default pricing
    };
    
    // Together.ai doesn't currently have thinking models
    let is_thinking = false;
    let thinking_price = None;
    
    (model.id.clone(), context_length, input_price, output_price, is_thinking, thinking_price)
}
