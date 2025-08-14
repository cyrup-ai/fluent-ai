use super::response_types::TogetherModel;
use super::{ModelData, ProviderBuilder};

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

    fn api_key_env_vars(&self) -> cyrup_sugars::ZeroOneOrMany<&'static str> {
        cyrup_sugars::ZeroOneOrMany::One("TOGETHER_API_KEY")
    }

    fn jsonpath_selector(&self) -> &'static str {
        "$[*]" // Together.ai returns direct array format
    }

    fn response_to_models(&self, response: Self::ListResponse) -> Vec<ModelData> {
        vec![together_model_to_data(&response)]
    }

    fn model_to_data(&self, model: &Self::GetResponse) -> ModelData {
        together_model_to_data(model)
    }
}

/// Convert Together model to ModelData with appropriate pricing and capabilities
/// Uses pricing from API response when available
fn together_model_to_data(model: &TogetherModel) -> ModelData {
    let context_length = if model.context_length == 0 {
        8192
    } else {
        model.context_length
    };
    let (input_price, output_price) = if let Some(ref pricing) = model.pricing {
        (pricing.input, pricing.output)
    } else {
        (0.001, 0.002) // Default pricing
    };

    // Together.ai doesn't currently have thinking models
    let is_thinking = false;
    let thinking_price = None;

    (
        model.id.clone(),
        context_length,
        input_price,
        output_price,
        is_thinking,
        thinking_price,
    )
}
