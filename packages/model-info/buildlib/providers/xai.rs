use super::response_types::{OpenAiModel, OpenAiModelsListResponse};
use super::{ModelData, ProviderBuilder};

/// X.AI provider implementation with dynamic API fetching
/// Uses the official X.AI API endpoint as documented at https://docs.x.ai/docs/api-reference#list-models
pub struct XaiProvider;

impl ProviderBuilder for XaiProvider {
    type ListResponse = OpenAiModelsListResponse; // Uses OpenAI-compatible format
    type GetResponse = OpenAiModel;

    fn provider_name(&self) -> &'static str {
        "xai"
    }

    fn base_url(&self) -> &'static str {
        "https://api.x.ai"
    }

    fn api_key_env_vars(&self) -> cyrup_sugars::ZeroOneOrMany<&'static str> {
        cyrup_sugars::ZeroOneOrMany::One("XAI_API_KEY")
    }

    fn response_to_models(&self, response: Self::ListResponse) -> Vec<ModelData> {
        response
            .data
            .into_iter()
            .map(|model| xai_model_to_data(&model.id))
            .collect()
    }

    fn model_to_data(&self, model: &Self::GetResponse) -> ModelData {
        xai_model_to_data(&model.id)
    }
}

/// Convert X.AI model ID to ModelData with appropriate pricing and capabilities
/// Based on current X.AI pricing as of 2024
fn xai_model_to_data(model_id: &str) -> ModelData {
    match model_id {
        "grok-beta" => (model_id.to_string(), 131072, 5.0, 15.0, true, Some(1.0)),
        "grok-2" => (model_id.to_string(), 131072, 2.0, 10.0, true, Some(1.0)),
        "grok-2-mini" => (model_id.to_string(), 131072, 0.3, 0.5, true, None),

        // Default for unknown models
        _ => (model_id.to_string(), 131072, 2.0, 10.0, true, None),
    }
}
