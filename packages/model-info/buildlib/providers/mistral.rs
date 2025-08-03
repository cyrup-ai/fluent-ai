use super::{ModelData, ProviderBuilder, StandardModelsResponse, StandardModel};

/// Mistral provider implementation with dynamic API fetching
/// API must be available - no static data
pub struct MistralProvider;

impl ProviderBuilder for MistralProvider {
    type ListResponse = StandardModelsResponse; // Uses OpenAI-compatible format
    type GetResponse = StandardModel;

    fn provider_name(&self) -> &'static str {
        "mistral"
    }

    fn base_url(&self) -> &'static str {
        "https://api.mistral.ai"
    }

    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("MISTRAL_API_KEY")
    }

    fn response_to_models(&self, response: Self::ListResponse) -> Vec<ModelData> {
        response.data.into_iter()
            .map(|model| mistral_model_to_data(&model.id))
            .collect()
    }

    fn model_to_data(&self, model: &Self::GetResponse) -> ModelData {
        mistral_model_to_data(&model.id)
    }
}

/// Convert Mistral model ID to ModelData with appropriate pricing and capabilities
/// Based on current Mistral pricing as of 2024
fn mistral_model_to_data(model_id: &str) -> ModelData {
    match model_id {
        "mistral-large-latest" => (model_id.to_string(), 128000, 2.0, 6.0, false, None),
        "mistral-large-2407" => (model_id.to_string(), 128000, 2.0, 6.0, false, None),
        "mistral-large-2402" => (model_id.to_string(), 32000, 4.0, 12.0, false, None),
        "mistral-medium-latest" => (model_id.to_string(), 32000, 2.7, 8.1, false, None),
        "mistral-small-latest" => (model_id.to_string(), 32000, 1.0, 3.0, false, None),
        "mistral-tiny" => (model_id.to_string(), 32000, 0.14, 0.42, false, None),
        "codestral-latest" => (model_id.to_string(), 32000, 0.2, 0.6, false, None),
        "codestral-2405" => (model_id.to_string(), 32000, 0.2, 0.6, false, None),
        
        // Default for unknown models
        _ => (model_id.to_string(), 32000, 1.0, 3.0, false, None),
    }
}
