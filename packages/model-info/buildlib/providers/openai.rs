use super::{ModelData, ProviderBuilder, StandardModelsResponse, StandardModel};

/// OpenAI provider implementation with dynamic API fetching
/// API must be available - no static data
pub struct OpenAiProvider;

impl ProviderBuilder for OpenAiProvider {
    type ListResponse = StandardModelsResponse;
    type GetResponse = StandardModel;

    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn base_url(&self) -> &'static str {
        "https://api.openai.com"
    }

    fn api_key_env_var(&self) -> Option<&'static str> {
        Some("OPENAI_API_KEY")
    }


    fn response_to_models(&self, response: Self::ListResponse) -> Vec<ModelData> {
        response.data.into_iter()
            .map(|model| openai_model_to_data(&model.id))
            .collect()
    }

    fn model_to_data(&self, model: &Self::GetResponse) -> ModelData {
        openai_model_to_data(&model.id)
    }
}

/// Convert OpenAI model ID to ModelData with appropriate pricing and capabilities
/// Based on current OpenAI pricing as of 2024
fn openai_model_to_data(model_id: &str) -> ModelData {
    match model_id {
        // GPT-4 models
        "gpt-4" => (model_id.to_string(), 128000, 0.03, 0.06, false, Some(0.0)),
        "gpt-4-turbo" => (model_id.to_string(), 128000, 0.01, 0.03, false, Some(0.0)),
        "gpt-4o" => (model_id.to_string(), 128000, 0.005, 0.015, false, Some(0.0)),
        "gpt-4o-mini" => (model_id.to_string(), 128000, 0.00015, 0.0006, false, Some(0.0)),
        
        // GPT-3.5 models
        "gpt-3.5-turbo" => (model_id.to_string(), 16384, 0.001, 0.002, false, Some(0.0)),
        "gpt-3.5-turbo-instruct" => (model_id.to_string(), 4096, 0.0015, 0.002, false, Some(0.0)),
        
        // Text models
        "text-davinci-003" => (model_id.to_string(), 4097, 0.02, 0.02, false, Some(0.0)),
        "text-curie-001" => (model_id.to_string(), 2049, 0.002, 0.002, false, Some(0.0)),
        "text-babbage-001" => (model_id.to_string(), 2049, 0.0005, 0.0005, false, Some(0.0)),
        "text-ada-001" => (model_id.to_string(), 2049, 0.0004, 0.0004, false, Some(0.0)),
        
        // Default for unknown models
        _ => (model_id.to_string(), 8192, 0.001, 0.002, false, Some(0.0)),
    }
}
