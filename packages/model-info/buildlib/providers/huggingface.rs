use super::response_types::HuggingFaceModel;
use super::{ModelData, ProviderBuilder};

/// HuggingFace provider implementation with dynamic API fetching
/// API must be available - no static data
pub struct HuggingFaceProvider;

impl ProviderBuilder for HuggingFaceProvider {
    type ListResponse = HuggingFaceModel; // Http3 needs element type, not Vec
    type GetResponse = HuggingFaceModel;

    fn provider_name(&self) -> &'static str {
        "huggingface"
    }

    fn base_url(&self) -> &'static str {
        "https://huggingface.co"
    }

    fn list_url(&self) -> &'static str {
        "/api/models?filter=text-generation&sort=downloads&direction=-1&limit=50"
    }

    fn api_key_env_vars(&self) -> cyrup_sugars::ZeroOneOrMany<&'static str> {
        // HuggingFace API is public for model listing
        cyrup_sugars::ZeroOneOrMany::None
    }

    fn jsonpath_selector(&self) -> &'static str {
        "$[*]" // HuggingFace returns direct array format
    }

    fn response_to_models(&self, response: Self::ListResponse) -> Vec<ModelData> {
        vec![huggingface_model_to_data(&response)]
    }

    fn model_to_data(&self, model: &Self::GetResponse) -> ModelData {
        huggingface_model_to_data(model)
    }
}

/// Convert HuggingFace model to ModelData with appropriate context and capabilities
/// HuggingFace models are typically free/open-source with pricing of 0.0
fn huggingface_model_to_data(model: &HuggingFaceModel) -> ModelData {
    // Determine context length based on model name/tags
    let context_length = match model.id.as_str() {
        id if id.contains("llama-2") => 4096,
        id if id.contains("llama-3") => 8192,
        id if id.contains("mixtral") => 32768,
        id if id.contains("codellama") => 16384,
        id if id.contains("mistral") => 8192,
        id if id.contains("gemma") => 8192,
        id if id.contains("qwen") => 32768,
        _ => 4096, // Default context length
    };

    // HuggingFace models are typically free/open-source (pricing 0.0)
    // No thinking models on HuggingFace currently
    (model.id.clone(), context_length, 0.0, 0.0, false, None)
}
