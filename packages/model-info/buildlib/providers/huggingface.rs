use fluent_ai_http3::{Http3, HttpStreamExt};
use super::{ModelData, ProviderBuilder, HuggingFaceModel, ProcessProviderResult};

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

    fn api_key_env_var(&self) -> Option<&'static str> {
        // HuggingFace API is public for model listing
        None
    }

    fn response_to_models(&self, response: Self::ListResponse) -> Vec<ModelData> {
        vec![huggingface_model_to_data(&response)]
    }

    // Custom process() implementation for HuggingFace's direct array response format
    fn process(&self) -> ProcessProviderResult {
        // Construct full URL
        let full_url = format!("{}{}", self.base_url(), self.list_url());
        
        // HuggingFace returns direct array - collect as Vec<Vec<HuggingFaceModel>> then flatten
        let models_nested: Vec<Vec<HuggingFaceModel>> = Http3::json()
            .get(&full_url)
            .collect::<Vec<HuggingFaceModel>>();
        
        // Flatten nested Vec<Vec<HuggingFaceModel>> to Vec<HuggingFaceModel>
        let models_array: Vec<HuggingFaceModel> = models_nested.into_iter().flatten().collect();

        // Convert to ModelData format
        let models: Vec<ModelData> = models_array
            .iter()
            .map(huggingface_model_to_data)
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