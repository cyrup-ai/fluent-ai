

pub mod anthropic;
pub mod huggingface;
pub mod mistral;
pub mod openai;
pub mod together;
pub mod xai;

/// Type alias for model data tuple: (name, max_tokens, input_price, output_price, supports_thinking, required_temp)
pub type ModelData = (String, u64, f64, f64, bool, Option<f64>);

/// OpenAI-compatible /v1/models request (empty for GET requests)
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct OpenAiModelsListRequest;

/// OpenAI-compatible /v1/models response structure used by OpenAI, Mistral, XAI
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct OpenAiModelsListResponse {
    pub data: Vec<OpenAiModel>,
    pub object: String,
}

/// Standard model object from OpenAI-compatible /v1/models endpoints
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct OpenAiModel {
    pub id: String,
    pub object: String,
    pub created: Option<u64>,
    pub owned_by: Option<String>,
}

/// Together.ai direct array response format
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct TogetherModelsListResponse(pub Vec<TogetherModel>);

/// Together.ai model object with extended pricing and config data
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct TogetherModel {
    pub id: String,
    pub object: String,
    pub created: Option<u64>,
    #[serde(rename = "type")]
    pub model_type: Option<String>,
    pub running: Option<bool>,
    pub display_name: Option<String>,
    pub organization: Option<String>,
    pub link: Option<String>,
    pub context_length: Option<u64>,
    pub config: Option<TogetherModelConfig>,
    pub pricing: Option<TogetherModelPricing>,
}

#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct TogetherModelConfig {
    pub chat_template: Option<String>,
    pub stop: Vec<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct TogetherModelPricing {
    pub hourly: Option<f64>,
    pub input: Option<f64>,
    pub output: Option<f64>,
    pub base: Option<f64>,
    pub finetune: Option<f64>,
}

/// HuggingFace API response format (different structure)
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct HuggingFaceModelsListResponse {
    pub models: Vec<HuggingFaceModel>,
}

#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct HuggingFaceModel {
    pub id: String,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
}

/// Legacy aliases for backward compatibility
pub type StandardModelsResponse = OpenAiModelsListResponse;
pub type StandardModel = OpenAiModel;

/// Result type for provider processing operations
#[derive(Debug, Clone)]
pub struct ProcessProviderResult {
    pub success: bool,
    pub status: String,
}

/// Strategy pattern interface for all provider modules
/// Each provider implements this trait to define how it fetches models and generates code
pub trait ProviderBuilder: Send + Sync {
    /// Response type for /v1/models list endpoint
    type ListResponse: serde::de::DeserializeOwned + serde::Serialize + Default + Send + Sync;
    
    /// Response type for individual model get operations (future use)
    type GetResponse: serde::de::DeserializeOwned + serde::Serialize + Send + Sync;

    /// Provider name for identification and enum generation
    fn provider_name(&self) -> &'static str;

    /// Base API URL (e.g. "https://api.openai.com")
    fn base_url(&self) -> &'static str;

    /// List models endpoint path (e.g. "/v1/models")
    fn list_url(&self) -> &'static str {
        "/v1/models"
    }

    /// Get single model endpoint path (e.g. "/v1/models/{id}")
    fn get_url(&self) -> &'static str {
        "/v1/models"
    }

    /// Environment variable name for API key
    fn api_key_env_var(&self) -> Option<&'static str>;



    /// Convert provider-specific response into common ModelData format
    fn response_to_models(&self, response: Self::ListResponse) -> Vec<ModelData>;

    /// Process provider: fetch models, generate code, return result
    /// DEFAULT IMPLEMENTATION works for OpenAI-compatible providers (OpenAI, Mistral, XAI)
    /// Override for providers with different response formats (Anthropic, Together, HuggingFace)
    fn process(&self) -> ProcessProviderResult {
        use std::env;
        use fluent_ai_http3::{Http3, HttpStreamExt};
        
        // Get API key if required
        if let Some(env_var) = self.api_key_env_var() {
            let api_key = match env::var(env_var) {
                Ok(key) => key,
                Err(_) => {
                    // Check for static models fallback
                    if let Some(static_models) = self.static_models() {
                        match self.generate_code(&static_models) {
                            Ok((_enum_code, _impl_code)) => return ProcessProviderResult {
                                success: true,
                                status: format!("Using static models for {} (API key {} not found)", self.provider_name(), env_var),
                            },
                            Err(e) => return ProcessProviderResult {
                                success: false,
                                status: format!("Code generation failed for static models in {}: {}", self.provider_name(), e),
                            },
                        }
                    } else {
                        return ProcessProviderResult {
                            success: false,
                            status: format!("Missing API key {} for {} and no static models available", env_var, self.provider_name()),
                        };
                    }
                }
            };

            // Construct full URL
            let full_url = format!("{}{}", self.base_url(), self.list_url());
            
            // Use high-level Http3 builder with proper error handling (OpenAI-compatible format)
            let provider_name = self.provider_name();
            // Get JSON response and try to deserialize into expected format
            let json_response = Http3::json()
                .bearer_auth(&api_key)
                .get(&full_url)
                .collect_one_or_else(move |error| {
                    eprintln!("{} API request failed: {}", provider_name, error);
                    serde_json::Value::Array(vec![])
                });
            
            let response: Self::ListResponse = serde_json::from_value(json_response)
                .unwrap_or_else(|_| Self::ListResponse::default());
            
            let models = self.response_to_models(response);

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
        } else {
            // No API key required - use static models if available
            if let Some(static_models) = self.static_models() {
                match self.generate_code(&static_models) {
                    Ok((_enum_code, _impl_code)) => ProcessProviderResult {
                        success: true,
                        status: format!("Using static models for {} (no API key required)", self.provider_name()),
                    },
                    Err(e) => ProcessProviderResult {
                        success: false,
                        status: format!("Code generation failed for static models in {}: {}", self.provider_name(), e),
                    },
                }
            } else {
                ProcessProviderResult {
                    success: false,
                    status: format!("No API key configured and no static models for {}", self.provider_name()),
                }
            }
        }
    }

    /// Get static/hardcoded model data for providers without APIs
    /// Used as the primary data source for providers like Anthropic
    fn static_models(&self) -> Option<Vec<ModelData>> { None }

    /// Generate model enum and implementation code
    /// Takes the model data and produces the Rust code for the enum and trait impl
    fn generate_code(&self, models: &[ModelData]) -> anyhow::Result<(String, String)> {
        use crate::buildlib::codegen::SynCodeGenerator;
        let codegen = SynCodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }
}

/// Process all providers individually using impl Trait pattern
pub fn process_all_providers() -> Vec<ProcessProviderResult> {
    vec![
        openai::OpenAiProvider.process(),
        mistral::MistralProvider.process(),
        xai::XaiProvider.process(),
        together::TogetherProvider.process(),
        huggingface::HuggingFaceProvider.process(),
        anthropic::AnthropicProvider.process(),
    ]
}

/// Utility function to sanitize model IDs into valid Rust identifiers
#[inline]
pub fn sanitize_ident(id: &str) -> String {
    let words: Vec<&str> = id.split(|c: char| !c.is_alphanumeric()).collect();
    let mut pascal_case = String::new();

    for word in words {
        if !word.is_empty() {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                pascal_case.push(first.to_ascii_uppercase());
                for ch in chars {
                    pascal_case.push(ch.to_ascii_lowercase());
                }
            }
        }
    }

    // Handle leading digits and empty strings
    if pascal_case
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit())
    {
        format!("Model{pascal_case}")
    } else if pascal_case.is_empty() {
        "Unknown".to_string()
    } else {
        pascal_case
    }
}


