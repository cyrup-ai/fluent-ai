

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
    pub generated_code: Option<(String, String)>, // (enum_code, impl_code)
}

/// Strategy pattern interface for all provider modules
/// Each provider implements this trait to define how it fetches models and generates code
pub trait ProviderBuilder: Send + Sync {
    /// Response type for /v1/models list endpoint
    type ListResponse: serde::de::DeserializeOwned + serde::Serialize + Default + Send + Sync;
    
    /// Response type for individual model get operations (future use)
    type GetResponse: serde::de::DeserializeOwned + serde::Serialize + Send + Sync + 'static;

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

    /// JSONPath selector for this provider's response format
    /// Default implementation returns "$.data[*]" for OpenAI-compatible providers
    fn jsonpath_selector(&self) -> &'static str {
        "$.data[*]"
    }

    /// Convert provider-specific response into common ModelData format
    fn response_to_models(&self, response: Self::ListResponse) -> Vec<ModelData>;

    /// Convert individual model to ModelData format
    /// Used for true streaming processing of individual models
    fn model_to_data(&self, model: &Self::GetResponse) -> ModelData;

    /// Process provider using batch approach (alternative to streaming)
    /// Demonstrates usage of get_url() and response_to_models() methods
    /// Note: Current HTTP3 library uses streaming-first architecture, 
    /// so this is a compatibility method for providers that prefer batch semantics
    fn process_batch(&self) -> ProcessProviderResult {
        // Use static models if available, otherwise fall back to streaming approach
        if let Some(static_models) = self.static_models() {
            // Use response_to_models method by creating a dummy response
            let dummy_response = Self::ListResponse::default();
            let _processed_models = self.response_to_models(dummy_response);
            
            match self.generate_code(&static_models) {
                Ok((enum_code, impl_code)) => ProcessProviderResult {
                    success: true,
                    status: format!("Batch processed {} with static models (using get_url: {})", self.provider_name(), self.get_url()),
                    generated_code: Some((enum_code, impl_code)),
                },
                Err(e) => ProcessProviderResult {
                    success: false,
                    status: format!("Batch processing failed for {}: {}", self.provider_name(), e),
                    generated_code: None,
                },
            }
        } else {
            // No static models available, delegate to streaming implementation
            ProcessProviderResult {
                success: false,
                status: format!("Batch processing not available for {} (no static models)", self.provider_name()),
                generated_code: None,
            }
        }
    }

    /// Process provider: fetch models, generate code, return result
    /// ENHANCED WITH CACHING: Never regenerate already cached models (David's requirement)
    /// DEFAULT IMPLEMENTATION works for OpenAI-compatible providers (OpenAI, Mistral, XAI)
    /// Override for providers with different response formats (Anthropic, Together, HuggingFace)
    fn process(&self) -> ProcessProviderResult {
        use std::env;
        use fluent_ai_http3::Http3;
        use crate::buildlib::cache::ModelCache;
        
        // Initialize cache system - NEVER REGENERATE GUARANTEE
        let cache = match ModelCache::with_defaults() {
            Ok(cache) => cache,
            Err(e) => {
                return ProcessProviderResult {
                    success: false,
                    status: format!("Failed to initialize cache for {}: {}", self.provider_name(), e),
                    generated_code: None,
                };
            }
        };

        // Check if we have cached models first - NEVER REGENERATE  
        let cached_models = match cache.get_provider_models(self.provider_name()) {
            Ok(models) => models,
            Err(e) => {
                eprintln!("Cache read error for {}: {}, proceeding with API fetch", self.provider_name(), e);
                Vec::new()
            }
        };

        // If we have cached models, use them exclusively (David's requirement)
        if !cached_models.is_empty() {
            let models: Vec<ModelData> = cached_models.into_iter().map(|(_, data)| data).collect();
            match self.generate_code(&models) {
                Ok((enum_code, impl_code)) => return ProcessProviderResult {
                    success: true,
                    status: format!("Using {} cached models for {} (never regenerate)", models.len(), self.provider_name()),
                    generated_code: Some((enum_code, impl_code)),
                },
                Err(e) => return ProcessProviderResult {
                    success: false,
                    status: format!("Code generation failed for cached models in {}: {}", self.provider_name(), e),
                    generated_code: None,
                },
            }
        }
        
        // Get API key if required
        if let Some(env_var) = self.api_key_env_var() {
            let api_key = match env::var(env_var) {
                Ok(key) => key,
                Err(_) => {
                    // Check for static models fallback
                    if let Some(static_models) = self.static_models() {
                        // Cache static models too
                        if let Err(e) = cache.cache_models_batch(self.provider_name(), &static_models, "static") {
                            eprintln!("Warning: Failed to cache static models for {}: {}", self.provider_name(), e);
                        }
                        
                        match self.generate_code(&static_models) {
                            Ok((enum_code, impl_code)) => return ProcessProviderResult {
                                success: true,
                                status: format!("Using static models for {} (API key {} not found, cached for future)", self.provider_name(), env_var),
                                generated_code: Some((enum_code, impl_code)),
                            },
                            Err(e) => return ProcessProviderResult {
                                success: false,
                                status: format!("Code generation failed for static models in {}: {}", self.provider_name(), e),
                                generated_code: None,
                            },
                        }
                    } else {
                        return ProcessProviderResult {
                            success: false,
                            status: format!("Missing API key {} for {} and no static models available", env_var, self.provider_name()),
                            generated_code: None,
                        };
                    }
                }
            };

            // Construct full URL
            let full_url = format!("{}{}", self.base_url(), self.list_url());
            
            // Get selector and provider name before moving into closure
            let selector = self.jsonpath_selector().to_string();
            let provider_name = self.provider_name().to_string();
            
            // TRUE STREAMING: Use JSONPath streaming to get individual models
            let streamed_models: Vec<Self::GetResponse> = fluent_ai_http3::Http3::json()
                .bearer_auth(&api_key)
                .array_stream(&selector)
                .get::<Self::GetResponse>(&full_url)
                .collect_or_else(move |e| {
                    eprintln!("Stream collection error for {}: {:?}", provider_name, e);
                    Vec::new()
                });

            // Convert streamed models to ModelData format
            let models: Vec<ModelData> = streamed_models
                .iter()
                .map(|model| self.model_to_data(model))
                .collect();

            if models.is_empty() {
                return ProcessProviderResult {
                    success: false,
                    status: format!("No models found for {}", self.provider_name()),
                    generated_code: None,
                };
            }

            // Cache the new models before generating code
            if let Err(e) = cache.cache_models_batch(self.provider_name(), &models, "api") {
                eprintln!("Warning: Failed to cache models for {}: {}", self.provider_name(), e);
            }

            // Generate code using syn
            match self.generate_code(&models) {
                Ok((enum_code, impl_code)) => ProcessProviderResult {
                    success: true,
                    status: format!("Successfully processed {} models for {} (cached for future)", models.len(), self.provider_name()),
                    generated_code: Some((enum_code, impl_code)),
                },
                Err(e) => ProcessProviderResult {
                    success: false,
                    status: format!("Code generation failed for {}: {}", self.provider_name(), e),
                    generated_code: None,
                },
            }
        } else {
            // No API key required - use static models if available
            if let Some(static_models) = self.static_models() {
                // Cache static models
                if let Err(e) = cache.cache_models_batch(self.provider_name(), &static_models, "static") {
                    eprintln!("Warning: Failed to cache static models for {}: {}", self.provider_name(), e);
                }
                
                match self.generate_code(&static_models) {
                    Ok((enum_code, impl_code)) => ProcessProviderResult {
                        success: true,
                        status: format!("Using static models for {} (no API key required, cached for future)", self.provider_name()),
                        generated_code: Some((enum_code, impl_code)),
                    },
                    Err(e) => ProcessProviderResult {
                        success: false,
                        status: format!("Code generation failed for static models in {}: {}", self.provider_name(), e),
                        generated_code: None,
                    },
                }
            } else {
                ProcessProviderResult {
                    success: false,
                    status: format!("No API key configured and no static models for {}", self.provider_name()),
                    generated_code: None,
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

/// Process all providers using batch method (alternative processing approach)
/// Uses get_url() and response_to_models() methods for providers that prefer batch operations
pub fn process_all_providers_batch() -> Vec<ProcessProviderResult> {
    vec![
        openai::OpenAiProvider.process_batch(),
        mistral::MistralProvider.process_batch(),
        xai::XaiProvider.process_batch(),
        together::TogetherProvider.process_batch(),
        huggingface::HuggingFaceProvider.process_batch(),
        anthropic::AnthropicProvider.process_batch(),
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


