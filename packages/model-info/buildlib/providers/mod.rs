pub mod anthropic;
pub mod huggingface;
pub mod mistral;
pub mod openai;
pub mod response_types;
pub mod together;
pub mod xai;

// pub use response_types::*; // Currently unused, will be needed for batch processing

/// Type alias for model data tuple: (name, max_tokens, input_price, output_price, supports_thinking, required_temp)
pub type ModelData = (String, u64, f64, f64, bool, Option<f64>);

/// Result type for provider processing operations
#[derive(Debug, Clone)]
pub struct ProcessProviderResult {
    pub success: bool,
    pub status: String,
    pub generated_code: Option<(String, String)>, // (enum_code, impl_code)
}

/// Strategy pattern interface for all provider modules - BUILDLIB VERSION
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

    /// Get single model endpoint path (e.g. "/v1/models/{id}") - Future batch API
    #[allow(dead_code)] // Future API - will be used for batch processing
    fn get_url(&self) -> &'static str {
        "/v1/models"
    }

    /// Environment variable name for API key - SUPPORTS MULTIPLE KEYS via ZeroOneOrMany
    fn api_key_env_vars(&self) -> cyrup_sugars::ZeroOneOrMany<&'static str>;

    /// JSONPath selector for this provider's response format
    /// Default implementation returns "$.data[*]" for OpenAI-compatible providers
    fn jsonpath_selector(&self) -> &'static str {
        "$.data[*]"
    }

    /// Convert provider-specific response into common ModelData format - Future batch API
    #[allow(dead_code)] // Future API - will be used for batch processing instead of streaming
    fn response_to_models(&self, response: Self::ListResponse) -> Vec<ModelData>;

    /// Convert individual model to ModelData format
    /// Used for true streaming processing of individual models
    fn model_to_data(&self, model: &Self::GetResponse) -> ModelData;

    /// Get static/hardcoded model data for providers without APIs
    /// Used as the primary data source for providers like Anthropic
    fn static_models(&self) -> Option<Vec<ModelData>> {
        None
    }

    /// Generate model enum and implementation code
    /// Takes the model data and produces the Rust code for the enum and trait impl
    fn generate_code(&self, models: &[ModelData]) -> anyhow::Result<(String, String)> {
        use super::codegen::SynCodeGenerator;
        let codegen = SynCodeGenerator::new(self.provider_name());
        let enum_code = codegen.generate_enum(models)?;
        let impl_code = codegen.generate_trait_impl(models)?;
        Ok((enum_code, impl_code))
    }

    /// Process provider: fetch models, generate code, return result
    fn process(&self) -> ProcessProviderResult {
        use std::env;

        // Removed unused import: fluent_ai_http3::Http3 to fix warning
        // This import was causing a warning and has been commented out.
        use super::cache::ModelCache;

        // Initialize cache system - NEVER REGENERATE GUARANTEE
        let cache = match ModelCache::with_defaults() {
            Ok(cache) => cache,
            Err(e) => {
                return ProcessProviderResult {
                    success: false,
                    status: format!(
                        "Failed to initialize cache for {}: {}",
                        self.provider_name(),
                        e
                    ),
                    generated_code: None,
                };
            }
        };

        // Check if we have cached models first - NEVER REGENERATE
        let cached_models = match cache.get_provider_models(self.provider_name()) {
            Ok(models) => models,
            Err(e) => {
                eprintln!(
                    "Cache read error for {}: {}, proceeding with API fetch",
                    self.provider_name(),
                    e
                );
                Vec::new()
            }
        };

        // If we have cached models, use them exclusively (David's requirement)
        if !cached_models.is_empty() {
            let models: Vec<ModelData> = cached_models.into_iter().map(|(_, data)| data).collect();
            match self.generate_code(&models) {
                Ok((enum_code, impl_code)) => {
                    return ProcessProviderResult {
                        success: true,
                        status: format!(
                            "Using {} cached models for {} (never regenerate)",
                            models.len(),
                            self.provider_name()
                        ),
                        generated_code: Some((enum_code, impl_code)),
                    };
                }
                Err(e) => {
                    return ProcessProviderResult {
                        success: false,
                        status: format!(
                            "Code generation failed for cached models in {}: {}",
                            self.provider_name(),
                            e
                        ),
                        generated_code: None,
                    };
                }
            }
        }

        // Get API keys - SUPPORTS MULTIPLE KEYS via ZeroOneOrMany
        let api_keys = self.api_key_env_vars();
        let resolved_key = match api_keys {
            cyrup_sugars::ZeroOneOrMany::None => {
                // No API key required - use static models if available
                if let Some(static_models) = self.static_models() {
                    // Cache static models
                    if let Err(e) =
                        cache.cache_models_batch(self.provider_name(), &static_models, "static")
                    {
                        eprintln!(
                            "Warning: Failed to cache static models for {}: {}",
                            self.provider_name(),
                            e
                        );
                    }

                    match self.generate_code(&static_models) {
                        Ok((enum_code, impl_code)) => {
                            return ProcessProviderResult {
                                success: true,
                                status: format!(
                                    "Using static models for {} (no API key required, cached for future)",
                                    self.provider_name()
                                ),
                                generated_code: Some((enum_code, impl_code)),
                            };
                        }
                        Err(e) => {
                            return ProcessProviderResult {
                                success: false,
                                status: format!(
                                    "Code generation failed for static models in {}: {}",
                                    self.provider_name(),
                                    e
                                ),
                                generated_code: None,
                            };
                        }
                    }
                } else {
                    return ProcessProviderResult {
                        success: false,
                        status: format!(
                            "No API key configured and no static models for {}",
                            self.provider_name()
                        ),
                        generated_code: None,
                    };
                }
            }
            cyrup_sugars::ZeroOneOrMany::One(env_var) => {
                match env::var(env_var) {
                    Ok(key) => key,
                    Err(_) => {
                        // Check for static models fallback
                        if let Some(static_models) = self.static_models() {
                            // Cache static models too
                            if let Err(e) = cache.cache_models_batch(
                                self.provider_name(),
                                &static_models,
                                "static",
                            ) {
                                eprintln!(
                                    "Warning: Failed to cache static models for {}: {}",
                                    self.provider_name(),
                                    e
                                );
                            }

                            match self.generate_code(&static_models) {
                                Ok((enum_code, impl_code)) => {
                                    return ProcessProviderResult {
                                        success: true,
                                        status: format!(
                                            "Using static models for {} (API key {} not found, cached for future)",
                                            self.provider_name(),
                                            env_var
                                        ),
                                        generated_code: Some((enum_code, impl_code)),
                                    };
                                }
                                Err(e) => {
                                    return ProcessProviderResult {
                                        success: false,
                                        status: format!(
                                            "Code generation failed for static models in {}: {}",
                                            self.provider_name(),
                                            e
                                        ),
                                        generated_code: None,
                                    };
                                }
                            }
                        } else {
                            return ProcessProviderResult {
                                success: false,
                                status: format!(
                                    "Missing API key {} for {} and no static models available",
                                    env_var,
                                    self.provider_name()
                                ),
                                generated_code: None,
                            };
                        }
                    }
                }
            }
            cyrup_sugars::ZeroOneOrMany::Many(env_vars) => {
                // Try each API key in sequence until one works
                let mut last_error = String::new();
                let mut found_key = None;
                for env_var in env_vars {
                    match env::var(env_var) {
                        Ok(key) => {
                            found_key = Some(key);
                            break;
                        }
                        Err(e) => {
                            last_error = format!("Failed to get key from {}: {}", env_var, e);
                            continue;
                        }
                    }
                }

                match found_key {
                    Some(key) => key,
                    None => {
                        // If no keys found, try static models
                        if let Some(static_models) = self.static_models() {
                            // Cache static models too
                            if let Err(e) = cache.cache_models_batch(
                                self.provider_name(),
                                &static_models,
                                "static",
                            ) {
                                eprintln!(
                                    "Warning: Failed to cache static models for {}: {}",
                                    self.provider_name(),
                                    e
                                );
                            }

                            match self.generate_code(&static_models) {
                                Ok((enum_code, impl_code)) => {
                                    return ProcessProviderResult {
                                        success: true,
                                        status: format!(
                                            "Using static models for {} (no API keys found: {}, cached for future)",
                                            self.provider_name(),
                                            last_error
                                        ),
                                        generated_code: Some((enum_code, impl_code)),
                                    };
                                }
                                Err(e) => {
                                    return ProcessProviderResult {
                                        success: false,
                                        status: format!(
                                            "Code generation failed for static models in {}: {}",
                                            self.provider_name(),
                                            e
                                        ),
                                        generated_code: None,
                                    };
                                }
                            }
                        } else {
                            return ProcessProviderResult {
                                success: false,
                                status: format!(
                                    "All API keys failed for {} and no static models available: {}",
                                    self.provider_name(),
                                    last_error
                                ),
                                generated_code: None,
                            };
                        }
                    }
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
            .bearer_auth(&resolved_key)
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
            eprintln!(
                "Warning: Failed to cache models for {}: {}",
                self.provider_name(),
                e
            );
        }

        // Generate code using syn
        match self.generate_code(&models) {
            Ok((enum_code, impl_code)) => ProcessProviderResult {
                success: true,
                status: format!(
                    "Successfully processed {} models for {} (cached for future)",
                    models.len(),
                    self.provider_name()
                ),
                generated_code: Some((enum_code, impl_code)),
            },
            Err(e) => ProcessProviderResult {
                success: false,
                status: format!("Code generation failed for {}: {}", self.provider_name(), e),
                generated_code: None,
            },
        }
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

/// Process all providers using batch method (alternative processing approach) - Future API
/// Uses get_url() and response_to_models() methods for providers that prefer batch operations
#[allow(dead_code)] // Future API - alternative to streaming approach
pub fn process_all_providers_batch() -> Vec<ProcessProviderResult> {
    // For now, just use regular process() method since process_batch() is not implemented
    process_all_providers()
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
