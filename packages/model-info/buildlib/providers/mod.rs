use anyhow::Result;
use fluent_ai_async::AsyncStream;

pub mod anthropic;
pub mod huggingface;
pub mod mistral;
pub mod openai;
pub mod openrouter;
pub mod together;
pub mod xai;

/// Type alias for model data tuple: (name, max_tokens, input_price, output_price, supports_thinking, required_temp)
pub type ModelData = (String, u64, f64, f64, bool, Option<f64>);

/// Standard /v1/models API response structure used by OpenAI, Mistral, OpenRouter, XAI, Together
#[derive(serde::Deserialize, serde::Serialize, Default)]
pub struct StandardModelsResponse {
    pub data: Vec<StandardModel>,
    pub object: String,
}

/// Standard model object from /v1/models endpoints
#[derive(serde::Deserialize, serde::Serialize)]
pub struct StandardModel {
    pub id: String,
    pub object: String,
    pub created: Option<u64>,
    pub owned_by: Option<String>,
}

/// Strategy pattern interface for all provider modules
/// Each provider implements this trait to define how it fetches models and generates code
pub trait ProviderBuilder: Send + Sync {
    /// Provider name for identification and enum generation
    fn provider_name(&self) -> &'static str;

    /// API endpoint for fetching models (None for providers without dynamic endpoints)
    #[allow(dead_code)] // Library method - may be used in future API implementations
    fn api_endpoint(&self) -> Option<&'static str>;

    /// Environment variable name for API key
    #[allow(dead_code)] // Library method - may be used in future API implementations
    fn api_key_env_var(&self) -> Option<&'static str>;

    /// Fetch models from the provider's API
    /// Returns AsyncStream for providers that support dynamic fetching
    fn fetch_models(&self) -> AsyncStream<ModelData>;

    /// Get static/hardcoded model data for providers without APIs
    /// Used as the primary data source for providers like Anthropic
    fn static_models(&self) -> Option<Vec<ModelData>>;

    /// Generate model enum and implementation code
    /// Takes the model data and produces the Rust code for the enum and trait impl
    fn generate_code(&self, models: &[ModelData]) -> anyhow::Result<(String, String)>;
}

/// Enum containing all available provider types
/// This allows us to avoid trait objects while still processing providers uniformly
pub enum Provider {
    OpenAi(openai::OpenAiProvider),
    Mistral(mistral::MistralProvider),
    Anthropic(anthropic::AnthropicProvider),
    Together(together::TogetherProvider),
    OpenRouter(openrouter::OpenRouterProvider),
    HuggingFace(huggingface::HuggingFaceProvider),
    Xai(xai::XaiProvider),
}

impl ProviderBuilder for Provider {
    fn provider_name(&self) -> &'static str {
        match self {
            Provider::OpenAi(p) => p.provider_name(),
            Provider::Mistral(p) => p.provider_name(),
            Provider::Anthropic(p) => p.provider_name(),
            Provider::Together(p) => p.provider_name(),
            Provider::OpenRouter(p) => p.provider_name(),
            Provider::HuggingFace(p) => p.provider_name(),
            Provider::Xai(p) => p.provider_name(),
        }
    }

    fn api_endpoint(&self) -> Option<&'static str> {
        match self {
            Provider::OpenAi(p) => p.api_endpoint(),
            Provider::Mistral(p) => p.api_endpoint(),
            Provider::Anthropic(p) => p.api_endpoint(),
            Provider::Together(p) => p.api_endpoint(),
            Provider::OpenRouter(p) => p.api_endpoint(),
            Provider::HuggingFace(p) => p.api_endpoint(),
            Provider::Xai(p) => p.api_endpoint(),
        }
    }

    fn api_key_env_var(&self) -> Option<&'static str> {
        match self {
            Provider::OpenAi(p) => p.api_key_env_var(),
            Provider::Mistral(p) => p.api_key_env_var(),
            Provider::Anthropic(p) => p.api_key_env_var(),
            Provider::Together(p) => p.api_key_env_var(),
            Provider::OpenRouter(p) => p.api_key_env_var(),
            Provider::HuggingFace(p) => p.api_key_env_var(),
            Provider::Xai(p) => p.api_key_env_var(),
        }
    }

    fn fetch_models(&self) -> AsyncStream<ModelData> {
        match self {
            Provider::OpenAi(p) => p.fetch_models(),
            Provider::Mistral(p) => p.fetch_models(),
            Provider::Anthropic(p) => p.fetch_models(),
            Provider::Together(p) => p.fetch_models(),
            Provider::OpenRouter(p) => p.fetch_models(),
            Provider::HuggingFace(p) => p.fetch_models(),
            Provider::Xai(p) => p.fetch_models(),
        }
    }

    fn static_models(&self) -> Option<Vec<ModelData>> {
        match self {
            Provider::OpenAi(p) => p.static_models(),
            Provider::Mistral(p) => p.static_models(),
            Provider::Anthropic(p) => p.static_models(),
            Provider::Together(p) => p.static_models(),
            Provider::OpenRouter(p) => p.static_models(),
            Provider::HuggingFace(p) => p.static_models(),
            Provider::Xai(p) => p.static_models(),
        }
    }

    fn generate_code(&self, models: &[ModelData]) -> Result<(String, String)> {
        match self {
            Provider::OpenAi(p) => p.generate_code(models),
            Provider::Mistral(p) => p.generate_code(models),
            Provider::Anthropic(p) => p.generate_code(models),
            Provider::Together(p) => p.generate_code(models),
            Provider::OpenRouter(p) => p.generate_code(models),
            Provider::HuggingFace(p) => p.generate_code(models),
            Provider::Xai(p) => p.generate_code(models),
        }
    }
}

/// Get all available provider builders
#[inline]
pub fn all_providers() -> Vec<Provider> {
    vec![
        Provider::OpenAi(openai::OpenAiProvider),
        Provider::Mistral(mistral::MistralProvider),
        Provider::Anthropic(anthropic::AnthropicProvider),
        Provider::Together(together::TogetherProvider),
        Provider::OpenRouter(openrouter::OpenRouterProvider),
        Provider::HuggingFace(huggingface::HuggingFaceProvider),
        Provider::Xai(xai::XaiProvider),
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

use dashmap::DashMap;
use once_cell::sync::Lazy;

/// Global registry for dynamic provider registration.
/// Allows runtime addition of custom providers for extensibility.
#[allow(dead_code)] // Library feature - extensibility for custom providers
pub static PROVIDER_REGISTRY: Lazy<DashMap<String, Box<dyn ProviderBuilder + Send + Sync>>> =
    Lazy::new(DashMap::new);

/// Register a new provider dynamically.
/// This can be called at runtime to add custom providers.
#[allow(dead_code)] // Library function - extensibility for custom providers
pub fn register_provider(name: String, provider: Box<dyn ProviderBuilder + Send + Sync>) {
    PROVIDER_REGISTRY.insert(name, provider);
}

/// Get all providers, including dynamically registered ones.
#[allow(dead_code)] // Library function - extensibility for custom providers
pub fn all_providers_extended() -> Vec<Box<dyn ProviderBuilder + Send + Sync>> {
    let providers: Vec<Box<dyn ProviderBuilder + Send + Sync>> = all_providers()
        .into_iter()
        .map(|p| Box::new(p) as Box<dyn ProviderBuilder + Send + Sync>)
        .collect();
    // Note: PROVIDER_REGISTRY is not included in default providers list
    // as it contains dynamic providers that need special handling
    providers
}
