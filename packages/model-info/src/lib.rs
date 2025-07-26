pub mod common;
pub mod providers;

// Re-export common types for external use
pub use common::{ModelInfo, ProviderTrait, Model};
use providers::{
    anthropic::AnthropicProvider, huggingface::HuggingFaceProvider, mistral::MistralProvider,
    openai::OpenAiProvider, openrouter::OpenRouterProvider, together::TogetherProvider,
    xai::XaiProvider,
};
use fluent_ai_async::AsyncStream;

include!(concat!(env!("OUT_DIR"), "/generated_models.rs"));

#[derive(Clone)]
pub enum Provider {
    OpenAi(OpenAiProvider),
    Mistral(MistralProvider),
    Anthropic(AnthropicProvider),
    Together(TogetherProvider),
    OpenRouter(OpenRouterProvider),
    HuggingFace(HuggingFaceProvider),
    Xai(XaiProvider),
}

impl Provider {
    pub fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        match self {
            Provider::OpenAi(p) => p.get_model_info(model),
            Provider::Mistral(p) => p.get_model_info(model),
            Provider::Anthropic(p) => p.get_model_info(model),
            Provider::Together(p) => p.get_model_info(model),
            Provider::OpenRouter(p) => p.get_model_info(model),
            Provider::HuggingFace(p) => p.get_model_info(model),
            Provider::Xai(p) => p.get_model_info(model),
        }
    }
}