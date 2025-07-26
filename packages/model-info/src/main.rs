pub mod common;
pub mod providers;

use common::{ModelInfo, ProviderTrait};
use providers::{
    anthropic::AnthropicProvider, huggingface::HuggingFaceProvider, mistral::MistralProvider,
    openai::OpenAiProvider, openrouter::OpenRouterProvider, together::TogetherProvider,
    xai::XaiProvider,
};
use anyhow::Result;

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
    pub async fn get_model_info(&self, model: &str) -> Result<ModelInfo> {
        match self {
            Provider::OpenAi(p) => p.get_model_info(model).await,
            Provider::Mistral(p) => p.get_model_info(model).await,
            Provider::Anthropic(p) => p.get_model_info(model).await,
            Provider::Together(p) => p.get_model_info(model).await,
            Provider::OpenRouter(p) => p.get_model_info(model).await,
            Provider::HuggingFace(p) => p.get_model_info(model).await,
            Provider::Xai(p) => p.get_model_info(model).await,
        }
    }
}

fn main() {
    println!("Hello, world!");
}
