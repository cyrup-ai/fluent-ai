//! Candle provider implementation that complies with domain traits

use fluent_ai_domain::{Provider, ZeroOneOrMany};

use super::models::CandleModel;

/// Candle provider for local ML model inference
#[derive(Debug, Clone)]
pub struct CandleProvider {
    /// Available Candle models
    models: Vec<CandleModel>,
}

impl CandleProvider {
    /// Create a new Candle provider
    pub fn new() -> Self {
        Self {
            models: vec![
                CandleModel::Llama2_7B,
                CandleModel::Llama2_13B,
                CandleModel::Mistral_7B,
                CandleModel::CodeLlama_7B,
            ],
        }
    }

    /// Create a provider with specific models
    pub fn with_models(models: Vec<CandleModel>) -> Self {
        Self { models }
    }
}

impl Provider for CandleProvider {
    type Model = CandleModel;

    fn name(&self) -> &'static str {
        "candle"
    }

    fn models(&self) -> ZeroOneOrMany<Self::Model> {
        ZeroOneOrMany::Many(self.models.clone())
    }
}

impl Default for CandleProvider {
    fn default() -> Self {
        Self::new()
    }
}
