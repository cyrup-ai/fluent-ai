//! Candle provider implementation that complies with domain traits

use fluent_ai_async::AsyncStream;
use model_info::{ProviderTrait, ModelInfo};

use super::models::CandleModel;

/// Candle provider for local ML model inference
#[derive(Debug, Clone)]
pub struct CandleProvider;

impl CandleProvider {
    /// Create a new Candle provider
    pub fn new() -> Self {
        Self
    }
}

impl ProviderTrait for CandleProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        AsyncStream::empty() // TODO: Implement actual model info retrieval
    }

    fn list_models(&self) -> AsyncStream<ModelInfo> {
        AsyncStream::empty() // TODO: Implement actual model listing
    }

    fn provider_name(&self) -> &'static str {
        "candle"
    }
}

impl Default for CandleProvider {
    fn default() -> Self {
        Self::new()
    }
}
