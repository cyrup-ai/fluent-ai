//! Candle provider implementation that complies with domain traits

use fluent_ai_domain::{Provider, ZeroOneOrMany};

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

impl Provider for CandleProvider {
    type Model = CandleModel;

    fn name(&self) -> &'static str {
        "candle"
    }

    fn models(&self) -> ZeroOneOrMany<Self::Model> {
        // NOTE: Model enumeration is handled by model-info package
        // Providers should not enumerate models - they provide capabilities
        ZeroOneOrMany::Many(vec![])
    }
}

impl Default for CandleProvider {
    fn default() -> Self {
        Self::new()
    }
}
