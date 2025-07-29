use crate::domain::model::CandleModelRegistry as ModelRegistry;
use crate::domain::model::error::CandleResult as Result;
use crate::domain::model::traits::CandleModel;

// Import RegisteredModel from the model registry
use crate::model::registry::RegisteredModel;

/// Builder for registering models with the global registry
pub struct ModelBuilder<M: CandleModel + 'static> {
    provider: &'static str,
    model: M,
}

impl<M: CandleModel + 'static> ModelBuilder<M> {
    /// Create a new model builder
    pub fn new(provider: &'static str, model: M) -> Self {
        Self { provider, model }
    }

    /// Register the model with the global registry
    pub fn register(self) -> Result<RegisteredModel<M>> {
        ModelRegistry::new().register(self.provider, self.model)
    }
}