use crate::domain::model::{CandleModel as Model, CandleModelRegistry as ModelRegistry};
use crate::domain::model::error::CandleResult as Result;

// Alias for RegisteredModel - could be same as CandleModel for now
type RegisteredModel = Model;

/// Builder for registering models with the global registry
pub struct ModelBuilder<M: Model + 'static> {
    provider: &'static str,
    model: M,
}

impl<M: Model + 'static> ModelBuilder<M> {
    /// Create a new model builder
    pub fn new(provider: &'static str, model: M) -> Self {
        Self { provider, model }
    }

    /// Register the model with the global registry
    pub fn register(self) -> Result<RegisteredModel<M>> {
        ModelRegistry::new().register(self.provider, self.model)
    }
}