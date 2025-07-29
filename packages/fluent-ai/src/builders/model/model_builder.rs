use fluent_ai_domain::model::{Model, ModelRegistry, registry::RegisteredModel, Result};

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