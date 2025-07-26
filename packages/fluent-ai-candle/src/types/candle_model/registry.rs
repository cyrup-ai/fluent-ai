//! Model registry for dynamic model discovery and lookup

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use ahash::RandomState;
use dashmap::{DashMap, DashSet};
use once_cell::sync::Lazy;

use crate::model::error::ModelError;
use crate::types::CandleCompletionResult as Result;
use crate::types::CandleModel;
use crate::types::CandleModelInfo as ModelInfo;

/// A type-erased model reference
struct ModelHandle {
    model: Box<dyn Any + Send + Sync>,
    info: &'static ModelInfo}

impl ModelHandle {
    fn new<M: CandleModel + 'static>(model: M) -> Self {
        let info = model.info();
        Self {
            model: Box::new(model),
            info}
    }

    fn as_any(&self) -> &dyn Any {
        &*self.model
    }

    fn as_model<M: CandleModel + 'static>(&self) -> Option<&M> {
        self.model.downcast_ref::<M>()
    }

    fn info(&self) -> &'static ModelInfo {
        self.info
    }
}

/// The global model registry
struct ModelRegistryInner {
    // Maps provider name to model name to model handle
    models:
        DashMap<&'static str, DashMap<&'static str, Arc<ModelHandle>, RandomState>, RandomState>,

    // Maps model type to provider+name
    type_registry: DashMap<TypeId, DashSet<(&'static str, &'static str), RandomState>, RandomState>}

impl Default for ModelRegistryInner {
    fn default() -> Self {
        Self {
            models: DashMap::with_hasher(RandomState::default()),
            type_registry: DashMap::with_hasher(RandomState::default())}
    }
}

/// The global model registry
static GLOBAL_REGISTRY: Lazy<ModelRegistryInner> = Lazy::new(Default::default);

/// A registry for managing model instances
///
/// This provides a thread-safe way to register, look up, and manage model instances.
/// It supports dynamic model loading and type-safe retrieval.
#[derive(Clone, Default)]
pub struct ModelRegistry;

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self
    }

    /// Register a model with the registry
    ///
    /// # Arguments
    /// * `provider` - The provider name (e.g., "openai", "anthropic")
    /// * `model` - The model instance to register
    ///
    /// # Returns
    /// A result containing the registered model or an error if registration fails
    pub fn register<M: CandleModel + 'static>(
        &self,
        provider: &'static str,
        model: M,
    ) -> Result<RegisteredModel<M>> {
        let handle = Arc::new(ModelHandle::new(model));
        let model_name = handle.info().name();

        // Get or create the provider's model map
        let provider_models = GLOBAL_REGISTRY
            .models
            .entry(provider)
            .or_insert_with(|| DashMap::with_hasher(RandomState::default()));

        // Check for duplicate model
        if provider_models.contains_key(model_name) {
            return Err(ModelError::ModelAlreadyExists {
                name: format!("{}::{}", provider, model_name).into()}
            .into());
        }

        // Register the model
        provider_models.insert(model_name, handle.clone());

        // Register the model type
        let type_id = TypeId::of::<M>();
        let type_entries = GLOBAL_REGISTRY
            .type_registry
            .entry(type_id)
            .or_insert_with(|| DashSet::with_hasher(RandomState::default()));

        type_entries.insert((provider, model_name));

        // Return a registered model handle
        Ok(RegisteredModel {
            handle,
            _marker: PhantomData})
    }

    /// Get a model by provider and name
    ///
    /// # Arguments
    /// * `provider` - The provider name
    /// * `name` - The model name
    ///
    /// # Returns
    /// A result containing the model if found, or an error if not found or type mismatch
    pub fn get<M: CandleModel + 'static>(
        &self,
        provider: &str,
        name: &str,
    ) -> Result<Option<RegisteredModel<M>>> {
        let provider_models = match GLOBAL_REGISTRY.models.get(provider) {
            Some(provider) => provider,
            None => return Ok(None)};

        let handle = match provider_models.get(name) {
            Some(handle) => handle,
            None => return Ok(None)};

        // Verify the model type
        if handle.as_any().downcast_ref::<M>().is_none() {
            return Err(ModelError::InvalidConfiguration(
                "model type does not match requested type".into(),
            )
            .into());
        }

        Ok(Some(RegisteredModel {
            handle: handle.clone(),
            _marker: PhantomData}))
    }

    /// Get a model by provider and name, returning an error if not found
    ///
    /// # Arguments
    /// * `provider` - The provider name
    /// * `name` - The model name
    ///
    /// # Returns
    /// A result containing the model if found, or an error if not found or type mismatch
    pub fn get_required<M: CandleModel + 'static>(
        &self,
        provider: &'static str,
        name: &'static str,
    ) -> Result<RegisteredModel<M>> {
        self.get(provider, name)?.ok_or_else(|| {
            ModelError::ModelNotFound {
                provider: provider.into(),
                name: name.into()}
            .into()
        })
    }

    /// Find all models of a specific type
    ///
    /// # Returns
    /// A vector of registered models of the specified type
    pub fn find_all<M: CandleModel + 'static>(&self) -> Vec<RegisteredModel<M>> {
        let type_id = TypeId::of::<M>();
        let mut result = Vec::new();

        if let Some(type_entries) = GLOBAL_REGISTRY.type_registry.get(&type_id) {
            for entry in type_entries.iter() {
                let (provider, name) = *entry;
                if let Some(provider_models) = GLOBAL_REGISTRY.models.get(provider) {
                    if let Some(handle) = provider_models.get(name) {
                        if handle.as_any().downcast_ref::<M>().is_some() {
                            result.push(RegisteredModel {
                                handle: handle.clone(),
                                _marker: PhantomData});
                        }
                    }
                }
            }
        }

        result
    }

    /// Get a model as a specific trait object
    ///
    /// # Arguments
    /// * `provider` - The provider name
    /// * `name` - The model name
    ///
    /// # Returns
    /// A result containing the model as the requested trait object
    pub fn get_as<T: 'static>(
        &self,
        provider: &'static str,
        name: &'static str,
    ) -> Result<Option<Arc<T>>>
    where
        T: Send + Sync + Sized,
    {
        let provider_models = match GLOBAL_REGISTRY.models.get(provider) {
            Some(provider) => provider,
            None => return Ok(None)};

        let handle = match provider_models.get(name) {
            Some(handle) => handle,
            None => return Ok(None)};

        // Attempt to downcast the handle to the requested type
        match handle.as_any().downcast_ref::<T>() {
            Some(_) => {
                // For now, this method is not fully implemented due to Arc<T> conversion complexity
                Err(ModelError::InvalidConfiguration(
                    format!("Model downcast for '{}' from provider '{}' requires additional implementation", name, provider).into()
                ).into())
            }
            None => Err(ModelError::InvalidConfiguration(
                format!(
                    "Model '{}' from provider '{}' is not of the requested type",
                    name, provider
                )
                .into(),
            )
            .into())}
    }

    /// Get a model as a boxed trait object
    ///
    /// # Arguments
    /// * `provider` - The provider name
    /// * `name` - The model name
    ///
    /// # Returns
    /// A result containing the model as a boxed trait object
    pub fn get_boxed<T: 'static + ?Sized>(
        &self,
        provider: &'static str,
        name: &'static str,
    ) -> Result<Option<Box<T>>>
    where
        T: Send + Sync,
    {
        let provider_models = match GLOBAL_REGISTRY.models.get(provider) {
            Some(provider) => provider,
            None => return Ok(None)};

        let _handle = match provider_models.get(name) {
            Some(handle) => handle,
            None => return Ok(None)};

        // Attempt to convert the handle to a boxed trait object
        // This is complex for ?Sized types and requires careful implementation
        Err(ModelError::InvalidConfiguration(Cow::Owned(format!(
            "Boxed trait object conversion for model '{}' from provider '{}' requires additional implementation for ?Sized types",
            name, provider
        ))).into())
    }

    /// Get a model as a specific trait object, returning an error if not found
    ///
    /// # Arguments
    /// * `provider` - The provider name
    /// * `name` - The model name
    ///
    /// # Returns
    /// A result containing the model as the requested trait object
    pub fn get_required_as<T: 'static>(
        &self,
        provider: &'static str,
        name: &'static str,
    ) -> Result<Arc<T>>
    where
        T: Send + Sync + Sized,
    {
        self.get_as(provider, name)?.ok_or_else(|| {
            ModelError::ModelNotFound {
                provider: provider.into(),
                name: name.into()}
            .into()
        })
    }

    /// Get a model as a boxed trait object, returning an error if not found
    ///
    /// # Arguments
    /// * `provider` - The provider name
    /// * `name` - The model name
    ///
    /// # Returns
    /// A result containing the model as a boxed trait object
    pub fn get_required_boxed<T: 'static>(
        &self,
        provider: &'static str,
        name: &'static str,
    ) -> Result<Box<T>>
    where
        T: Send + Sync + Sized,
    {
        self.get_boxed(provider, name)?.ok_or_else(|| {
            ModelError::ModelNotFound {
                provider: provider.into(),
                name: name.into()}
            .into()
        })
    }
}

/// A handle to a registered model
///
/// This provides type-safe access to a registered model and ensures
/// proper cleanup when the last reference is dropped.
pub struct RegisteredModel<M: CandleModel + 'static> {
    handle: Arc<ModelHandle>,
    _marker: PhantomData<M>}

impl<M: CandleModel + 'static> Clone for RegisteredModel<M> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            _marker: PhantomData}
    }
}

impl<M: CandleModel + 'static> std::ops::Deref for RegisteredModel<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        self.handle
            .as_model()
            .expect("type mismatch in RegisteredModel")
    }
}

impl<M: CandleModel + 'static> AsRef<M> for RegisteredModel<M> {
    fn as_ref(&self) -> &M {
        self.handle
            .as_model()
            .expect("type mismatch in RegisteredModel")
    }
}

impl<M: CandleModel + 'static> std::fmt::Debug for RegisteredModel<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredModel")
            .field("provider", &self.info().provider())
            .field("name", &self.info().name())
            .finish()
    }
}

impl<M: CandleModel + 'static> PartialEq for RegisteredModel<M> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.handle, &other.handle)
    }
}

impl<M: CandleModel + 'static> Eq for RegisteredModel<M> {}

impl<M: CandleModel + 'static> Hash for RegisteredModel<M> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.info().provider().hash(state);
        self.handle.info().name().hash(state);
    }
}

/// A builder for configuring and registering models
pub struct ModelBuilder<M: CandleModel + 'static> {
    provider: &'static str,
    model: M}

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
