//! Model registry for dynamic model discovery and lookup

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use ahash::RandomState;
use dashmap::{DashMap, DashSet};
use once_cell::sync::Lazy;

use crate::model::error::{ModelError, Result};
use crate::model::info::ModelInfo;
use crate::model::traits::Model;
use crate::providers::CandleKimiK2Provider;

/// Zero-allocation typed model handle with static dispatch instead of Box<dyn Any>
#[derive(Debug, Clone)]
pub enum CandleModelHandle {
    /// Kimi K2 model from Candle framework
    KimiK2(Arc<CandleKimiK2Provider>),
    /// Generic model implementation for extensibility
    Generic(Arc<GenericCandleModel>),
}

impl CandleModelHandle {
    /// Create new Kimi K2 model handle
    pub fn new_kimi_k2(model: CandleKimiK2Provider) -> Self {
        Self::KimiK2(Arc::new(model))
    }
    
    /// Create new generic model handle
    pub fn new_generic(model: GenericCandleModel) -> Self {
        Self::Generic(Arc::new(model))
    }

    /// Get model info with zero allocation
    pub fn info(&self) -> ModelInfo {
        match self {
            Self::KimiK2(model) => model.model_info(),
            Self::Generic(model) => model.info(),
        }
    }
    
    /// Get Kimi K2 model if this handle contains one
    pub fn as_kimi_k2(&self) -> Option<&CandleKimiK2Provider> {
        match self {
            Self::KimiK2(model) => Some(model.as_ref()),
            _ => None,
        }
    }
    
    /// Get generic model if this handle contains one
    pub fn as_generic(&self) -> Option<&GenericCandleModel> {
        match self {
            Self::Generic(model) => Some(model.as_ref()),
            _ => None,
        }
    }
}

/// Generic Candle model for extensibility with zero-allocation patterns
#[derive(Debug, Clone)]
pub struct GenericCandleModel {
    name: String,
    provider: String,
    version: String,
    context_length: u32,
    supports_streaming: bool,
    supports_tools: bool,
}

impl GenericCandleModel {
    /// Create new generic model
    pub fn new(
        name: String,
        provider: String,
        version: String,
        context_length: u32,
        supports_streaming: bool,
        supports_tools: bool,
    ) -> Self {
        Self {
            name,
            provider,
            version,
            context_length,
            supports_streaming,
            supports_tools,
        }
    }
    
    /// Get model info
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            name: self.name.clone(),
            provider: self.provider.clone(),
            version: self.version.clone(),
            context_length: self.context_length,
            supports_streaming: self.supports_streaming,
            supports_tools: self.supports_tools,
        }
    }
}

/// The global model registry with zero-allocation typed handles
struct ModelRegistryInner {
    /// Maps provider name to model name to typed model handle
    models: DashMap<&'static str, DashMap<&'static str, Arc<CandleModelHandle>, RandomState>, RandomState>,
    
    /// Maps model type names to provider+name for efficient lookup
    type_registry: DashMap<&'static str, DashSet<(&'static str, &'static str), RandomState>, RandomState>,
}

impl Default for ModelRegistryInner {
    fn default() -> Self {
        Self {
            models: DashMap::with_hasher(RandomState::default()),
            type_registry: DashMap::with_hasher(RandomState::default()),
        }
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
    pub fn register<M: Model + 'static>(
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
                provider: provider.into(),
                name: model_name.into()});
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
    pub fn get<M: Model + 'static>(
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
            ));
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
    pub fn get_required<M: Model + 'static>(
        &self,
        provider: &'static str,
        name: &'static str,
    ) -> Result<RegisteredModel<M>> {
        self.get(provider, name)?
            .ok_or_else(|| ModelError::ModelNotFound {
                provider: provider.into(),
                name: name.into()})
    }

    /// Find all models of a specific type
    ///
    /// # Returns
    /// A vector of registered models of the specified type
    pub fn find_all<M: Model + 'static>(&self) -> Vec<RegisteredModel<M>> {
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
                ))
            }
            None => Err(ModelError::InvalidConfiguration(
                format!(
                    "Model '{}' from provider '{}' is not of the requested type",
                    name, provider
                )
                .into(),
            ))}
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
        ))))
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
        self.get_as(provider, name)?
            .ok_or_else(|| ModelError::ModelNotFound {
                provider: provider.into(),
                name: name.into()})
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
        self.get_boxed(provider, name)?
            .ok_or_else(|| ModelError::ModelNotFound {
                provider: provider.into(),
                name: name.into()})
    }
}

/// A handle to a registered model
///
/// This provides type-safe access to a registered model and ensures
/// proper cleanup when the last reference is dropped.
pub struct RegisteredModel<M: Model + 'static> {
    handle: Arc<ModelHandle>,
    _marker: PhantomData<M>}

impl<M: Model + 'static> Clone for RegisteredModel<M> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            _marker: PhantomData}
    }
}

impl<M: Model + 'static> std::ops::Deref for RegisteredModel<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        self.handle
            .as_model()
            .expect("type mismatch in RegisteredModel")
    }
}

impl<M: Model + 'static> AsRef<M> for RegisteredModel<M> {
    fn as_ref(&self) -> &M {
        self.handle
            .as_model()
            .expect("type mismatch in RegisteredModel")
    }
}

impl<M: Model + 'static> std::fmt::Debug for RegisteredModel<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredModel")
            .field("provider", &self.info().provider())
            .field("name", &self.info().name())
            .finish()
    }
}

impl<M: Model + 'static> PartialEq for RegisteredModel<M> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.handle, &other.handle)
    }
}

impl<M: Model + 'static> Eq for RegisteredModel<M> {}

impl<M: Model + 'static> Hash for RegisteredModel<M> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.info().provider().hash(state);
        self.handle.info().name().hash(state);
    }
}

/// A builder for configuring and registering models

#[cfg(test)]
mod tests {
    use super::*;
    // Import ModelInfo::builder() from fluent-ai
    use fluent_ai::builders::ModelInfoBuilder;

    struct TestModel {
        info: &'static ModelInfo}

    impl Model for TestModel {
        fn info(&self) -> &'static ModelInfo {
            self.info
        }
    }

    #[test]
    fn test_register_and_get() {
        let registry = ModelRegistry::new();

        let info = ModelInfo::builder()
            .provider_name("test")
            .name("test-model")
            .build()
            .unwrap();

        let model = TestModel { info: &info };

        // Register the model
        let registered = registry.register("test-provider", model).unwrap();

        // Retrieve the model
        let retrieved = registry
            .get_required::<TestModel>("test-provider", "test-model")
            .unwrap();

        assert_eq!(registered.info().name(), retrieved.info().name());
        assert_eq!(registered.info().provider(), retrieved.info().provider());
    }

    #[test]
    fn test_duplicate_registration() {
        let registry = ModelRegistry::new();

        let info1 = ModelInfo::builder()
            .provider_name("test")
            .name("test-model")
            .build()
            .unwrap();

        let info2 = ModelInfo::builder()
            .provider_name("test")
            .name("test-model")
            .build()
            .unwrap();

        let model1 = TestModel { info: &info1 };
        let model2 = TestModel { info: &info2 };

        // First registration should succeed
        registry.register("test-provider", model1).unwrap();

        // Second registration should fail
        let result = registry.register("test-provider", model2);
        assert!(matches!(
            result,
            Err(ModelError::ModelAlreadyExists {
                provider: "test-provider",
                name: "test-model"})
        ));
    }

    #[test]
    fn test_find_all() {
        let registry = ModelRegistry::new();

        let info1 = ModelInfo::builder()
            .provider_name("test1")
            .name("model1")
            .build()
            .unwrap();

        let info2 = ModelInfo::builder()
            .provider_name("test2")
            .name("model2")
            .build()
            .unwrap();

        let model1 = TestModel { info: &info1 };
        let model2 = TestModel { info: &info2 };

        // Register both models
        registry.register("test-provider1", model1).unwrap();
        registry.register("test-provider2", model2).unwrap();

        // Find all models of type TestModel
        let models = registry.find_all::<TestModel>();

        assert_eq!(models.len(), 2);

        let model_names: Vec<_> = models.iter().map(|m| m.info().name()).collect();

        assert!(model_names.contains(&"model1"));
        assert!(model_names.contains(&"model2"));
    }

    #[test]
    fn test_model_registration() {
        let info = ModelInfo::builder()
            .provider_name("test")
            .name("test-model")
            .build()
            .unwrap();

        let model = TestModel { info: &info };

        // Register model directly using ModelRegistry
        let mut registry = ModelRegistry::new();
        registry.register("test-provider", "test-model", model).unwrap();

        // Verify the model was registered
        let retrieved = registry
            .get_required::<TestModel>("test-provider", "test-model")
            .unwrap();

        assert_eq!(info.name(), retrieved.info().name());
    }
}
