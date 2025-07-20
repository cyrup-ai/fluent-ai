//! Model registry for dynamic model discovery and lookup

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Weak};

use ahash::RandomState;
use dashmap::{DashMap, DashSet};
use once_cell::sync::Lazy;
use parking_lot::RwLock as ParkingRwLock;

use crate::model::error::{ModelError, Result};
use crate::model::info::ModelInfo;
use crate::model::traits::{
    AnyChatCompletionCapable, AnyEmbeddingCapable, AnyModel, AnyTextGenerationCapable, Model,
};

/// A type-erased model reference
struct ModelHandle {
    model: Box<dyn Any + Send + Sync>,
    info: &'static ModelInfo,
    ref_count: AtomicUsize,
}

impl ModelHandle {
    fn new<M: Model + 'static>(model: M) -> Self {
        let info = model.info();
        Self {
            model: Box::new(model),
            info,
            ref_count: AtomicUsize::new(1),
        }
    }

    fn as_any(&self) -> &dyn Any {
        &*self.model
    }

    fn as_model<M: Model + 'static>(&self) -> Option<&M> {
        self.model.downcast_ref::<M>()
    }

    fn info(&self) -> &'static ModelInfo {
        self.info
    }

    fn clone_handle(&self) -> Self {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
        Self {
            model: self.model.clone(),
            info: self.info,
            ref_count: AtomicUsize::new(1),
        }
    }
}

/// A weak reference to a model handle
struct WeakModelHandle {
    model: Option<Weak<dyn Any + Send + Sync>>,
    info: &'static ModelInfo,
}

/// The global model registry
struct ModelRegistryInner {
    // Maps provider name to model name to model handle
    models:
        DashMap<&'static str, DashMap<&'static str, Arc<ModelHandle>, RandomState>, RandomState>,

    // Maps model type to provider+name
    type_registry: DashMap<TypeId, DashSet<(&'static str, &'static str), RandomState>, RandomState>,

    // For cleanup when the last strong reference is dropped
    cleanup: ParkingRwLock<Vec<Box<dyn Fn() + Send + Sync>>>,
}

impl Default for ModelRegistryInner {
    fn default() -> Self {
        Self {
            models: DashMap::with_hasher(RandomState::default()),
            type_registry: DashMap::with_hasher(RandomState::default()),
            cleanup: ParkingRwLock::new(Vec::new()),
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
                name: model_name.into(),
            });
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

        // Set up cleanup when the last strong reference is dropped
        let provider = provider.to_owned();
        let model_name = model_name.to_owned();

        let cleanup: Box<dyn Fn() + Send + Sync> = Box::new(move || {
            let type_id = TypeId::of::<M>();

            if let Some(mut provider_models) = GLOBAL_REGISTRY.models.get_mut(provider.as_str()) {
                provider_models.remove(model_name.as_str());

                // Clean up empty provider
                if provider_models.is_empty() {
                    GLOBAL_REGISTRY.models.remove(provider.as_str());
                }
            }

            // Clean up type registry
            if let Some(mut type_entries) = GLOBAL_REGISTRY.type_registry.get_mut(&type_id) {
                type_entries.remove(&(&provider[..], &model_name[..]));

                // Clean up empty type registry
                if type_entries.is_empty() {
                    GLOBAL_REGISTRY.type_registry.remove(&type_id);
                }
            }
        });

        // Store the cleanup function
        GLOBAL_REGISTRY.cleanup.write().push(Box::new(cleanup));

        // Return a registered model handle
        Ok(RegisteredModel {
            handle,
            _marker: PhantomData,
        })
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
            None => return Ok(None),
        };

        let handle = match provider_models.get(name) {
            Some(handle) => handle,
            None => return Ok(None),
        };

        // Verify the model type
        if handle.as_any().downcast_ref::<M>().is_none() {
            return Err(ModelError::InvalidConfiguration(
                "model type does not match requested type".into(),
            ));
        }

        Ok(Some(RegisteredModel {
            handle: handle.clone(),
            _marker: PhantomData,
        }))
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
                name: name.into(),
            })
    }

    /// Find all models of a specific type
    ///
    /// # Returns
    /// A vector of registered models of the specified type
    pub fn find_all<M: Model + 'static>(&self) -> Vec<RegisteredModel<M>> {
        let type_id = TypeId::of::<M>();
        let mut result = Vec::new();

        if let Some(type_entries) = GLOBAL_REGISTRY.type_registry.get(&type_id) {
            for (provider, name) in type_entries.iter() {
                if let Some(provider_models) = GLOBAL_REGISTRY.models.get(provider) {
                    if let Some(handle) = provider_models.get(name) {
                        if handle.as_any().downcast_ref::<M>().is_some() {
                            result.push(RegisteredModel {
                                handle: handle.clone(),
                                _marker: PhantomData,
                            });
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
    pub fn get_as<T: ?Sized + 'static>(
        &self,
        provider: &'static str,
        name: &'static str,
    ) -> Result<Option<Arc<T>>>
    where
        T: Send + Sync,
    {
        let provider_models = match GLOBAL_REGISTRY.models.get(provider) {
            Some(provider) => provider,
            None => return Ok(None),
        };

        let handle = match provider_models.get(name) {
            Some(handle) => handle,
            None => return Ok(None),
        };

        // Try to downcast to the requested type
        match handle.as_any().downcast_ref::<T>() {
            Some(model) => Ok(Some(Arc::new(model.clone()))),
            None => Err(ModelError::InvalidConfiguration(
                "model cannot be converted to the requested trait object".into(),
            )),
        }
    }

    /// Get a model as a boxed trait object
    ///
    /// # Arguments
    /// * `provider` - The provider name
    /// * `name` - The model name
    ///
    /// # Returns
    /// A result containing the model as a boxed trait object
    pub fn get_boxed<T: ?Sized + 'static>(
        &self,
        provider: &'static str,
        name: &'static str,
    ) -> Result<Option<Box<T>>>
    where
        T: Send + Sync,
    {
        let provider_models = match GLOBAL_REGISTRY.models.get(provider) {
            Some(provider) => provider,
            None => return Ok(None),
        };

        let handle = match provider_models.get(name) {
            Some(handle) => handle,
            None => return Ok(None),
        };

        // Try to downcast to the requested type
        match handle.as_any().downcast_ref::<T>() {
            Some(model) => Ok(Some(Box::new(model.clone()))),
            None => Err(ModelError::InvalidConfiguration(
                "model cannot be converted to the requested trait object".into(),
            )),
        }
    }

    /// Get a model as a specific trait object, returning an error if not found
    ///
    /// # Arguments
    /// * `provider` - The provider name
    /// * `name` - The model name
    ///
    /// # Returns
    /// A result containing the model as the requested trait object
    pub fn get_required_as<T: ?Sized + 'static>(
        &self,
        provider: &'static str,
        name: &'static str,
    ) -> Result<Arc<T>>
    where
        T: Send + Sync,
    {
        self.get_as(provider, name)?
            .ok_or_else(|| ModelError::ModelNotFound {
                provider: provider.into(),
                name: name.into(),
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
    pub fn get_required_boxed<T: ?Sized + 'static>(
        &self,
        provider: &'static str,
        name: &'static str,
    ) -> Result<Box<T>>
    where
        T: Send + Sync,
    {
        self.get_boxed(provider, name)?
            .ok_or_else(|| ModelError::ModelNotFound {
                provider: provider.into(),
                name: name.into(),
            })
    }
}

/// A handle to a registered model
///
/// This provides type-safe access to a registered model and ensures
/// proper cleanup when the last reference is dropped.
pub struct RegisteredModel<M: Model + 'static> {
    handle: Arc<ModelHandle>,
    _marker: PhantomData<M>,
}

impl<M: Model + 'static> Clone for RegisteredModel<M> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            _marker: PhantomData,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::info::ModelInfoBuilder;

    struct TestModel {
        info: &'static ModelInfo,
    }

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
                name: "test-model",
            })
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
    fn test_model_builder() {
        let info = ModelInfo::builder()
            .provider_name("test")
            .name("test-model")
            .build()
            .unwrap();

        let model = TestModel { info: &info };

        // Create a model builder
        let builder = ModelBuilder::new("test-provider", model);

        // Register the model using the builder
        let registered = builder.register().unwrap();

        // Verify the model was registered
        let retrieved = ModelRegistry::new()
            .get_required::<TestModel>("test-provider", "test-model")
            .unwrap();

        assert_eq!(registered.info().name(), retrieved.info().name());
    }
}
