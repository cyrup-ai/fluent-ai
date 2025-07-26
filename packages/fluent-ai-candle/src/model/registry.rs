//! Model registry for managing available models
//!
//! Contains the registry system for discovering and managing available models.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::types::CandleModel as Model;

/// Registry for managing available models
#[derive(Debug)]
pub struct ModelRegistry {
    /// Registered models by name
    models: RwLock<HashMap<String, Arc<dyn Model>>>,

    /// Model information by name
    info: RwLock<HashMap<String, &'static crate::types::CandleModelInfo>>}

impl ModelRegistry {
    /// Create a new empty model registry
    ///
    /// Initializes a new registry with empty storage for models and their metadata.
    /// The registry uses read-write locks for thread-safe concurrent access.
    ///
    /// # Returns
    ///
    /// A new `ModelRegistry` instance ready for use
    ///
    /// # Example
    ///
    /// ```rust
    /// let registry = ModelRegistry::new();
    /// assert!(registry.is_empty().unwrap());
    /// ```
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            info: RwLock::new(HashMap::new())}
    }

    /// Register a model in the registry with thread-safe storage
    ///
    /// Adds a model to the registry with the given name. The model must implement
    /// the `Model` trait. If a model with the same name already exists, returns
    /// an error without modifying the registry.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for the model
    /// * `model` - The model implementation to register
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or `RegistryError` if the model name already exists
    /// or if there's a lock contention issue.
    ///
    /// # Errors
    ///
    /// - `RegistryError::ModelAlreadyExists` - A model with this name is already registered
    /// - `RegistryError::LockError` - Failed to acquire write lock on internal storage
    ///
    /// # Example
    ///
    /// ```rust
    /// let registry = ModelRegistry::new();
    /// let model = MyModel::new();
    /// registry.register("my_model".to_string(), model)?;
    /// ```
    pub fn register<M: Model>(&self, name: String, model: M) -> Result<(), RegistryError> {
        let model = Arc::new(model);
        let info = model.info();

        {
            let mut models = self.models.write().map_err(|_| RegistryError::LockError)?;
            let mut info_map = self.info.write().map_err(|_| RegistryError::LockError)?;

            if models.contains_key(&name) {
                return Err(RegistryError::ModelAlreadyExists { name });
            }

            models.insert(name.clone(), model);
            info_map.insert(name, info);
        }

        Ok(())
    }

    /// Get a registered model by name with thread-safe access
    ///
    /// Retrieves a model from the registry by its registered name. Returns a
    /// reference-counted pointer to the model, allowing safe sharing across threads.
    ///
    /// # Arguments
    ///
    /// * `name` - The registered name of the model to retrieve
    ///
    /// # Returns
    ///
    /// `Arc<dyn Model>` if the model exists, or `RegistryError` if not found
    /// or if there's a lock contention issue.
    ///
    /// # Errors
    ///
    /// - `RegistryError::ModelNotFound` - No model registered with this name
    /// - `RegistryError::LockError` - Failed to acquire read lock on internal storage
    ///
    /// # Example
    ///
    /// ```rust
    /// let model = registry.get("my_model")?;
    /// let result = model.process_input("test input");
    /// ```
    pub fn get(&self, name: &str) -> Result<Arc<dyn Model>, RegistryError> {
        let models = self.models.read().map_err(|_| RegistryError::LockError)?;

        models
            .get(name)
            .cloned()
            .ok_or_else(|| RegistryError::ModelNotFound {
                name: name.to_string()})
    }

    /// Get model metadata information by registered name
    ///
    /// Retrieves the static metadata information for a registered model.
    /// This includes model capabilities, version, and other descriptive information
    /// without accessing the actual model instance.
    ///
    /// # Arguments
    ///
    /// * `name` - The registered name of the model
    ///
    /// # Returns
    ///
    /// Static reference to `CandleModelInfo` if the model exists, or `RegistryError`
    /// if not found or if there's a lock contention issue.
    ///
    /// # Errors
    ///
    /// - `RegistryError::ModelNotFound` - No model registered with this name
    /// - `RegistryError::LockError` - Failed to acquire read lock on internal storage
    ///
    /// # Example
    ///
    /// ```rust
    /// let info = registry.info("my_model")?;
    /// println!("Model version: {}", info.version);
    /// ```
    pub fn info(
        &self,
        name: &str,
    ) -> Result<&'static crate::types::CandleModelInfo, RegistryError> {
        let info = self.info.read().map_err(|_| RegistryError::LockError)?;

        info.get(name)
            .copied()
            .ok_or_else(|| RegistryError::ModelNotFound {
                name: name.to_string()})
    }

    /// List all registered model names in the registry
    ///
    /// Returns a vector containing the names of all currently registered models.
    /// The order of names is not guaranteed and may vary between calls.
    ///
    /// # Returns
    ///
    /// `Vec<String>` containing all registered model names, or `RegistryError`
    /// if there's a lock contention issue. Returns empty vector if no models are registered.
    ///
    /// # Errors
    ///
    /// - `RegistryError::LockError` - Failed to acquire read lock on internal storage
    ///
    /// # Example
    ///
    /// ```rust
    /// let model_names = registry.list()?;
    /// for name in model_names {
    ///     println!("Registered model: {}", name);
    /// }
    /// ```
    pub fn list(&self) -> Result<Vec<String>, RegistryError> {
        let models = self.models.read().map_err(|_| RegistryError::LockError)?;
        Ok(models.keys().cloned().collect())
    }

    /// Check if a model with the given name is registered
    ///
    /// Performs a membership test to determine if a model with the specified
    /// name exists in the registry without retrieving the model itself.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name to check for
    ///
    /// # Returns
    ///
    /// `true` if a model with this name is registered, `false` otherwise,
    /// or `RegistryError` if there's a lock contention issue.
    ///
    /// # Errors
    ///
    /// - `RegistryError::LockError` - Failed to acquire read lock on internal storage
    ///
    /// # Example
    ///
    /// ```rust
    /// if registry.contains("my_model")? {
    ///     let model = registry.get("my_model")?;
    ///     // Use the model...
    /// }
    /// ```
    pub fn contains(&self, name: &str) -> Result<bool, RegistryError> {
        let models = self.models.read().map_err(|_| RegistryError::LockError)?;
        Ok(models.contains_key(name))
    }

    /// Unregister and remove a model from the registry
    ///
    /// Removes a model from the registry by name and returns the model instance.
    /// This allows for cleanup or transfer of the model to another registry.
    /// Both the model and its metadata are removed atomically.
    ///
    /// # Arguments
    ///
    /// * `name` - The registered name of the model to remove
    ///
    /// # Returns
    ///
    /// The removed `Arc<dyn Model>` if successful, or `RegistryError` if the model
    /// doesn't exist or if there's a lock contention issue.
    ///
    /// # Errors
    ///
    /// - `RegistryError::ModelNotFound` - No model registered with this name
    /// - `RegistryError::LockError` - Failed to acquire write lock on internal storage
    ///
    /// # Example
    ///
    /// ```rust
    /// let removed_model = registry.unregister("my_model")?;
    /// // Model is no longer in registry but can still be used
    /// ```
    pub fn unregister(&self, name: &str) -> Result<Arc<dyn Model>, RegistryError> {
        let mut models = self.models.write().map_err(|_| RegistryError::LockError)?;
        let mut info = self.info.write().map_err(|_| RegistryError::LockError)?;

        let model = models
            .remove(name)
            .ok_or_else(|| RegistryError::ModelNotFound {
                name: name.to_string()})?;

        info.remove(name);

        Ok(model)
    }

    /// Clear all registered models from the registry
    ///
    /// Removes all models and their metadata from the registry, returning it
    /// to an empty state. This operation is atomic - either all models are
    /// removed or none are (in case of lock contention).
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or `RegistryError` if there's a lock contention issue.
    ///
    /// # Errors
    ///
    /// - `RegistryError::LockError` - Failed to acquire write lock on internal storage
    ///
    /// # Example
    ///
    /// ```rust
    /// registry.clear()?;
    /// assert!(registry.is_empty()?);
    /// ```
    pub fn clear(&self) -> Result<(), RegistryError> {
        let mut models = self.models.write().map_err(|_| RegistryError::LockError)?;
        let mut info = self.info.write().map_err(|_| RegistryError::LockError)?;

        models.clear();
        info.clear();

        Ok(())
    }

    /// Get the total number of registered models in the registry
    ///
    /// Returns the count of models currently registered in the registry.
    /// This operation is thread-safe and provides an atomic snapshot of the count.
    ///
    /// # Returns
    ///
    /// The number of registered models as `usize`, or `RegistryError`
    /// if there's a lock contention issue.
    ///
    /// # Errors
    ///
    /// - `RegistryError::LockError` - Failed to acquire read lock on internal storage
    ///
    /// # Example
    ///
    /// ```rust
    /// let count = registry.len()?;
    /// println!("Registry contains {} models", count);
    /// ```
    pub fn len(&self) -> Result<usize, RegistryError> {
        let models = self.models.read().map_err(|_| RegistryError::LockError)?;
        Ok(models.len())
    }

    /// Check if the registry contains no registered models
    ///
    /// Returns `true` if the registry is empty (contains no models), `false` otherwise.
    /// This is more efficient than checking if `len() == 0` for large registries.
    ///
    /// # Returns
    ///
    /// `true` if empty, `false` if contains models, or `RegistryError`
    /// if there's a lock contention issue.
    ///
    /// # Errors
    ///
    /// - `RegistryError::LockError` - Failed to acquire read lock on internal storage
    ///
    /// # Example
    ///
    /// ```rust
    /// if registry.is_empty()? {
    ///     println!("No models registered yet");
    /// }
    /// ```
    pub fn is_empty(&self) -> Result<bool, RegistryError> {
        let models = self.models.read().map_err(|_| RegistryError::LockError)?;
        Ok(models.is_empty())
    }

    /// Retrieve all registered models as a collection
    ///
    /// Returns a vector containing references to all currently registered models.
    /// This provides bulk access to all models without needing to iterate through
    /// names individually. The order of models is not guaranteed.
    ///
    /// # Returns
    ///
    /// `Vec<Arc<dyn Model>>` containing all registered models, or `RegistryError`
    /// if there's a lock contention issue. Returns empty vector if no models are registered.
    ///
    /// # Errors
    ///
    /// - `RegistryError::LockError` - Failed to acquire read lock on internal storage
    ///
    /// # Example
    ///
    /// ```rust
    /// let all_models = registry.find_all()?;
    /// for model in all_models {
    ///     // Process each model...
    /// }
    /// ```
    pub fn find_all(&self) -> Result<Vec<Arc<dyn Model>>, RegistryError> {
        let models = self.models.read().map_err(|_| RegistryError::LockError)?;
        Ok(models.values().cloned().collect())
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during registry operations
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// Model not found in registry
    #[error("Model not found: {name}")]
    ModelNotFound { name: String },

    /// Model already exists in registry
    #[error("Model already exists: {name}")]
    ModelAlreadyExists { name: String },

    /// Lock error (internal synchronization issue)
    #[error("Lock error: failed to acquire registry lock")]
    LockError,

    /// Invalid model name
    #[error("Invalid model name: {name}")]
    InvalidName { name: String }}

/// Global model registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<ModelRegistry> = std::sync::OnceLock::new();

/// Get the global singleton model registry instance
///
/// Returns a reference to the global model registry that can be used throughout
/// the application. The registry is initialized lazily on first access and remains
/// available for the lifetime of the program.
///
/// # Returns
///
/// Static reference to the global `ModelRegistry` instance
///
/// # Thread Safety
///
/// This function is thread-safe and multiple threads can safely access the
/// global registry simultaneously.
///
/// # Example
///
/// ```rust
/// let registry = global_registry();
/// registry.register("my_model".to_string(), model)?;
/// ```
pub fn global_registry() -> &'static ModelRegistry {
    GLOBAL_REGISTRY.get_or_init(ModelRegistry::new)
}

/// Register a model in the global registry (convenience function)
///
/// A convenience function that registers a model in the global registry instance.
/// This is equivalent to calling `global_registry().register(name, model)`.
///
/// # Arguments
///
/// * `name` - Unique identifier for the model
/// * `model` - The model implementation to register
///
/// # Returns
///
/// `Ok(())` if successful, or `RegistryError` if registration fails
///
/// # Errors
///
/// - `RegistryError::ModelAlreadyExists` - A model with this name is already registered
/// - `RegistryError::LockError` - Failed to acquire write lock on global registry
///
/// # Example
///
/// ```rust
/// let model = MyModel::new();
/// register_model("my_model".to_string(), model)?;
/// ```
pub fn register_model<M: Model>(name: String, model: M) -> Result<(), RegistryError> {
    global_registry().register(name, model)
}

/// Get a model from the global registry (convenience function)
///
/// A convenience function that retrieves a model from the global registry instance.
/// This is equivalent to calling `global_registry().get(name)`.
///
/// # Arguments
///
/// * `name` - The registered name of the model to retrieve
///
/// # Returns
///
/// `Arc<dyn Model>` if the model exists, or `RegistryError` if not found
///
/// # Errors
///
/// - `RegistryError::ModelNotFound` - No model registered with this name
/// - `RegistryError::LockError` - Failed to acquire read lock on global registry
///
/// # Example
///
/// ```rust
/// let model = get_model("my_model")?;
/// let result = model.process_input("test input");
/// ```
pub fn get_model(name: &str) -> Result<Arc<dyn Model>, RegistryError> {
    global_registry().get(name)
}

/// List all model names in the global registry (convenience function)
///
/// A convenience function that lists all registered model names from the global
/// registry instance. This is equivalent to calling `global_registry().list()`.
///
/// # Returns
///
/// `Vec<String>` containing all registered model names, or `RegistryError`
/// if there's a lock contention issue.
///
/// # Errors
///
/// - `RegistryError::LockError` - Failed to acquire read lock on global registry
///
/// # Example
///
/// ```rust
/// let model_names = list_models()?;
/// for name in model_names {
///     println!("Available model: {}", name);
/// }
/// ```
pub fn list_models() -> Result<Vec<String>, RegistryError> {
    global_registry().list()
}

/// Check if a model exists in the global registry (convenience function)
///
/// A convenience function that checks if a model with the given name exists in
/// the global registry instance. This is equivalent to calling `global_registry().contains(name)`.
///
/// # Arguments
///
/// * `name` - The model name to check for
///
/// # Returns
///
/// `true` if a model with this name is registered, `false` otherwise,
/// or `RegistryError` if there's a lock contention issue.
///
/// # Errors
///
/// - `RegistryError::LockError` - Failed to acquire read lock on global registry
///
/// # Example
///
/// ```rust
/// if model_exists("my_model")? {
///     let model = get_model("my_model")?;
///     // Use the model...
/// }
/// ```
pub fn model_exists(name: &str) -> Result<bool, RegistryError> {
    global_registry().contains(name)
}

impl Clone for ModelRegistry {
    fn clone(&self) -> Self {
        // Clone the data from the RwLocks
        let models = self.models.read().unwrap().clone();
        let info = self.info.read().unwrap().clone();

        Self {
            models: RwLock::new(models),
            info: RwLock::new(info)}
    }
}
