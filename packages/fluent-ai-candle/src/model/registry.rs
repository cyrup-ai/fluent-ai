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
    info: RwLock<HashMap<String, &'static crate::types::CandleModelInfo>>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            info: RwLock::new(HashMap::new()),
        }
    }
    
    /// Register a model in the registry
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
    
    /// Get a model by name
    pub fn get(&self, name: &str) -> Result<Arc<dyn Model>, RegistryError> {
        let models = self.models.read().map_err(|_| RegistryError::LockError)?;
        
        models.get(name)
            .cloned()
            .ok_or_else(|| RegistryError::ModelNotFound { 
                name: name.to_string() 
            })
    }
    
    /// Get model information by name
    pub fn info(&self, name: &str) -> Result<&'static crate::types::CandleModelInfo, RegistryError> {
        let info = self.info.read().map_err(|_| RegistryError::LockError)?;
        
        info.get(name)
            .copied()
            .ok_or_else(|| RegistryError::ModelNotFound { 
                name: name.to_string() 
            })
    }
    
    /// List all registered model names
    pub fn list(&self) -> Result<Vec<String>, RegistryError> {
        let models = self.models.read().map_err(|_| RegistryError::LockError)?;
        Ok(models.keys().cloned().collect())
    }
    
    /// Check if a model is registered
    pub fn contains(&self, name: &str) -> Result<bool, RegistryError> {
        let models = self.models.read().map_err(|_| RegistryError::LockError)?;
        Ok(models.contains_key(name))
    }
    
    /// Unregister a model
    pub fn unregister(&self, name: &str) -> Result<Arc<dyn Model>, RegistryError> {
        let mut models = self.models.write().map_err(|_| RegistryError::LockError)?;
        let mut info = self.info.write().map_err(|_| RegistryError::LockError)?;
        
        let model = models.remove(name)
            .ok_or_else(|| RegistryError::ModelNotFound { 
                name: name.to_string() 
            })?;
        
        info.remove(name);
        
        Ok(model)
    }
    
    /// Clear all registered models
    pub fn clear(&self) -> Result<(), RegistryError> {
        let mut models = self.models.write().map_err(|_| RegistryError::LockError)?;
        let mut info = self.info.write().map_err(|_| RegistryError::LockError)?;
        
        models.clear();
        info.clear();
        
        Ok(())
    }
    
    /// Get the number of registered models
    pub fn len(&self) -> Result<usize, RegistryError> {
        let models = self.models.read().map_err(|_| RegistryError::LockError)?;
        Ok(models.len())
    }
    
    /// Check if the registry is empty
    pub fn is_empty(&self) -> Result<bool, RegistryError> {
        let models = self.models.read().map_err(|_| RegistryError::LockError)?;
        Ok(models.is_empty())
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
    InvalidName { name: String },
}

/// Global model registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<ModelRegistry> = std::sync::OnceLock::new();

/// Get the global model registry
pub fn global_registry() -> &'static ModelRegistry {
    GLOBAL_REGISTRY.get_or_init(ModelRegistry::new)
}

/// Register a model in the global registry
pub fn register_model<M: Model>(name: String, model: M) -> Result<(), RegistryError> {
    global_registry().register(name, model)
}

/// Get a model from the global registry
pub fn get_model(name: &str) -> Result<Arc<dyn Model>, RegistryError> {
    global_registry().get(name)
}

/// List all models in the global registry
pub fn list_models() -> Result<Vec<String>, RegistryError> {
    global_registry().list()
}

/// Check if a model exists in the global registry
pub fn model_exists(name: &str) -> Result<bool, RegistryError> {
    global_registry().contains(name)
}
