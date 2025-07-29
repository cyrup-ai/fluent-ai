//! Template storage implementation
//!
//! Provides persistent storage for templates with caching support.

use std::collections::HashMap;
use std::sync::Arc;

use crate::domain::chat::templates::core::{ChatTemplate, TemplateError, TemplateResult};

/// Template storage interface
pub trait TemplateStore: Send + Sync {
    /// Store a template
    fn store(&self, template: &ChatTemplate) -> TemplateResult<()>;

    /// Retrieve a template by name
    fn get(&self, name: &str) -> TemplateResult<Option<ChatTemplate>>;

    /// Delete a template
    fn delete(&self, name: &str) -> TemplateResult<bool>;

    /// List all template names
    fn list(&self) -> TemplateResult<Vec<Arc<str>>>;

    /// Check if template exists
    fn exists(&self, name: &str) -> TemplateResult<bool>;
}

/// In-memory template store implementation
pub struct MemoryStore {
    templates: std::sync::RwLock<HashMap<Arc<str>, ChatTemplate>>}

impl MemoryStore {
    /// Create a new memory store
    pub fn new() -> Self {
        Self {
            templates: std::sync::RwLock::new(HashMap::new())}
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateStore for MemoryStore {
    fn store(&self, template: &ChatTemplate) -> TemplateResult<()> {
        let mut store = self
            .templates
            .write()
            .map_err(|_| TemplateError::StorageError {
                message: Arc::from("Failed to acquire write lock")})?;

        store.insert(template.name().clone(), template.clone());
        Ok(())
    }

    fn get(&self, name: &str) -> TemplateResult<Option<ChatTemplate>> {
        let store = self
            .templates
            .read()
            .map_err(|_| TemplateError::StorageError {
                message: Arc::from("Failed to acquire read lock")})?;

        Ok(store.get(name).cloned())
    }

    fn delete(&self, name: &str) -> TemplateResult<bool> {
        let mut store = self
            .templates
            .write()
            .map_err(|_| TemplateError::StorageError {
                message: Arc::from("Failed to acquire write lock")})?;

        Ok(store.remove(name).is_some())
    }

    fn list(&self) -> TemplateResult<Vec<Arc<str>>> {
        let store = self
            .templates
            .read()
            .map_err(|_| TemplateError::StorageError {
                message: Arc::from("Failed to acquire read lock")})?;

        Ok(store.keys().cloned().collect())
    }

    fn exists(&self, name: &str) -> TemplateResult<bool> {
        let store = self
            .templates
            .read()
            .map_err(|_| TemplateError::StorageError {
                message: Arc::from("Failed to acquire read lock")})?;

        Ok(store.contains_key(name))
    }
}
