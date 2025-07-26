//! Template manager for storing and retrieving templates
//!
//! Provides high-performance template storage with lock-free operations.

use std::sync::Arc;

use crossbeam_skiplist::SkipMap;

use crate::domain::chat::templates::core::{ChatTemplate as CandleChatTemplate, TemplateError as CandleTemplateError, TemplateInfo as CandleTemplateInfo, TemplateResult as CandleTemplateResult};

/// Template manager for storing and managing templates
#[derive(Debug)]
pub struct TemplateManager {
    templates: SkipMap<Arc<str>, CandleChatTemplate>}

impl TemplateManager {
    /// Create a new template manager
    pub fn new() -> Self {
        Self {
            templates: SkipMap::new()}
    }

    /// Store a template
    pub fn store(&self, template: CandleChatTemplate) -> CandleTemplateResult<()> {
        template.validate()?;
        let name = template.name().clone();
        self.templates.insert(name, template);
        Ok(())
    }

    /// Get a template by name
    pub fn get(&self, name: &str) -> CandleTemplateResult<CandleChatTemplate> {
        self.templates
            .get(name)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| CandleTemplateError::NotFound {
                name: Arc::from(name)})
    }

    /// Delete a template
    pub fn delete(&self, name: &str) -> CandleTemplateResult<()> {
        self.templates
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| CandleTemplateError::NotFound {
                name: Arc::from(name)})
    }

    /// List all template names
    pub fn list_names(&self) -> Vec<Arc<str>> {
        self.templates
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get template info
    pub fn get_info(&self, name: &str) -> CandleTemplateResult<CandleTemplateInfo> {
        let template = self.get(name)?;
        Ok(template.info())
    }

    /// Check if template exists
    pub fn exists(&self, name: &str) -> bool {
        self.templates.contains_key(name)
    }

    /// Get template count
    pub fn count(&self) -> usize {
        self.templates.len()
    }

    /// Clear all templates
    pub fn clear(&self) {
        self.templates.clear();
    }
}

impl Default for TemplateManager {
    fn default() -> Self {
        Self::new()
    }
}
