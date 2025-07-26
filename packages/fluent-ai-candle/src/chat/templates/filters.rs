//! Template filters for data transformation
//!
//! Provides built-in filters for template processing.

use std::sync::Arc;

use crate::chat::templates::core::{TemplateError, TemplateResult, TemplateValue};

/// Template filter function type
pub type FilterFunction =
    Arc<dyn Fn(&TemplateValue, &[TemplateValue]) -> TemplateResult<TemplateValue> + Send + Sync>;

/// Filter registry for managing template filters
pub struct FilterRegistry {
    filters: std::collections::HashMap<Arc<str>, FilterFunction>}

impl FilterRegistry {
    /// Create a new filter registry
    pub fn new() -> Self {
        Self {
            filters: std::collections::HashMap::new()}
    }

    /// Create a registry with default filters
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register_default_filters();
        registry
    }

    /// Register a filter
    pub fn register(&mut self, name: impl Into<Arc<str>>, filter: FilterFunction) {
        self.filters.insert(name.into(), filter);
    }

    /// Get a filter by name
    pub fn get(&self, name: &str) -> Option<&FilterFunction> {
        self.filters.get(name)
    }

    /// Apply a filter to a value
    pub fn apply(
        &self,
        name: &str,
        value: &TemplateValue,
        args: &[TemplateValue],
    ) -> TemplateResult<TemplateValue> {
        match self.get(name) {
            Some(filter) => filter(value, args),
            None => Err(TemplateError::RenderError {
                message: Arc::from(format!("Unknown filter: {}", name))})}
    }

    /// Register default filters
    fn register_default_filters(&mut self) {
        // uppercase filter
        self.register(
            "uppercase",
            Arc::new(|value, _args| match value {
                TemplateValue::String(s) => Ok(TemplateValue::String(Arc::from(s.to_uppercase()))),
                _ => Err(TemplateError::RenderError {
                    message: Arc::from("uppercase filter can only be applied to strings")})}),
        );

        // lowercase filter
        self.register(
            "lowercase",
            Arc::new(|value, _args| match value {
                TemplateValue::String(s) => Ok(TemplateValue::String(Arc::from(s.to_lowercase()))),
                _ => Err(TemplateError::RenderError {
                    message: Arc::from("lowercase filter can only be applied to strings")})}),
        );

        // trim filter
        self.register(
            "trim",
            Arc::new(|value, _args| match value {
                TemplateValue::String(s) => Ok(TemplateValue::String(Arc::from(s.trim()))),
                _ => Err(TemplateError::RenderError {
                    message: Arc::from("trim filter can only be applied to strings")})}),
        );

        // length filter
        self.register(
            "length",
            Arc::new(|value, _args| match value {
                TemplateValue::String(s) => Ok(TemplateValue::Number(s.len() as f64)),
                TemplateValue::Array(arr) => Ok(TemplateValue::Number(arr.len() as f64)),
                _ => Err(TemplateError::RenderError {
                    message: Arc::from("length filter can only be applied to strings or arrays")})}),
        );

        // default filter
        self.register(
            "default",
            Arc::new(|value, args| {
                let is_empty = match value {
                    TemplateValue::String(s) => s.is_empty(),
                    TemplateValue::Null => true,
                    _ => false};

                if is_empty && !args.is_empty() {
                    Ok(args[0].clone())
                } else {
                    Ok(value.clone())
                }
            }),
        );
    }
}

impl Default for FilterRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}
