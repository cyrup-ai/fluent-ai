//! Template rendering engines
//!
//! Provides different template rendering implementations.

use std::sync::Arc;

use crate::types::candle_chat::templates::core::{
    ChatTemplate, TemplateContext, TemplateError, TemplateResult, TemplateValue};

/// Template rendering engine trait
pub trait TemplateEngine: Send + Sync {
    /// Render a template with the given context
    fn render(
        &self,
        template: &ChatTemplate,
        context: &TemplateContext,
    ) -> TemplateResult<Arc<str>>;

    /// Check if the engine supports the template format
    fn supports(&self, template: &ChatTemplate) -> bool;

    /// Get engine name
    fn name(&self) -> &'static str;
}

/// Simple string interpolation engine
pub struct SimpleEngine;

impl TemplateEngine for SimpleEngine {
    fn render(
        &self,
        template: &ChatTemplate,
        context: &TemplateContext,
    ) -> TemplateResult<Arc<str>> {
        let mut result = template.get_content().to_string();

        // Simple variable replacement: {{variable_name}}
        for (name, value) in context.variables() {
            let placeholder = format!("{{{{{}}}}}", name);
            let replacement = match value {
                TemplateValue::String(s) => s.as_ref(),
                TemplateValue::Number(n) => &n.to_string(),
                TemplateValue::Boolean(b) => {
                    if *b {
                        "true"
                    } else {
                        "false"
                    }
                }
                TemplateValue::Array(_) => "[array]", // Simplified
                TemplateValue::Object(_) => "[object]", // Simplified
                TemplateValue::Null => ""};
            result = result.replace(&placeholder, replacement);
        }

        Ok(Arc::from(result))
    }

    fn supports(&self, _template: &ChatTemplate) -> bool {
        true // Simple engine supports all templates
    }

    fn name(&self) -> &'static str {
        "simple"
    }
}

/// Template engine registry
pub struct EngineRegistry {
    engines: Vec<Box<dyn TemplateEngine>>}

impl EngineRegistry {
    /// Create a new engine registry
    pub fn new() -> Self {
        Self {
            engines: Vec::new()}
    }

    /// Create a registry with default engines
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(SimpleEngine));
        registry
    }

    /// Register a new engine
    pub fn register(&mut self, engine: Box<dyn TemplateEngine>) {
        self.engines.push(engine);
    }

    /// Find the best engine for a template
    pub fn find_engine(&self, template: &ChatTemplate) -> Option<&dyn TemplateEngine> {
        self.engines
            .iter()
            .find(|engine| engine.supports(template))
            .map(|engine| engine.as_ref())
    }

    /// Render a template using the best available engine
    pub fn render(
        &self,
        template: &ChatTemplate,
        context: &TemplateContext,
    ) -> TemplateResult<Arc<str>> {
        match self.find_engine(template) {
            Some(engine) => engine.render(template, context),
            None => Err(TemplateError::RenderError {
                message: Arc::from("No suitable rendering engine found")})}
    }
}

impl Default for EngineRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}
