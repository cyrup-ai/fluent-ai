//! Chat template system - Modular, high-performance template management
//!
//! This module provides a comprehensive template system for chat applications with
//! zero-allocation, lock-free architecture optimized for high-throughput scenarios.

pub mod cache;
pub mod compiler;
pub mod core;
pub mod engines;
pub mod filters;
pub mod manager;
pub mod parser;

// Re-export core types for convenience
pub use core::{
    ChatTemplate, CompiledTemplate, TemplateAst, TemplateCategory, TemplateConfig, TemplateContext,
    TemplateError, TemplateExample, TemplateInfo, TemplateMetadata, TemplateResult, TemplateTag,
    TemplateValue};
// Global template functions for convenience
use std::collections::HashMap;
use std::sync::Arc;

pub use compiler::TemplateCompiler;
// Re-export other important types
pub use manager::TemplateManager;
pub use parser::TemplateParser;

/// Create a simple template
pub fn template(name: impl Into<String>, content: impl Into<String>) -> ChatTemplate {
    let template_name: Arc<str> = Arc::from(name.into());
    let template_content: Arc<str> = Arc::from(content.into());

    let metadata = core::TemplateMetadata {
        id: template_name.clone(),
        name: template_name,
        description: Arc::from(""),
        author: Arc::from(""),
        version: Arc::from("1.0.0"),
        category: core::TemplateCategory::Chat,
        tags: Arc::new([]),
        created_at: 0,
        modified_at: 0,
        usage_count: 0,
        rating: 0.0,
        permissions: core::TemplatePermissions::default()};

    ChatTemplate::new(metadata, template_content, Arc::new([]))
}

/// Get global template manager
pub fn get_template_manager() -> &'static TemplateManager {
    use std::sync::OnceLock;
    static MANAGER: OnceLock<TemplateManager> = OnceLock::new();
    MANAGER.get_or_init(|| TemplateManager::new())
}

/// Store a template in global manager
pub fn store_template(template: ChatTemplate) -> TemplateResult<()> {
    let manager = get_template_manager();
    manager.store(template)
}

/// Get a template from global manager
pub fn get_template(name: &str) -> Option<ChatTemplate> {
    let manager = get_template_manager();
    manager.get(name).ok()
}

/// Render a template with variables
pub fn render_template(
    name: &str,
    variables: HashMap<Arc<str>, Arc<str>>,
) -> TemplateResult<Arc<str>> {
    if let Some(template) = get_template(name) {
        template.render(&variables)
    } else {
        Err(TemplateError::NotFound {
            name: Arc::from(name)})
    }
}

/// Simple render function with string variables
pub fn render_simple(name: &str, variables: HashMap<&str, &str>) -> TemplateResult<Arc<str>> {
    let arc_variables: HashMap<Arc<str>, Arc<str>> = variables
        .into_iter()
        .map(|(k, v)| (Arc::from(k), Arc::from(v)))
        .collect();
    render_template(name, arc_variables)
}

/// Template builder for fluent API
pub fn builder() -> TemplateBuilder {
    TemplateBuilder::new()
}

/// Template builder struct
#[derive(Debug, Clone)]
pub struct TemplateBuilder {
    name: Option<String>,
    content: Option<String>,
    description: Option<String>,
    category: TemplateCategory,
    variables: Vec<String>}

impl TemplateBuilder {
    pub fn new() -> Self {
        Self {
            name: None,
            content: None,
            description: None,
            category: TemplateCategory::Chat,
            variables: Vec::new()}
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn category(mut self, category: TemplateCategory) -> Self {
        self.category = category;
        self
    }

    pub fn variable(mut self, var: impl Into<String>) -> Self {
        self.variables.push(var.into());
        self
    }

    pub fn build(self) -> ChatTemplate {
        let name = self.name.unwrap_or_else(|| "untitled".to_string());
        let content = self.content.unwrap_or_else(|| "".to_string());
        let description = self.description.unwrap_or_else(|| "".to_string());

        let template_name: Arc<str> = Arc::from(name);
        let template_content: Arc<str> = Arc::from(content);

        let metadata = core::TemplateMetadata {
            id: template_name.clone(),
            name: template_name,
            description: Arc::from(description),
            author: Arc::from(""),
            version: Arc::from("1.0.0"),
            category: core::TemplateCategory::Chat,
            tags: Arc::new([]),
            created_at: 0,
            modified_at: 0,
            usage_count: 0,
            rating: 0.0,
            permissions: core::TemplatePermissions::default()};

        let variables: Arc<[core::TemplateVariable]> = self
            .variables
            .into_iter()
            .map(|v| core::TemplateVariable {
                name: Arc::from(v),
                description: Arc::from(""),
                var_type: core::VariableType::String,
                default_value: None,
                required: false,
                validation_pattern: None,
                valid_values: None,
                min_value: None,
                max_value: None})
            .collect::<Vec<_>>()
            .into();

        ChatTemplate::new(metadata, template_content, variables)
    }
}

/// Create a simple render function
pub fn render(
    template: &ChatTemplate,
    variables: &HashMap<Arc<str>, Arc<str>>,
) -> TemplateResult<Arc<str>> {
    template.render(variables)
}

/// Create template from string
pub fn create_template(name: &str, content: &str) -> ChatTemplate {
    template(name, content)
}
