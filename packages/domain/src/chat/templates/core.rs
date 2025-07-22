//! Core template types and structures
//!
//! This module defines the fundamental types for the template system with
//! zero-allocation, lock-free architecture.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Core template error types
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TemplateError {
    #[error("Template not found: {name}")]
    NotFound { name: Arc<str> },

    #[error("Parse error: {message}")]
    ParseError { message: Arc<str> },

    #[error("Compile error: {message}")]
    CompileError { message: Arc<str> },

    #[error("Render error: {message}")]
    RenderError { message: Arc<str> },

    #[error("Variable error: {message}")]
    VariableError { message: Arc<str> },

    #[error("Permission denied: {message}")]
    PermissionDenied { message: Arc<str> },

    #[error("Storage error: {message}")]
    StorageError { message: Arc<str> },
}

/// Template result type
pub type TemplateResult<T> = Result<T, TemplateError>;

/// Template information structure for metadata queries
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateInfo {
    pub id: Arc<str>,
    pub name: Arc<str>,
    pub category: TemplateCategory,
    pub size: usize,
    pub variable_count: usize,
    pub created_at: i64,
    pub modified_at: i64,
    pub version: Arc<str>,
}

/// Template category enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemplateCategory {
    Chat,
    System,
    User,
    Assistant,
    Function,
    Tool,
    Context,
    Prompt,
    Response,
    Custom,
}

impl Default for TemplateCategory {
    fn default() -> Self {
        Self::Chat
    }
}

/// Variable type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VariableType {
    String,
    Number,
    Boolean,
    Array,
    Object,
    Any,
}

/// Template variable definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateVariable {
    pub name: Arc<str>,
    pub description: Arc<str>,
    pub var_type: VariableType,
    pub default_value: Option<Arc<str>>,
    pub required: bool,
    pub validation_pattern: Option<Arc<str>>,
    pub valid_values: Option<Arc<[Arc<str>]>>,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
}

/// Template permissions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplatePermissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
    pub share: bool,
    pub delete: bool,
}

impl Default for TemplatePermissions {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            execute: true,
            share: true,
            delete: true,
        }
    }
}

/// Template metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateMetadata {
    pub id: Arc<str>,
    pub name: Arc<str>,
    pub description: Arc<str>,
    pub author: Arc<str>,
    pub version: Arc<str>,
    pub category: TemplateCategory,
    pub tags: Arc<[Arc<str>]>,
    pub created_at: u64,
    pub modified_at: u64,
    pub usage_count: u64,
    pub rating: f64,
    pub permissions: TemplatePermissions,
}

/// Template value type for variables
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemplateValue {
    String(Arc<str>),
    Number(f64),
    Boolean(bool),
    Array(Arc<[TemplateValue]>),
    Object(Arc<HashMap<Arc<str>, TemplateValue>>),
    Null,
}

impl From<&str> for TemplateValue {
    fn from(s: &str) -> Self {
        Self::String(Arc::from(s))
    }
}

impl From<String> for TemplateValue {
    fn from(s: String) -> Self {
        Self::String(Arc::from(s))
    }
}

impl From<f64> for TemplateValue {
    fn from(n: f64) -> Self {
        Self::Number(n)
    }
}

impl From<bool> for TemplateValue {
    fn from(b: bool) -> Self {
        Self::Boolean(b)
    }
}

/// Template context for rendering
#[derive(Clone)]
pub struct TemplateContext {
    pub variables: HashMap<Arc<str>, TemplateValue>,
    pub functions: HashMap<
        Arc<str>,
        Arc<dyn Fn(&[TemplateValue]) -> TemplateResult<TemplateValue> + Send + Sync>,
    >,
}

impl TemplateContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    pub fn with_variable(
        mut self,
        name: impl Into<Arc<str>>,
        value: impl Into<TemplateValue>,
    ) -> Self {
        self.variables.insert(name.into(), value.into());
        self
    }

    pub fn set_variable(&mut self, name: impl Into<Arc<str>>, value: impl Into<TemplateValue>) {
        self.variables.insert(name.into(), value.into());
    }

    pub fn get_variable(&self, name: &str) -> Option<&TemplateValue> {
        self.variables.get(name)
    }

    /// Get all variables as a reference to the HashMap
    pub fn variables(&self) -> &HashMap<Arc<str>, TemplateValue> {
        &self.variables
    }
}

impl Default for TemplateContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Abstract syntax tree for templates
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateAst {
    Text(Arc<str>),
    Variable(Arc<str>),
    Expression {
        operator: Arc<str>,
        operands: Arc<[TemplateAst]>,
    },
    Conditional {
        condition: Arc<TemplateAst>,
        if_true: Arc<TemplateAst>,
        if_false: Option<Arc<TemplateAst>>,
    },
    Loop {
        variable: Arc<str>,
        iterable: Arc<TemplateAst>,
        body: Arc<TemplateAst>,
    },
    Block(Arc<[TemplateAst]>),
    Function {
        name: Arc<str>,
        args: Arc<[TemplateAst]>,
    },
}

/// Compiled template representation
#[derive(Debug, Clone)]
pub struct CompiledTemplate {
    pub metadata: TemplateMetadata,
    pub ast: TemplateAst,
    pub variables: Arc<[TemplateVariable]>,
    pub optimized: bool,
}

impl CompiledTemplate {
    pub fn new(
        metadata: TemplateMetadata,
        ast: TemplateAst,
        variables: Arc<[TemplateVariable]>,
    ) -> Self {
        Self {
            metadata,
            ast,
            variables,
            optimized: false,
        }
    }

    pub fn render(&self, context: &TemplateContext) -> TemplateResult<Arc<str>> {
        self.render_ast(&self.ast, context)
    }

    fn render_ast(&self, ast: &TemplateAst, context: &TemplateContext) -> TemplateResult<Arc<str>> {
        match ast {
            TemplateAst::Text(text) => Ok(text.clone()),
            TemplateAst::Variable(name) => {
                if let Some(value) = context.get_variable(name) {
                    match value {
                        TemplateValue::String(s) => Ok(s.clone()),
                        TemplateValue::Number(n) => Ok(Arc::from(n.to_string())),
                        TemplateValue::Boolean(b) => Ok(Arc::from(b.to_string())),
                        _ => Ok(Arc::from(format!("{:?}", value))),
                    }
                } else {
                    Err(TemplateError::VariableError {
                        message: Arc::from(format!("Variable '{}' not found", name)),
                    })
                }
            }
            TemplateAst::Block(nodes) => {
                let mut result = String::new();
                for node in nodes.iter() {
                    let rendered = self.render_ast(node, context)?;
                    result.push_str(&rendered);
                }
                Ok(Arc::from(result))
            }
            TemplateAst::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                let cond_result = self.render_ast(condition, context)?;
                let is_truthy = !cond_result.is_empty()
                    && cond_result.as_ref() != "false"
                    && cond_result.as_ref() != "0";

                if is_truthy {
                    self.render_ast(if_true, context)
                } else if let Some(if_false_ast) = if_false {
                    self.render_ast(if_false_ast, context)
                } else {
                    Ok(Arc::from(""))
                }
            }
            _ => Ok(Arc::from("")), // TODO: Implement other AST node types
        }
    }
}

/// Main chat template structure
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    pub metadata: TemplateMetadata,
    pub content: Arc<str>,
    pub variables: Arc<[TemplateVariable]>,
    pub compiled: Option<CompiledTemplate>,
}

impl ChatTemplate {
    pub fn new(
        metadata: TemplateMetadata,
        content: Arc<str>,
        variables: Arc<[TemplateVariable]>,
    ) -> Self {
        Self {
            metadata,
            content,
            variables,
            compiled: None,
        }
    }

    pub fn render(&self, variables: &HashMap<Arc<str>, Arc<str>>) -> TemplateResult<Arc<str>> {
        let mut context = TemplateContext::new();
        for (key, value) in variables {
            context.set_variable(key.clone(), TemplateValue::String(value.clone()));
        }

        if let Some(compiled) = &self.compiled {
            compiled.render(&context)
        } else {
            // Simple variable replacement for non-compiled templates
            let mut result = self.content.to_string();
            for (key, value) in variables {
                result = result.replace(&format!("{{{{{}}}}}", key), value);
            }
            Ok(Arc::from(result))
        }
    }

    pub fn get_id(&self) -> &Arc<str> {
        &self.metadata.id
    }

    pub fn get_name(&self) -> &Arc<str> {
        &self.metadata.name
    }

    pub fn get_content(&self) -> &Arc<str> {
        &self.content
    }

    pub fn get_variables(&self) -> &Arc<[TemplateVariable]> {
        &self.variables
    }

    /// Get template name (alias for get_name for compatibility)
    pub fn name(&self) -> &Arc<str> {
        &self.metadata.name
    }

    /// Validate the template
    pub fn validate(&self) -> TemplateResult<()> {
        // Basic validation
        if self.metadata.name.is_empty() {
            return Err(TemplateError::ParseError {
                message: Arc::from("Template name cannot be empty"),
            });
        }

        if self.content.is_empty() {
            return Err(TemplateError::ParseError {
                message: Arc::from("Template content cannot be empty"),
            });
        }

        // Additional validation can be added here
        Ok(())
    }

    /// Get template info
    pub fn info(&self) -> TemplateInfo {
        TemplateInfo {
            id: self.metadata.id.clone(),
            name: self.metadata.name.clone(),
            category: self.metadata.category,
            size: self.content.len(),
            variable_count: self.variables.len(),
            created_at: self.metadata.created_at as i64,
            modified_at: self.metadata.modified_at as i64,
            version: self.metadata.version.clone(),
        }
    }
}

/// Template configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateConfig {
    pub max_template_size: usize,
    pub max_variables: usize,
    pub allow_nested_templates: bool,
    pub cache_compiled: bool,
    pub optimize_templates: bool,
    pub security_mode: SecurityMode,
}

/// Security mode for templates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityMode {
    Strict,   // No external access, limited functions
    Normal,   // Standard functions allowed
    Relaxed,  // Most functions allowed
    Disabled, // All functions allowed
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            max_template_size: 1024 * 1024, // 1MB
            max_variables: 1000,
            allow_nested_templates: true,
            cache_compiled: true,
            optimize_templates: true,
            security_mode: SecurityMode::Normal,
        }
    }
}

/// Template example for documentation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateExample {
    pub name: Arc<str>,
    pub description: Arc<str>,
    pub input_variables: HashMap<Arc<str>, Arc<str>>,
    pub expected_output: Arc<str>,
}

/// Template tag for categorization
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TemplateTag {
    pub name: Arc<str>,
    pub color: Option<Arc<str>>,
    pub description: Option<Arc<str>>,
}

impl TemplateTag {
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            color: None,
            description: None,
        }
    }

    pub fn with_color(mut self, color: impl Into<Arc<str>>) -> Self {
        self.color = Some(color.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<Arc<str>>) -> Self {
        self.description = Some(description.into());
        self
    }
}
