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
    /// Template with specified name was not found
    #[error("Template not found: {name}")]
    NotFound { 
        /// Name of the template that was not found
        name: Arc<str> 
    },

    /// Template parsing failed
    #[error("Parse error: {message}")]
    ParseError { 
        /// Detailed error message
        message: Arc<str> 
    },

    /// Template compilation failed
    #[error("Compile error: {message}")]
    CompileError { 
        /// Detailed error message
        message: Arc<str> 
    },

    /// Template rendering failed
    #[error("Render error: {message}")]
    RenderError { 
        /// Detailed error message
        message: Arc<str> 
    },

    /// Variable processing error
    #[error("Variable error: {message}")]
    VariableError { 
        /// Detailed error message
        message: Arc<str> 
    },

    /// Access permission denied
    #[error("Permission denied: {message}")]
    PermissionDenied { 
        /// Detailed error message
        message: Arc<str> 
    },

    /// Storage operation failed
    #[error("Storage error: {message}")]
    StorageError { 
        /// Detailed error message
        message: Arc<str> 
    }
}

/// Template result type
pub type TemplateResult<T> = Result<T, TemplateError>;

/// Template information structure for metadata queries
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateInfo {
    /// Unique identifier for the template
    pub id: Arc<str>,
    /// Human-readable name of the template
    pub name: Arc<str>,
    /// Category classification for the template
    pub category: TemplateCategory,
    /// Size of the template in bytes
    pub size: usize,
    /// Number of variables in the template
    pub variable_count: usize,
    /// Creation timestamp as Unix timestamp
    pub created_at: i64,
    /// Last modification timestamp as Unix timestamp
    pub modified_at: i64,
    /// Version string for the template
    pub version: Arc<str>
}

/// Template category enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemplateCategory {
    /// General chat templates
    Chat,
    /// System-level templates
    System,
    /// User-specific templates
    User,
    /// Assistant response templates
    Assistant,
    /// Function call templates
    Function,
    /// Tool usage templates
    Tool,
    /// Context injection templates
    Context,
    /// Prompt templates
    Prompt,
    /// Response formatting templates
    Response,
    /// Custom user-defined templates
    Custom
}

impl Default for TemplateCategory {
    fn default() -> Self {
        Self::Chat
    }
}

/// Variable type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VariableType {
    /// String variable type
    String,
    /// Numeric variable type
    Number,
    /// Boolean variable type
    Boolean,
    /// Array variable type
    Array,
    /// Object variable type
    Object,
    /// Any variable type (unrestricted)
    Any
}

/// Template variable definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateVariable {
    /// Variable name
    pub name: Arc<str>,
    /// Variable description
    pub description: Arc<str>,
    /// Variable data type
    pub var_type: VariableType,
    /// Default value for the variable
    pub default_value: Option<Arc<str>>,
    /// Whether the variable is required
    pub required: bool,
    /// Validation pattern for the variable value
    pub validation_pattern: Option<Arc<str>>,
    /// List of valid values for enumerated variables
    pub valid_values: Option<Arc<[Arc<str>]>>,
    /// Minimum value for numeric variables
    pub min_value: Option<f64>,
    /// Maximum value for numeric variables
    pub max_value: Option<f64>
}

/// Template permissions
/// Permissions for template access control
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplatePermissions {
    /// Whether the template can be read
    pub read: bool,
    /// Whether the template can be modified
    pub write: bool,
    /// Whether the template can be executed
    pub execute: bool,
    /// Whether the template can be shared with others
    pub share: bool,
    /// Whether the template can be deleted
    pub delete: bool}

impl Default for TemplatePermissions {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            execute: true,
            share: true,
            delete: true}
    }
}

/// Template metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Unique identifier for the template
    pub id: Arc<str>,
    /// Human-readable name of the template
    pub name: Arc<str>,
    /// Detailed description of template functionality
    pub description: Arc<str>,
    /// Author or creator of the template
    pub author: Arc<str>,
    /// Version string of the template
    pub version: Arc<str>,
    /// Category classification for the template
    pub category: TemplateCategory,
    /// Tags for template discovery and organization
    pub tags: Arc<[Arc<str>]>,
    /// Unix timestamp when template was created
    pub created_at: u64,
    /// Unix timestamp when template was last modified
    pub modified_at: u64,
    /// Number of times template has been used
    pub usage_count: u64,
    /// Average user rating (0.0 to 5.0)
    pub rating: f64,
    /// Access permissions for the template
    pub permissions: TemplatePermissions}

/// Template value type for variables
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemplateValue {
    /// String value
    String(Arc<str>),
    /// Numeric value (floating point)
    Number(f64),
    /// Boolean true/false value
    Boolean(bool),
    /// Array of template values
    Array(Arc<[TemplateValue]>),
    /// Object/map of key-value pairs
    Object(Arc<HashMap<Arc<str>, TemplateValue>>),
    /// Null/empty value
    Null}

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
    /// Variables available during template rendering
    pub variables: HashMap<Arc<str>, TemplateValue>,
    /// Functions available during template rendering
    pub functions: HashMap<
        Arc<str>,
        Arc<dyn Fn(&[TemplateValue]) -> TemplateResult<TemplateValue> + Send + Sync>,
    >}

impl TemplateContext {
    /// Create a new empty template context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new()}
    }

    /// Add a variable to the context (builder pattern)
    pub fn with_variable(
        mut self,
        name: impl Into<Arc<str>>,
        value: impl Into<TemplateValue>,
    ) -> Self {
        self.variables.insert(name.into(), value.into());
        self
    }

    /// Set a variable in the context (mutating)
    pub fn set_variable(&mut self, name: impl Into<Arc<str>>, value: impl Into<TemplateValue>) {
        self.variables.insert(name.into(), value.into());
    }

    /// Get a variable value by name
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
    /// Static text content
    Text(Arc<str>),
    /// Variable reference
    Variable(Arc<str>),
    /// Expression with operator and operands
    Expression {
        /// The operator for this expression
        operator: Arc<str>,
        /// The operands for this expression
        operands: Arc<[TemplateAst]>,
    },
    /// Conditional (if/else) statement
    Conditional {
        /// The condition to evaluate
        condition: Arc<TemplateAst>,
        /// AST to execute if condition is true
        if_true: Arc<TemplateAst>,
        /// Optional AST to execute if condition is false
        if_false: Option<Arc<TemplateAst>>,
    },
    /// Loop statement
    Loop {
        /// Loop variable name
        variable: Arc<str>,
        /// The iterable expression to loop over
        iterable: Arc<TemplateAst>,
        /// The body of the loop
        body: Arc<TemplateAst>,
    },
    /// Block of multiple AST nodes
    Block(Arc<[TemplateAst]>),
    /// Function call
    Function {
        /// Function name
        name: Arc<str>,
        /// Function arguments
        args: Arc<[TemplateAst]>,
    },
}

/// Compiled template representation
#[derive(Debug, Clone)]
pub struct CompiledTemplate {
    /// Template metadata and information
    pub metadata: TemplateMetadata,
    /// Compiled abstract syntax tree
    pub ast: TemplateAst,
    /// Template variables and their definitions
    pub variables: Arc<[TemplateVariable]>,
    /// Whether the template has been optimized for performance
    pub optimized: bool
}

impl CompiledTemplate {
    /// Create a new compiled template
    pub fn new(
        metadata: TemplateMetadata,
        ast: TemplateAst,
        variables: Arc<[TemplateVariable]>,
    ) -> Self {
        Self {
            metadata,
            ast,
            variables,
            optimized: false}
    }

    /// Render the template with the given context
    pub fn render(&self, context: &TemplateContext) -> TemplateResult<Arc<str>> {
        self.render_ast(&self.ast, context)
    }

    /// Internal method to render a specific AST node with the given context
    fn render_ast(&self, ast: &TemplateAst, context: &TemplateContext) -> TemplateResult<Arc<str>> {
        match ast {
            TemplateAst::Text(text) => Ok(text.clone()),
            TemplateAst::Variable(name) => {
                if let Some(value) = context.get_variable(name) {
                    match value {
                        TemplateValue::String(s) => Ok(s.clone()),
                        TemplateValue::Number(n) => Ok(Arc::from(n.to_string())),
                        TemplateValue::Boolean(b) => Ok(Arc::from(b.to_string())),
                        _ => Ok(Arc::from(format!("{:?}", value)))}
                } else {
                    Err(TemplateError::VariableError {
                        message: Arc::from(format!("Variable '{}' not found", name))})
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
                if_false} => {
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
    /// Template metadata and information
    pub metadata: TemplateMetadata,
    /// Raw template content string
    pub content: Arc<str>,
    /// Template variables and their definitions
    pub variables: Arc<[TemplateVariable]>,
    /// Optional compiled template for faster rendering
    pub compiled: Option<CompiledTemplate>
}

impl ChatTemplate {
    /// Create a new chat template
    pub fn new(
        metadata: TemplateMetadata,
        content: Arc<str>,
        variables: Arc<[TemplateVariable]>,
    ) -> Self {
        Self {
            metadata,
            content,
            variables,
            compiled: None}
    }

    /// Render the template with provided variables
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

    /// Get the template ID
    pub fn get_id(&self) -> &Arc<str> {
        &self.metadata.id
    }

    /// Get the template name
    pub fn get_name(&self) -> &Arc<str> {
        &self.metadata.name
    }

    /// Get the template content
    pub fn get_content(&self) -> &Arc<str> {
        &self.content
    }

    /// Get the template variables
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
                message: Arc::from("Template name cannot be empty")});
        }

        if self.content.is_empty() {
            return Err(TemplateError::ParseError {
                message: Arc::from("Template content cannot be empty")});
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
            version: self.metadata.version.clone()}
    }
}

/// Template configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateConfig {
    /// Maximum allowed template size in bytes
    pub max_template_size: usize,
    /// Maximum number of variables allowed per template
    pub max_variables: usize,
    /// Whether to allow nested template includes
    pub allow_nested_templates: bool,
    /// Whether to cache compiled templates for performance
    pub cache_compiled: bool,
    /// Whether to optimize templates during compilation
    pub optimize_templates: bool,
    /// Security mode for template execution
    pub security_mode: SecurityMode
}

/// Security mode for templates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityMode {
    /// Strict mode - no external access, limited functions
    Strict,
    /// Normal mode - standard functions allowed
    Normal,
    /// Relaxed mode - most functions allowed
    Relaxed,
    /// Disabled mode - all functions allowed (no restrictions)
    Disabled,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            max_template_size: 1024 * 1024, // 1MB
            max_variables: 1000,
            allow_nested_templates: true,
            cache_compiled: true,
            optimize_templates: true,
            security_mode: SecurityMode::Normal}
    }
}

/// Template example for documentation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateExample {
    /// Name of the example
    pub name: Arc<str>,
    /// Description of what the example demonstrates
    pub description: Arc<str>,
    /// Example input variables for the template
    pub input_variables: HashMap<Arc<str>, Arc<str>>,
    /// Expected output when template is rendered with input variables
    pub expected_output: Arc<str>
}

/// Template tag for categorization
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TemplateTag {
    /// Tag name
    pub name: Arc<str>,
    /// Optional color for the tag (hex code or color name)
    pub color: Option<Arc<str>>,
    /// Optional description of the tag
    pub description: Option<Arc<str>>
}

impl TemplateTag {
    /// Create a new template tag
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            color: None,
            description: None}
    }

    /// Set the tag color (builder pattern)
    pub fn with_color(mut self, color: impl Into<Arc<str>>) -> Self {
        self.color = Some(color.into());
        self
    }

    /// Set the tag description (builder pattern)
    pub fn with_description(mut self, description: impl Into<Arc<str>>) -> Self {
        self.description = Some(description.into());
        self
    }
}
