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

    /// Template parsing failed with syntax or semantic error
    #[error("Parse error: {message}")]
    ParseError { 
        /// Detailed error message describing the parse failure
        message: Arc<str> 
    },

    /// Template compilation failed during processing
    #[error("Compile error: {message}")]
    CompileError { 
        /// Detailed error message describing the compilation failure
        message: Arc<str> 
    },

    /// Template rendering failed during execution
    #[error("Render error: {message}")]
    RenderError { 
        /// Detailed error message describing the rendering failure
        message: Arc<str> 
    },

    /// Variable access or manipulation error
    #[error("Variable error: {message}")]
    VariableError { 
        /// Detailed error message describing the variable operation failure
        message: Arc<str> 
    },

    /// Access denied due to insufficient permissions
    #[error("Permission denied: {message}")]
    PermissionDenied { 
        /// Detailed error message describing the permission denial
        message: Arc<str> 
    },

    /// Template storage operation failed
    #[error("Storage error: {message}")]
    StorageError { 
        /// Detailed error message describing the storage failure
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
    /// Category classification for organization
    pub category: TemplateCategory,
    /// Size of the template in bytes
    pub size: usize,
    /// Number of variables defined in the template
    pub variable_count: usize,
    /// Unix timestamp when template was created
    pub created_at: i64,
    /// Unix timestamp when template was last modified
    pub modified_at: i64,
    /// Version string for template versioning
    pub version: Arc<str>
}

/// Template category enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemplateCategory {
    /// Chat conversation templates
    Chat,
    /// System-level templates for internal operations
    System,
    /// User input and interaction templates
    User,
    /// AI assistant response templates
    Assistant,
    /// Function call and result templates
    Function,
    /// Tool integration templates
    Tool,
    /// Context and background information templates
    Context,
    /// Prompt construction templates
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
    /// String/text value type
    String,
    /// Numeric value type (integer or float)
    Number,
    /// Boolean true/false value type
    Boolean,
    /// Array/list of values type
    Array,
    /// Object/map with key-value pairs type
    Object,
    /// Any type - no type restrictions
    Any
}

/// Template variable definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateVariable {
    /// Variable name identifier
    pub name: Arc<str>,
    /// Human-readable description of the variable
    pub description: Arc<str>,
    /// Type specification for the variable
    pub var_type: VariableType,
    /// Default value if not provided (as string representation)
    pub default_value: Option<Arc<str>>,
    /// Whether this variable must be provided
    pub required: bool,
    /// Regex pattern for validation (if applicable)
    pub validation_pattern: Option<Arc<str>>,
    /// List of valid values for enumeration types
    pub valid_values: Option<Arc<[Arc<str>]>>,
    /// Minimum numeric value (for Number type)
    pub min_value: Option<f64>,
    /// Maximum numeric value (for Number type)
    pub max_value: Option<f64>
}

/// Template permissions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplatePermissions {
    /// Permission to read/view the template
    pub read: bool,
    /// Permission to modify the template
    pub write: bool,
    /// Permission to execute/render the template
    pub execute: bool,
    /// Permission to share the template with others
    pub share: bool,
    /// Permission to delete the template
    pub delete: bool
}

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
    /// Display name of the template
    pub name: Arc<str>,
    /// Detailed description of template purpose
    pub description: Arc<str>,
    /// Template author/creator name
    pub author: Arc<str>,
    /// Version string for template versioning
    pub version: Arc<str>,
    /// Category classification
    pub category: TemplateCategory,
    /// Searchable tags for organization
    pub tags: Arc<[Arc<str>]>,
    /// Unix timestamp when template was created
    pub created_at: u64,
    /// Unix timestamp when template was last modified
    pub modified_at: u64,
    /// Total number of times template has been used
    pub usage_count: u64,
    /// Average user rating (0.0-5.0)
    pub rating: f64,
    /// Access permissions for the template
    pub permissions: TemplatePermissions
}

/// Template value type for variables
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemplateValue {
    /// String value contained in Arc for zero-copy sharing
    String(Arc<str>),
    /// Numeric value (integers and floats)
    Number(f64),
    /// Boolean true/false value
    Boolean(bool),
    /// Array of template values
    Array(Arc<[TemplateValue]>),
    /// Object with string keys and template value values
    Object(Arc<HashMap<Arc<str>, TemplateValue>>),
    /// Null/empty value
    Null
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
    /// Variables available in the template context
    pub variables: HashMap<Arc<str>, TemplateValue>,
    /// Functions available for template evaluation
    pub functions: HashMap<
        Arc<str>,
        Arc<dyn Fn(&[TemplateValue]) -> TemplateResult<TemplateValue> + Send + Sync>,
    >
}

impl TemplateContext {
    /// Create a new empty template context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new()
        }
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

    /// Set a variable in the context
    pub fn set_variable(&mut self, name: impl Into<Arc<str>>, value: impl Into<TemplateValue>) {
        self.variables.insert(name.into(), value.into());
    }

    /// Get a variable from the context
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
    /// Plain text content
    Text(Arc<str>),
    /// Variable reference
    Variable(Arc<str>),
    /// Expression with operator and operands
    Expression {
        /// Operator symbol (e.g., "+", "-", "==")
        operator: Arc<str>,
        /// List of operand expressions
        operands: Arc<[TemplateAst]>
    },
    /// Conditional statement (if/else)
    Conditional {
        /// Condition expression to evaluate
        condition: Arc<TemplateAst>,
        /// Expression to execute if condition is true
        if_true: Arc<TemplateAst>,
        /// Optional expression to execute if condition is false
        if_false: Option<Arc<TemplateAst>>
    },
    /// Loop iteration construct
    Loop {
        /// Loop variable name
        variable: Arc<str>,
        /// Expression that provides the iterable collection
        iterable: Arc<TemplateAst>,
        /// Loop body expression
        body: Arc<TemplateAst>
    },
    /// Block of multiple expressions
    Block(Arc<[TemplateAst]>),
    /// Function call
    Function {
        /// Function name
        name: Arc<str>,
        /// Function arguments
        args: Arc<[TemplateAst]>
    }
}

/// Compiled template representation
#[derive(Debug, Clone)]
pub struct CompiledTemplate {
    /// Template metadata and information
    pub metadata: TemplateMetadata,
    /// Compiled abstract syntax tree
    pub ast: TemplateAst,
    /// List of variables defined in the template
    pub variables: Arc<[TemplateVariable]>,
    /// Whether the template has been optimized
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
            optimized: false
        }
    }

    /// Render the compiled template with the given context
    pub fn render(&self, context: &TemplateContext) -> TemplateResult<Arc<str>> {
        self.render_ast(&self.ast, context)
    }

    /// Recursively render an AST node with the given context
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
    /// List of variables defined in the template
    pub variables: Arc<[TemplateVariable]>,
    /// Optional compiled representation for performance
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
            compiled: None
        }
    }

    /// Render the template with the given variables
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

    /// Get template information for queries
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
    /// Maximum size of template content in bytes
    pub max_template_size: usize,
    /// Maximum number of variables per template
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
    /// No external access, limited functions
    Strict,
    /// Standard functions allowed
    Normal,
    /// Most functions allowed
    Relaxed,
    /// All functions allowed
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
    /// Input variables for the example
    pub input_variables: HashMap<Arc<str>, Arc<str>>,
    /// Expected output from the template
    pub expected_output: Arc<str>
}

/// Template tag for categorization
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TemplateTag {
    /// Tag name
    pub name: Arc<str>,
    /// Optional color for UI display
    pub color: Option<Arc<str>>,
    /// Optional description of the tag
    pub description: Option<Arc<str>>
}

impl TemplateTag {
    /// Create a new template tag with the given name
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            color: None,
            description: None
        }
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
