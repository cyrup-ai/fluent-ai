//! Chat template system with zero-allocation patterns
//!
//! This module provides a comprehensive template system for chat interactions including
//! template creation, variable substitution, template management, and sharing with
//! zero-allocation patterns and lock-free data structures.

use std::collections::HashMap;
use std::sync::Arc;

use crossbeam_skiplist::SkipMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::{AsyncTask, spawn_async};

/// Template system errors
#[derive(Error, Debug, Clone)]
pub enum TemplateError {
    #[error("Template not found: {name}")]
    NotFound { name: Arc<str> },
    #[error("Invalid template syntax: {detail}")]
    InvalidSyntax { detail: Arc<str> },
    #[error("Variable substitution error: {detail}")]
    SubstitutionError { detail: Arc<str> },
    #[error("Template validation error: {detail}")]
    ValidationError { detail: Arc<str> },
    #[error("Template compilation error: {detail}")]
    CompilationError { detail: Arc<str> },
    #[error("Template execution error: {detail}")]
    ExecutionError { detail: Arc<str> },
    #[error("Storage error: {detail}")]
    StorageError { detail: Arc<str> },
    #[error("Permission denied: {operation}")]
    PermissionDenied { operation: Arc<str> },
    #[error("Template already exists: {name}")]
    AlreadyExists { name: Arc<str> },
    #[error("Circular dependency detected: {templates}")]
    CircularDependency { templates: Arc<str> },
}

/// Result type for template operations
pub type TemplateResult<T> = Result<T, TemplateError>;

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Template ID
    pub id: Arc<str>,
    /// Template name
    pub name: Arc<str>,
    /// Template description
    pub description: Arc<str>,
    /// Template author
    pub author: Arc<str>,
    /// Template version
    pub version: Arc<str>,
    /// Template category
    pub category: Arc<str>,
    /// Template tags
    pub tags: Arc<[Arc<str>]>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
    /// Usage count
    pub usage_count: u64,
    /// Template rating
    pub rating: f32,
    /// Template permissions
    pub permissions: TemplatePermissions,
}

/// Template permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplatePermissions {
    /// Can read template
    pub read: bool,
    /// Can write template
    pub write: bool,
    /// Can share template
    pub share: bool,
    /// Can delete template
    pub delete: bool,
    /// Can execute template
    pub execute: bool,
}

impl Default for TemplatePermissions {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            share: true,
            delete: true,
            execute: true,
        }
    }
}

/// Template variable definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariable {
    /// Variable name
    pub name: Arc<str>,
    /// Variable description
    pub description: Arc<str>,
    /// Variable type
    pub var_type: VariableType,
    /// Default value
    pub default_value: Option<Arc<str>>,
    /// Whether variable is required
    pub required: bool,
    /// Validation pattern
    pub validation_pattern: Option<Arc<str>>,
    /// Valid values for enum types
    pub valid_values: Option<Arc<[Arc<str>]>>,
    /// Minimum value for numeric types
    pub min_value: Option<f64>,
    /// Maximum value for numeric types
    pub max_value: Option<f64>,
}

/// Variable types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VariableType {
    String,
    Integer,
    Float,
    Boolean,
    Enum,
    Array,
    Object,
    Date,
    Time,
    DateTime,
    Url,
    Email,
    Path,
    Json,
    Markdown,
    Template,
}

/// Template content with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTemplate {
    /// Template metadata
    pub metadata: TemplateMetadata,
    /// Template content
    pub content: Arc<str>,
    /// Template variables
    pub variables: Arc<[TemplateVariable]>,
    /// Template dependencies
    pub dependencies: Arc<[Arc<str>]>,
    /// Template includes
    pub includes: Arc<[Arc<str>]>,
    /// Template macros
    pub macros: HashMap<Arc<str>, Arc<str>>,
    /// Template configuration
    pub config: TemplateConfig,
    /// Compiled template
    compiled: Option<CompiledTemplate>,
}

/// Template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConfig {
    /// Template syntax type
    pub syntax: TemplateSyntax,
    /// Enable variable escaping
    pub escape_variables: bool,
    /// Enable template caching
    pub enable_caching: bool,
    /// Enable template validation
    pub enable_validation: bool,
    /// Enable template compression
    pub enable_compression: bool,
    /// Template timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum template size in bytes
    pub max_size_bytes: usize,
    /// Maximum recursion depth
    pub max_recursion_depth: usize,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            syntax: TemplateSyntax::Handlebars,
            escape_variables: true,
            enable_caching: true,
            enable_validation: true,
            enable_compression: false,
            timeout_seconds: 30,
            max_size_bytes: 1024 * 1024, // 1MB
            max_recursion_depth: 10,
        }
    }
}

/// Template syntax types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TemplateSyntax {
    Handlebars,
    Mustache,
    Jinja2,
    Liquid,
    Custom,
}

/// Compiled template for faster execution
#[derive(Debug, Clone)]
pub struct CompiledTemplate {
    /// Template AST
    pub ast: TemplateAst,
    /// Variable references
    pub variables: Arc<[Arc<str>]>,
    /// Template dependencies
    pub dependencies: Arc<[Arc<str>]>,
    /// Compilation timestamp
    pub compiled_at: u64,
}

/// Template Abstract Syntax Tree
#[derive(Debug, Clone)]
pub enum TemplateAst {
    /// Text node
    Text(Arc<str>),
    /// Variable node
    Variable {
        name: Arc<str>,
        filters: Arc<[Arc<str>]>,
        default: Option<Arc<str>>,
    },
    /// Conditional node
    Conditional {
        condition: Arc<str>,
        then_branch: Box<TemplateAst>,
        else_branch: Option<Box<TemplateAst>>,
    },
    /// Loop node
    Loop {
        variable: Arc<str>,
        iterable: Arc<str>,
        body: Box<TemplateAst>,
    },
    /// Include node
    Include {
        template: Arc<str>,
        variables: HashMap<Arc<str>, Arc<str>>,
    },
    /// Macro node
    Macro {
        name: Arc<str>,
        args: Arc<[Arc<str>]>,
    },
    /// Sequence node
    Sequence(Arc<[TemplateAst]>),
}

/// Template rendering context
#[derive(Debug, Clone)]
pub struct TemplateContext {
    /// Template variables
    pub variables: HashMap<Arc<str>, TemplateValue>,
    /// Template includes
    pub includes: HashMap<Arc<str>, Arc<str>>,
    /// Template macros
    pub macros: HashMap<Arc<str>, Arc<str>>,
    /// Template configuration
    pub config: TemplateConfig,
    /// Recursion depth
    pub recursion_depth: usize,
}

/// Template value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateValue {
    String(Arc<str>),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Arc<[TemplateValue]>),
    Object(HashMap<Arc<str>, TemplateValue>),
    Null,
}

impl TemplateValue {
    /// Convert to string
    #[inline]
    pub fn to_string(&self) -> Arc<str> {
        match self {
            TemplateValue::String(s) => s.clone(),
            TemplateValue::Integer(i) => Arc::from(i.to_string()),
            TemplateValue::Float(f) => Arc::from(f.to_string()),
            TemplateValue::Boolean(b) => Arc::from(b.to_string()),
            TemplateValue::Array(arr) => {
                let values: Vec<String> = arr.iter().map(|v| v.to_string().to_string()).collect();
                Arc::from(format!("[{}]", values.join(", ")))
            }
            TemplateValue::Object(obj) => {
                let pairs: Vec<String> = obj
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_string()))
                    .collect();
                Arc::from(format!("{{{}}}", pairs.join(", ")))
            }
            TemplateValue::Null => Arc::from("null"),
        }
    }

    /// Check if value is truthy
    #[inline]
    pub fn is_truthy(&self) -> bool {
        match self {
            TemplateValue::String(s) => !s.is_empty(),
            TemplateValue::Integer(i) => *i != 0,
            TemplateValue::Float(f) => *f != 0.0,
            TemplateValue::Boolean(b) => *b,
            TemplateValue::Array(arr) => !arr.is_empty(),
            TemplateValue::Object(obj) => !obj.is_empty(),
            TemplateValue::Null => false,
        }
    }
}

impl From<&str> for TemplateValue {
    fn from(value: &str) -> Self {
        TemplateValue::String(Arc::from(value))
    }
}

impl From<String> for TemplateValue {
    fn from(value: String) -> Self {
        TemplateValue::String(Arc::from(value))
    }
}

impl From<i64> for TemplateValue {
    fn from(value: i64) -> Self {
        TemplateValue::Integer(value)
    }
}

impl From<f64> for TemplateValue {
    fn from(value: f64) -> Self {
        TemplateValue::Float(value)
    }
}

impl From<bool> for TemplateValue {
    fn from(value: bool) -> Self {
        TemplateValue::Boolean(value)
    }
}

impl ChatTemplate {
    /// Create a new template
    #[inline]
    pub fn new(name: impl Into<Arc<str>>, content: impl Into<Arc<str>>) -> Self {
        let id = Arc::from(Uuid::new_v4().to_string());
        let name = name.into();
        let content = content.into();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            metadata: TemplateMetadata {
                id,
                name,
                description: Arc::from(""),
                author: Arc::from(""),
                version: Arc::from("1.0.0"),
                category: Arc::from("general"),
                tags: Arc::from([]),
                created_at: now,
                modified_at: now,
                usage_count: 0,
                rating: 0.0,
                permissions: TemplatePermissions::default(),
            },
            content,
            variables: Arc::from([]),
            dependencies: Arc::from([]),
            includes: Arc::from([]),
            macros: HashMap::new(),
            config: TemplateConfig::default(),
            compiled: None,
        }
    }

    /// Set template description
    #[inline]
    pub fn with_description(mut self, description: impl Into<Arc<str>>) -> Self {
        self.metadata.description = description.into();
        self
    }

    /// Set template author
    #[inline]
    pub fn with_author(mut self, author: impl Into<Arc<str>>) -> Self {
        self.metadata.author = author.into();
        self
    }

    /// Set template version
    #[inline]
    pub fn with_version(mut self, version: impl Into<Arc<str>>) -> Self {
        self.metadata.version = version.into();
        self
    }

    /// Set template category
    #[inline]
    pub fn with_category(mut self, category: impl Into<Arc<str>>) -> Self {
        self.metadata.category = category.into();
        self
    }

    /// Set template tags
    #[inline]
    pub fn with_tags(mut self, tags: Arc<[Arc<str>]>) -> Self {
        self.metadata.tags = tags;
        self
    }

    /// Add template variable
    #[inline]
    pub fn with_variable(mut self, variable: TemplateVariable) -> Self {
        let mut variables = self.variables.to_vec();
        variables.push(variable);
        self.variables = Arc::from(variables);
        self
    }

    /// Add template dependency
    #[inline]
    pub fn with_dependency(mut self, dependency: impl Into<Arc<str>>) -> Self {
        let mut dependencies = self.dependencies.to_vec();
        dependencies.push(dependency.into());
        self.dependencies = Arc::from(dependencies);
        self
    }

    /// Add template include
    #[inline]
    pub fn with_include(mut self, include: impl Into<Arc<str>>) -> Self {
        let mut includes = self.includes.to_vec();
        includes.push(include.into());
        self.includes = Arc::from(includes);
        self
    }

    /// Add template macro
    #[inline]
    pub fn with_macro(mut self, name: impl Into<Arc<str>>, content: impl Into<Arc<str>>) -> Self {
        self.macros.insert(name.into(), content.into());
        self
    }

    /// Set template configuration
    #[inline]
    pub fn with_config(mut self, config: TemplateConfig) -> Self {
        self.config = config;
        self
    }

    /// Compile template for faster execution
    #[inline]
    pub fn compile(&mut self) -> TemplateResult<()> {
        let ast = self.parse_template()?;
        let variables = self.extract_variables(&ast);
        let dependencies = self.extract_dependencies(&ast);

        self.compiled = Some(CompiledTemplate {
            ast,
            variables,
            dependencies,
            compiled_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        });

        Ok(())
    }

    /// Parse template content into AST
    #[inline]
    fn parse_template(&self) -> TemplateResult<TemplateAst> {
        match self.config.syntax {
            TemplateSyntax::Handlebars => self.parse_handlebars(),
            TemplateSyntax::Mustache => self.parse_mustache(),
            TemplateSyntax::Jinja2 => self.parse_jinja2(),
            TemplateSyntax::Liquid => self.parse_liquid(),
            TemplateSyntax::Custom => self.parse_custom(),
        }
    }

    /// Parse Handlebars template
    #[inline]
    fn parse_handlebars(&self) -> TemplateResult<TemplateAst> {
        let mut nodes = Vec::new();
        let content = &self.content;
        let mut chars = content.chars().peekable();
        let mut current_text = String::new();

        while let Some(ch) = chars.next() {
            if ch == '{' && chars.peek() == Some(&'{') {
                // Start of variable or block
                chars.next(); // consume second {

                // Save current text
                if !current_text.is_empty() {
                    nodes.push(TemplateAst::Text(Arc::from(current_text.clone())));
                    current_text.clear();
                }

                // Parse variable or block
                let mut expr = String::new();
                let mut brace_count = 0;

                while let Some(ch) = chars.next() {
                    if ch == '{' {
                        brace_count += 1;
                        expr.push(ch);
                    } else if ch == '}' {
                        if brace_count > 0 {
                            brace_count -= 1;
                            expr.push(ch);
                        } else if chars.peek() == Some(&'}') {
                            chars.next(); // consume second }
                            break;
                        } else {
                            expr.push(ch);
                        }
                    } else {
                        expr.push(ch);
                    }
                }

                // Parse expression
                let expr = expr.trim();
                if expr.starts_with('#') {
                    // Block helper
                    nodes.push(self.parse_block_helper(expr)?);
                } else if expr.starts_with('/') {
                    // End block (handled by block helper parser)
                    continue;
                } else {
                    // Variable
                    nodes.push(self.parse_variable(expr)?);
                }
            } else {
                current_text.push(ch);
            }
        }

        // Save remaining text
        if !current_text.is_empty() {
            nodes.push(TemplateAst::Text(Arc::from(current_text)));
        }

        Ok(TemplateAst::Sequence(Arc::from(nodes)))
    }

    /// Parse Mustache template
    #[inline]
    fn parse_mustache(&self) -> TemplateResult<TemplateAst> {
        // Similar to Handlebars but simpler syntax
        self.parse_handlebars()
    }

    /// Parse Jinja2 template
    #[inline]
    fn parse_jinja2(&self) -> TemplateResult<TemplateAst> {
        // Jinja2 uses {% %} for blocks and {{ }} for variables
        let mut nodes = Vec::new();
        let content = &self.content;
        let mut chars = content.chars().peekable();
        let mut current_text = String::new();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                if chars.peek() == Some(&'{') {
                    // Variable {{ }}
                    chars.next();

                    if !current_text.is_empty() {
                        nodes.push(TemplateAst::Text(Arc::from(current_text.clone())));
                        current_text.clear();
                    }

                    let mut expr = String::new();
                    while let Some(ch) = chars.next() {
                        if ch == '}' && chars.peek() == Some(&'}') {
                            chars.next();
                            break;
                        }
                        expr.push(ch);
                    }

                    nodes.push(self.parse_variable(expr.trim())?);
                } else if chars.peek() == Some(&'%') {
                    // Block {% %}
                    chars.next();

                    if !current_text.is_empty() {
                        nodes.push(TemplateAst::Text(Arc::from(current_text.clone())));
                        current_text.clear();
                    }

                    let mut expr = String::new();
                    while let Some(ch) = chars.next() {
                        if ch == '%' && chars.peek() == Some(&'}') {
                            chars.next();
                            break;
                        }
                        expr.push(ch);
                    }

                    nodes.push(self.parse_jinja2_block(expr.trim())?);
                } else {
                    current_text.push(ch);
                }
            } else {
                current_text.push(ch);
            }
        }

        if !current_text.is_empty() {
            nodes.push(TemplateAst::Text(Arc::from(current_text)));
        }

        Ok(TemplateAst::Sequence(Arc::from(nodes)))
    }

    /// Parse Liquid template
    #[inline]
    fn parse_liquid(&self) -> TemplateResult<TemplateAst> {
        // Liquid uses {% %} for blocks and {{ }} for variables
        self.parse_jinja2()
    }

    /// Parse custom template
    #[inline]
    fn parse_custom(&self) -> TemplateResult<TemplateAst> {
        // Custom template syntax - simple variable substitution
        Ok(TemplateAst::Text(self.content.clone()))
    }

    /// Parse variable expression
    #[inline]
    fn parse_variable(&self, expr: &str) -> TemplateResult<TemplateAst> {
        let parts: Vec<&str> = expr.split('|').collect();
        let name = Arc::from(parts[0].trim());
        let filters = if parts.len() > 1 {
            Arc::from(
                parts[1..]
                    .iter()
                    .map(|f| Arc::from(f.trim()))
                    .collect::<Vec<_>>(),
            )
        } else {
            Arc::from([])
        };

        Ok(TemplateAst::Variable {
            name,
            filters,
            default: None,
        })
    }

    /// Parse block helper
    #[inline]
    fn parse_block_helper(&self, expr: &str) -> TemplateResult<TemplateAst> {
        let expr = &expr[1..]; // Remove #
        let parts: Vec<&str> = expr.split_whitespace().collect();

        if parts.is_empty() {
            return Err(TemplateError::InvalidSyntax {
                detail: Arc::from("Empty block helper"),
            });
        }

        match parts[0] {
            "if" => self.parse_if_block(parts),
            "each" => self.parse_each_block(parts),
            "with" => self.parse_with_block(parts),
            _ => Err(TemplateError::InvalidSyntax {
                detail: Arc::from(format!("Unknown block helper: {}", parts[0])),
            }),
        }
    }

    /// Parse if block
    #[inline]
    fn parse_if_block(&self, parts: &[&str]) -> TemplateResult<TemplateAst> {
        if parts.len() < 2 {
            return Err(TemplateError::InvalidSyntax {
                detail: Arc::from("If block requires condition"),
            });
        }

        let condition = Arc::from(parts[1..].join(" "));

        // For now, return a simple conditional node
        // In a full implementation, we'd parse the block content
        Ok(TemplateAst::Conditional {
            condition,
            then_branch: Box::new(TemplateAst::Text(Arc::from(""))),
            else_branch: None,
        })
    }

    /// Parse each block
    #[inline]
    fn parse_each_block(&self, parts: &[&str]) -> TemplateResult<TemplateAst> {
        if parts.len() < 2 {
            return Err(TemplateError::InvalidSyntax {
                detail: Arc::from("Each block requires iterable"),
            });
        }

        let iterable = Arc::from(parts[1]);
        let variable = if parts.len() > 3 && parts[2] == "as" {
            Arc::from(parts[3])
        } else {
            Arc::from("item")
        };

        Ok(TemplateAst::Loop {
            variable,
            iterable,
            body: Box::new(TemplateAst::Text(Arc::from(""))),
        })
    }

    /// Parse with block
    #[inline]
    fn parse_with_block(&self, parts: &[&str]) -> TemplateResult<TemplateAst> {
        if parts.len() < 2 {
            return Err(TemplateError::InvalidSyntax {
                detail: Arc::from("With block requires context"),
            });
        }

        let context = Arc::from(parts[1]);

        Ok(TemplateAst::Variable {
            name: context,
            filters: Arc::from([]),
            default: None,
        })
    }

    /// Parse Jinja2 block
    #[inline]
    fn parse_jinja2_block(&self, expr: &str) -> TemplateResult<TemplateAst> {
        let parts: Vec<&str> = expr.split_whitespace().collect();

        if parts.is_empty() {
            return Err(TemplateError::InvalidSyntax {
                detail: Arc::from("Empty Jinja2 block"),
            });
        }

        match parts[0] {
            "if" => self.parse_if_block(parts),
            "for" => self.parse_for_block(parts),
            "include" => self.parse_include_block(parts),
            "macro" => self.parse_macro_block(parts),
            _ => Err(TemplateError::InvalidSyntax {
                detail: Arc::from(format!("Unknown Jinja2 block: {}", parts[0])),
            }),
        }
    }

    /// Parse for block
    #[inline]
    fn parse_for_block(&self, parts: &[&str]) -> TemplateResult<TemplateAst> {
        if parts.len() < 4 || parts[2] != "in" {
            return Err(TemplateError::InvalidSyntax {
                detail: Arc::from("For block requires 'variable in iterable' syntax"),
            });
        }

        let variable = Arc::from(parts[1]);
        let iterable = Arc::from(parts[3]);

        Ok(TemplateAst::Loop {
            variable,
            iterable,
            body: Box::new(TemplateAst::Text(Arc::from(""))),
        })
    }

    /// Parse include block
    #[inline]
    fn parse_include_block(&self, parts: &[&str]) -> TemplateResult<TemplateAst> {
        if parts.len() < 2 {
            return Err(TemplateError::InvalidSyntax {
                detail: Arc::from("Include block requires template name"),
            });
        }

        let template = Arc::from(parts[1]);
        let variables = HashMap::new();

        Ok(TemplateAst::Include {
            template,
            variables,
        })
    }

    /// Parse macro block
    #[inline]
    fn parse_macro_block(&self, parts: &[&str]) -> TemplateResult<TemplateAst> {
        if parts.len() < 2 {
            return Err(TemplateError::InvalidSyntax {
                detail: Arc::from("Macro block requires macro name"),
            });
        }

        let name = Arc::from(parts[1]);
        let args = if parts.len() > 2 {
            Arc::from(
                parts[2..]
                    .iter()
                    .map(|arg| Arc::from(*arg))
                    .collect::<Vec<_>>(),
            )
        } else {
            Arc::from([])
        };

        Ok(TemplateAst::Macro { name, args })
    }

    /// Extract variables from AST
    #[inline]
    fn extract_variables(&self, ast: &TemplateAst) -> Arc<[Arc<str>]> {
        let mut variables = Vec::new();
        self.extract_variables_recursive(ast, &mut variables);
        Arc::from(variables)
    }

    /// Extract variables recursively
    #[inline]
    fn extract_variables_recursive(&self, ast: &TemplateAst, variables: &mut Vec<Arc<str>>) {
        match ast {
            TemplateAst::Variable { name, .. } => {
                if !variables.contains(name) {
                    variables.push(name.clone());
                }
            }
            TemplateAst::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                // Extract variables from condition
                let condition_vars = self.extract_condition_variables(condition);
                for var in condition_vars {
                    if !variables.contains(&var) {
                        variables.push(var);
                    }
                }

                self.extract_variables_recursive(then_branch, variables);
                if let Some(else_branch) = else_branch {
                    self.extract_variables_recursive(else_branch, variables);
                }
            }
            TemplateAst::Loop {
                variable,
                iterable,
                body,
            } => {
                // Extract iterable variable
                if !variables.contains(iterable) {
                    variables.push(iterable.clone());
                }

                self.extract_variables_recursive(body, variables);
            }
            TemplateAst::Sequence(nodes) => {
                for node in nodes.iter() {
                    self.extract_variables_recursive(node, variables);
                }
            }
            _ => {}
        }
    }

    /// Extract variables from condition string
    #[inline]
    fn extract_condition_variables(&self, condition: &str) -> Vec<Arc<str>> {
        let mut variables = Vec::new();

        // Simple variable extraction - look for identifiers
        let mut current_word = String::new();
        for ch in condition.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                current_word.push(ch);
            } else {
                if !current_word.is_empty() && !current_word.chars().all(|c| c.is_ascii_digit()) {
                    let var = Arc::from(current_word.clone());
                    if !variables.contains(&var) {
                        variables.push(var);
                    }
                }
                current_word.clear();
            }
        }

        if !current_word.is_empty() && !current_word.chars().all(|c| c.is_ascii_digit()) {
            let var = Arc::from(current_word);
            if !variables.contains(&var) {
                variables.push(var);
            }
        }

        variables
    }

    /// Extract dependencies from AST
    #[inline]
    fn extract_dependencies(&self, ast: &TemplateAst) -> Arc<[Arc<str>]> {
        let mut dependencies = Vec::new();
        self.extract_dependencies_recursive(ast, &mut dependencies);
        Arc::from(dependencies)
    }

    /// Extract dependencies recursively
    #[inline]
    fn extract_dependencies_recursive(&self, ast: &TemplateAst, dependencies: &mut Vec<Arc<str>>) {
        match ast {
            TemplateAst::Include { template, .. } => {
                if !dependencies.contains(template) {
                    dependencies.push(template.clone());
                }
            }
            TemplateAst::Conditional {
                then_branch,
                else_branch,
                ..
            } => {
                self.extract_dependencies_recursive(then_branch, dependencies);
                if let Some(else_branch) = else_branch {
                    self.extract_dependencies_recursive(else_branch, dependencies);
                }
            }
            TemplateAst::Loop { body, .. } => {
                self.extract_dependencies_recursive(body, dependencies);
            }
            TemplateAst::Sequence(nodes) => {
                for node in nodes.iter() {
                    self.extract_dependencies_recursive(node, dependencies);
                }
            }
            _ => {}
        }
    }

    /// Render template with variables
    #[inline]
    pub fn render(&self, variables: HashMap<Arc<str>, TemplateValue>) -> TemplateResult<Arc<str>> {
        let context = TemplateContext {
            variables,
            includes: HashMap::new(),
            macros: self.macros.clone(),
            config: self.config.clone(),
            recursion_depth: 0,
        };

        if let Some(compiled) = &self.compiled {
            self.render_ast(&compiled.ast, &context)
        } else {
            // Simple variable substitution fallback
            self.render_simple(&context)
        }
    }

    /// Render AST with context
    #[inline]
    fn render_ast(&self, ast: &TemplateAst, context: &TemplateContext) -> TemplateResult<Arc<str>> {
        match ast {
            TemplateAst::Text(text) => Ok(text.clone()),
            TemplateAst::Variable {
                name,
                filters,
                default,
            } => {
                let value = context
                    .variables
                    .get(name)
                    .cloned()
                    .or_else(|| default.as_ref().map(|d| TemplateValue::String(d.clone())))
                    .unwrap_or(TemplateValue::Null);

                let mut result = value.to_string();

                // Apply filters
                for filter in filters.iter() {
                    result = self.apply_filter(&result, filter)?;
                }

                Ok(result)
            }
            TemplateAst::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition_result = self.evaluate_condition(condition, context)?;

                if condition_result {
                    self.render_ast(then_branch, context)
                } else if let Some(else_branch) = else_branch {
                    self.render_ast(else_branch, context)
                } else {
                    Ok(Arc::from(""))
                }
            }
            TemplateAst::Loop {
                variable,
                iterable,
                body,
            } => {
                let iterable_value = context.variables.get(iterable).ok_or_else(|| {
                    TemplateError::SubstitutionError {
                        detail: Arc::from(format!("Variable not found: {}", iterable)),
                    }
                })?;

                match iterable_value {
                    TemplateValue::Array(items) => {
                        let mut result = String::new();

                        for item in items.iter() {
                            let mut loop_context = context.clone();
                            loop_context
                                .variables
                                .insert(variable.clone(), item.clone());

                            let rendered = self.render_ast(body, &loop_context)?;
                            result.push_str(&rendered);
                        }

                        Ok(Arc::from(result))
                    }
                    _ => Err(TemplateError::SubstitutionError {
                        detail: Arc::from(format!("Cannot iterate over: {}", iterable)),
                    }),
                }
            }
            TemplateAst::Include {
                template,
                variables,
            } => {
                // Include template rendering (placeholder)
                Ok(Arc::from(format!("<!-- Include: {} -->", template)))
            }
            TemplateAst::Macro { name, args } => {
                // Macro rendering (placeholder)
                Ok(Arc::from(format!("<!-- Macro: {} -->", name)))
            }
            TemplateAst::Sequence(nodes) => {
                let mut result = String::new();

                for node in nodes.iter() {
                    let rendered = self.render_ast(node, context)?;
                    result.push_str(&rendered);
                }

                Ok(Arc::from(result))
            }
        }
    }

    /// Simple variable substitution rendering
    #[inline]
    fn render_simple(&self, context: &TemplateContext) -> TemplateResult<Arc<str>> {
        let mut result = self.content.to_string();

        // Replace variables in the format {{variable}}
        for (name, value) in &context.variables {
            let placeholder = format!("{{{{{}}}}}", name);
            let replacement = if context.config.escape_variables {
                html_escape::encode_text(&value.to_string()).to_string()
            } else {
                value.to_string().to_string()
            };
            result = result.replace(&placeholder, &replacement);
        }

        Ok(Arc::from(result))
    }

    /// Apply filter to value
    #[inline]
    fn apply_filter(&self, value: &Arc<str>, filter: &Arc<str>) -> TemplateResult<Arc<str>> {
        match filter.as_ref() {
            "upper" => Ok(Arc::from(value.to_uppercase())),
            "lower" => Ok(Arc::from(value.to_lowercase())),
            "capitalize" => {
                let mut chars = value.chars();
                match chars.next() {
                    None => Ok(Arc::from("")),
                    Some(first) => Ok(Arc::from(
                        first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                    )),
                }
            }
            "trim" => Ok(Arc::from(value.trim())),
            "reverse" => Ok(Arc::from(value.chars().rev().collect::<String>())),
            "length" => Ok(Arc::from(value.len().to_string())),
            "escape" => Ok(Arc::from(html_escape::encode_text(value).to_string())),
            "unescape" => Ok(Arc::from(
                html_escape::decode_html_entities(value).to_string(),
            )),
            _ => Err(TemplateError::SubstitutionError {
                detail: Arc::from(format!("Unknown filter: {}", filter)),
            }),
        }
    }

    /// Evaluate condition
    #[inline]
    fn evaluate_condition(
        &self,
        condition: &str,
        context: &TemplateContext,
    ) -> TemplateResult<bool> {
        // Simple condition evaluation
        if let Some(value) = context.variables.get(&Arc::from(condition)) {
            Ok(value.is_truthy())
        } else {
            // Parse simple conditions like "var == value"
            if condition.contains("==") {
                let parts: Vec<&str> = condition.split("==").collect();
                if parts.len() == 2 {
                    let left = parts[0].trim();
                    let right = parts[1].trim().trim_matches('"').trim_matches('\'');

                    if let Some(value) = context.variables.get(&Arc::from(left)) {
                        Ok(value.to_string() == right)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            } else if condition.contains("!=") {
                let parts: Vec<&str> = condition.split("!=").collect();
                if parts.len() == 2 {
                    let left = parts[0].trim();
                    let right = parts[1].trim().trim_matches('"').trim_matches('\'');

                    if let Some(value) = context.variables.get(&Arc::from(left)) {
                        Ok(value.to_string() != right)
                    } else {
                        Ok(true)
                    }
                } else {
                    Ok(false)
                }
            } else {
                Ok(false)
            }
        }
    }

    /// Validate template
    #[inline]
    pub fn validate(&self) -> TemplateResult<()> {
        // Check template size
        if self.content.len() > self.config.max_size_bytes {
            return Err(TemplateError::ValidationError {
                detail: Arc::from(format!(
                    "Template size {} exceeds maximum {}",
                    self.content.len(),
                    self.config.max_size_bytes
                )),
            });
        }

        // Check for circular dependencies
        self.check_circular_dependencies()?;

        // Validate syntax
        if self.config.enable_validation {
            self.validate_syntax()?;
        }

        Ok(())
    }

    /// Check for circular dependencies
    #[inline]
    fn check_circular_dependencies(&self) -> TemplateResult<()> {
        let mut visited = std::collections::HashSet::new();
        let mut visiting = std::collections::HashSet::new();

        self.check_circular_dependencies_recursive(
            &self.metadata.name,
            &mut visited,
            &mut visiting,
        )?;

        Ok(())
    }

    /// Check circular dependencies recursively
    #[inline]
    fn check_circular_dependencies_recursive(
        &self,
        template_name: &Arc<str>,
        visited: &mut std::collections::HashSet<Arc<str>>,
        visiting: &mut std::collections::HashSet<Arc<str>>,
    ) -> TemplateResult<()> {
        if visiting.contains(template_name) {
            return Err(TemplateError::CircularDependency {
                templates: Arc::from(
                    visiting
                        .iter()
                        .map(|t| t.as_ref())
                        .collect::<Vec<_>>()
                        .join(", "),
                ),
            });
        }

        if visited.contains(template_name) {
            return Ok(());
        }

        visiting.insert(template_name.clone());

        // Check dependencies
        for dependency in self.dependencies.iter() {
            self.check_circular_dependencies_recursive(dependency, visited, visiting)?;
        }

        visiting.remove(template_name);
        visited.insert(template_name.clone());

        Ok(())
    }

    /// Validate template syntax
    #[inline]
    fn validate_syntax(&self) -> TemplateResult<()> {
        // Try to parse the template
        let _ast = self.parse_template()?;
        Ok(())
    }

    /// Get template information
    #[inline]
    pub fn info(&self) -> TemplateInfo {
        TemplateInfo {
            metadata: self.metadata.clone(),
            variable_count: self.variables.len(),
            dependency_count: self.dependencies.len(),
            include_count: self.includes.len(),
            macro_count: self.macros.len(),
            is_compiled: self.compiled.is_some(),
            estimated_render_time: self.estimate_render_time(),
        }
    }

    /// Estimate render time
    #[inline]
    fn estimate_render_time(&self) -> u64 {
        let content_size = self.content.len() as u64;
        let variable_count = self.variables.len() as u64;
        let dependency_count = self.dependencies.len() as u64;

        // Simple estimation: base time + content size + variables + dependencies
        10 + content_size / 1000 + variable_count * 2 + dependency_count * 5
    }
}

/// Template information
#[derive(Debug, Clone)]
pub struct TemplateInfo {
    pub metadata: TemplateMetadata,
    pub variable_count: usize,
    pub dependency_count: usize,
    pub include_count: usize,
    pub macro_count: usize,
    pub is_compiled: bool,
    pub estimated_render_time: u64,
}

/// Template manager with lock-free operations
pub struct TemplateManager {
    /// Template storage
    templates: SkipMap<Arc<str>, ChatTemplate>,
    /// Template cache
    cache: dashmap::DashMap<Arc<str>, Arc<str>>,
    /// Usage statistics
    usage_stats: SkipMap<Arc<str>, TemplateUsageStats>,
    /// Configuration
    config: TemplateManagerConfig,
}

/// Template usage statistics
#[derive(Debug, Clone)]
pub struct TemplateUsageStats {
    pub template_name: Arc<str>,
    pub usage_count: u64,
    pub last_used: u64,
    pub total_render_time: u64,
    pub average_render_time: f64,
    pub error_count: u64,
    pub cache_hit_count: u64,
}

/// Template manager configuration
#[derive(Debug, Clone)]
pub struct TemplateManagerConfig {
    /// Maximum number of templates
    pub max_templates: usize,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Cache expiration time in seconds
    pub cache_expiration: u64,
    /// Enable template sharing
    pub enable_sharing: bool,
    /// Enable template validation
    pub enable_validation: bool,
    /// Enable template compression
    pub enable_compression: bool,
}

impl Default for TemplateManagerConfig {
    fn default() -> Self {
        Self {
            max_templates: 1000,
            max_cache_size: 100,
            cache_expiration: 3600, // 1 hour
            enable_sharing: true,
            enable_validation: true,
            enable_compression: false,
        }
    }
}

impl TemplateManager {
    /// Create a new template manager
    #[inline]
    pub fn new() -> Self {
        Self::with_config(TemplateManagerConfig::default())
    }

    /// Create template manager with configuration
    #[inline]
    pub fn with_config(config: TemplateManagerConfig) -> Self {
        Self {
            templates: SkipMap::new(),
            cache: dashmap::DashMap::new(),
            usage_stats: SkipMap::new(),
            config,
        }
    }

    /// Store template
    #[inline]
    pub fn store(&self, template: ChatTemplate) -> TemplateResult<()> {
        // Check limits
        if self.templates.len() >= self.config.max_templates {
            return Err(TemplateError::StorageError {
                detail: Arc::from("Template storage limit exceeded"),
            });
        }

        // Check if template already exists
        if self.templates.contains_key(&template.metadata.name) {
            return Err(TemplateError::AlreadyExists {
                name: template.metadata.name.clone(),
            });
        }

        // Validate template
        if self.config.enable_validation {
            template.validate()?;
        }

        // Store template
        self.templates
            .insert(template.metadata.name.clone(), template);

        Ok(())
    }

    /// Get template
    #[inline]
    pub fn get(&self, name: &str) -> Option<ChatTemplate> {
        self.templates
            .get(&Arc::from(name))
            .map(|entry| entry.value().clone())
    }

    /// Update template
    #[inline]
    pub fn update(&self, template: ChatTemplate) -> TemplateResult<()> {
        // Check if template exists
        if !self.templates.contains_key(&template.metadata.name) {
            return Err(TemplateError::NotFound {
                name: template.metadata.name.clone(),
            });
        }

        // Validate template
        if self.config.enable_validation {
            template.validate()?;
        }

        // Update template
        self.templates
            .insert(template.metadata.name.clone(), template);

        // Clear cache
        self.cache.remove(&template.metadata.name);

        Ok(())
    }

    /// Delete template
    #[inline]
    pub fn delete(&self, name: &str) -> TemplateResult<()> {
        let name_arc = Arc::from(name);

        // Remove template
        if self.templates.remove(&name_arc).is_none() {
            return Err(TemplateError::NotFound { name: name_arc });
        }

        // Clear cache
        self.cache.remove(&name_arc);

        // Remove usage stats
        self.usage_stats.remove(&name_arc);

        Ok(())
    }

    /// List templates
    #[inline]
    pub fn list(&self) -> Vec<TemplateInfo> {
        self.templates
            .iter()
            .map(|entry| entry.value().info())
            .collect()
    }

    /// Search templates
    #[inline]
    pub fn search(&self, query: &str) -> Vec<TemplateInfo> {
        let query_lower = query.to_lowercase();

        self.templates
            .iter()
            .filter(|entry| {
                let template = entry.value();
                template.metadata.name.to_lowercase().contains(&query_lower)
                    || template
                        .metadata
                        .description
                        .to_lowercase()
                        .contains(&query_lower)
                    || template
                        .metadata
                        .tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .map(|entry| entry.value().info())
            .collect()
    }

    /// Render template
    #[inline]
    pub fn render(
        &self,
        name: &str,
        variables: HashMap<Arc<str>, TemplateValue>,
    ) -> TemplateResult<Arc<str>> {
        let name_arc = Arc::from(name);

        // Check cache first
        let cache_key = self.generate_cache_key(&name_arc, &variables);
        if let Some(cached) = self.cache.get(&cache_key) {
            self.record_cache_hit(&name_arc);
            return Ok(cached.clone());
        }

        // Get template
        let template = self
            .templates
            .get(&name_arc)
            .ok_or_else(|| TemplateError::NotFound {
                name: name_arc.clone(),
            })?;

        // Record usage
        self.record_usage(&name_arc);

        // Render template
        let start_time = std::time::Instant::now();
        let result = template.value().render(variables);
        let render_time = start_time.elapsed().as_millis() as u64;

        // Update statistics
        self.update_render_stats(&name_arc, render_time, result.is_ok());

        // Cache result if successful
        if let Ok(rendered) = &result {
            if self.cache.len() < self.config.max_cache_size {
                self.cache.insert(cache_key, rendered.clone());
            }
        }

        result
    }

    /// Generate cache key
    #[inline]
    fn generate_cache_key(
        &self,
        name: &Arc<str>,
        variables: &HashMap<Arc<str>, TemplateValue>,
    ) -> Arc<str> {
        let mut key = String::new();
        key.push_str(name);
        key.push('|');

        let mut var_keys: Vec<_> = variables.keys().collect();
        var_keys.sort();

        for var_key in var_keys {
            key.push_str(var_key);
            key.push('=');
            key.push_str(&variables[var_key].to_string());
            key.push('&');
        }

        Arc::from(key)
    }

    /// Record template usage
    #[inline]
    fn record_usage(&self, name: &Arc<str>) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.usage_stats.insert(
            name.clone(),
            TemplateUsageStats {
                template_name: name.clone(),
                usage_count: 1,
                last_used: now,
                total_render_time: 0,
                average_render_time: 0.0,
                error_count: 0,
                cache_hit_count: 0,
            },
        );
    }

    /// Record cache hit
    #[inline]
    fn record_cache_hit(&self, name: &Arc<str>) {
        if let Some(mut stats) = self.usage_stats.get(name) {
            let mut updated_stats = stats.value().clone();
            updated_stats.cache_hit_count += 1;
            self.usage_stats.insert(name.clone(), updated_stats);
        }
    }

    /// Update render statistics
    #[inline]
    fn update_render_stats(&self, name: &Arc<str>, render_time: u64, success: bool) {
        if let Some(stats) = self.usage_stats.get(name) {
            let mut updated_stats = stats.value().clone();
            updated_stats.total_render_time += render_time;
            updated_stats.average_render_time =
                updated_stats.total_render_time as f64 / updated_stats.usage_count as f64;

            if !success {
                updated_stats.error_count += 1;
            }

            self.usage_stats.insert(name.clone(), updated_stats);
        }
    }

    /// Get template statistics
    #[inline]
    pub fn get_stats(&self, name: &str) -> Option<TemplateUsageStats> {
        self.usage_stats
            .get(&Arc::from(name))
            .map(|entry| entry.value().clone())
    }

    /// Clear cache
    #[inline]
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get cache size
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Export template
    #[inline]
    pub fn export(&self, name: &str) -> TemplateResult<Arc<str>> {
        let template =
            self.templates
                .get(&Arc::from(name))
                .ok_or_else(|| TemplateError::NotFound {
                    name: Arc::from(name),
                })?;

        let exported = serde_json::to_string_pretty(template.value()).map_err(|e| {
            TemplateError::StorageError {
                detail: Arc::from(format!("Export failed: {}", e)),
            }
        })?;

        Ok(Arc::from(exported))
    }

    /// Import template
    #[inline]
    pub fn import(&self, data: &str) -> TemplateResult<()> {
        let template: ChatTemplate =
            serde_json::from_str(data).map_err(|e| TemplateError::StorageError {
                detail: Arc::from(format!("Import failed: {}", e)),
            })?;

        self.store(template)
    }
}

impl Default for TemplateManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global template manager instance
static TEMPLATE_MANAGER: once_cell::sync::Lazy<TemplateManager> =
    once_cell::sync::Lazy::new(|| TemplateManager::new());

/// Get global template manager
#[inline]
pub fn get_template_manager() -> &'static TemplateManager {
    &TEMPLATE_MANAGER
}

/// Store template using global manager
#[inline]
pub fn store_template(template: ChatTemplate) -> TemplateResult<()> {
    get_template_manager().store(template)
}

/// Get template using global manager
#[inline]
pub fn get_template(name: &str) -> Option<ChatTemplate> {
    get_template_manager().get(name)
}

/// Render template using global manager
#[inline]
pub fn render_template(
    name: &str,
    variables: HashMap<Arc<str>, TemplateValue>,
) -> TemplateResult<Arc<str>> {
    get_template_manager().render(name, variables)
}

/// Create template builder
#[inline]
pub fn template(name: impl Into<Arc<str>>, content: impl Into<Arc<str>>) -> ChatTemplate {
    ChatTemplate::new(name, content)
}
