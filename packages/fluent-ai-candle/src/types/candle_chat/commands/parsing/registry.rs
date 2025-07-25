//! Command registry and information types
//!
//! Defines command metadata structures for registration, discovery, and documentation.

/// Command parameter type
#[derive(Debug, Clone)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Enum,
    Path}

/// Parameter information for command documentation
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub name: String,
    pub description: String,
    pub parameter_type: ParameterType,
    pub required: bool,
    pub default_value: Option<String>}

/// Command information for registration and help
#[derive(Debug, Clone)]
pub struct CommandInfo {
    pub name: String,
    pub description: String,
    pub usage: String,
    pub parameters: Vec<ParameterInfo>,
    pub aliases: Vec<String>,
    pub category: String,
    pub examples: Vec<String>}