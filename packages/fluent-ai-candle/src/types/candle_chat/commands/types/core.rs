//! Core command types and error definitions
//!
//! Fundamental types for the command system including error handling,
//! parameter definitions, and resource tracking with zero-allocation design.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Command execution errors with minimal allocations
#[derive(Error, Debug, Clone)]
pub enum CommandError {
    #[error("Unknown command: {command}")]
    UnknownCommand { command: String },
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("Invalid syntax: {detail}")]
    InvalidSyntax { detail: String },
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Permission denied")]
    PermissionDenied,
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Configuration error: {detail}")]
    ConfigurationError { detail: String },
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Command timeout")]
    Timeout,
    #[error("Resource not found")]
    NotFound,
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type for command operations
pub type CommandResult<T> = Result<T, CommandError>;

/// Parameter type enumeration for command parameters
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ParameterType {
    /// String parameter
    String,
    /// Integer parameter
    Integer,
    /// Float parameter
    Float,
    /// Boolean parameter
    Boolean,
    /// Array of strings
    StringArray,
    /// File path parameter
    FilePath,
    /// URL parameter
    Url,
    /// JSON object parameter
    Json,
    /// Enumeration parameter with possible values
    Enum,
    /// Path parameter for file/directory paths
    Path,
}

/// Parameter information for command definitions with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: String,
    /// Parameter description  
    pub description: String,
    /// Parameter type
    pub parameter_type: ParameterType,
    /// Whether the parameter is required
    pub required: bool,
    /// Default value if not required
    pub default_value: Option<String>,
}

/// Command information for command registry with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandInfo {
    /// Command name
    pub name: String,
    /// Command description
    pub description: String,
    /// Usage string
    pub usage: String,
    /// Command parameters
    pub parameters: Vec<ParameterInfo>,
    /// Command aliases
    pub aliases: Vec<String>,
    /// Command category
    pub category: String,
    /// Usage examples
    pub examples: Vec<String>,
}

/// Resource usage tracking for command execution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceUsage {
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU time in microseconds
    pub cpu_time_us: u64,
    /// Number of network requests made
    pub network_requests: u32,
    /// Number of disk operations performed
    pub disk_operations: u32,
}