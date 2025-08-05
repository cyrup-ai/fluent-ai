//! Command execution errors and result types
//!
//! This module defines all error types and result handling for command execution
//! with minimal allocations and clear error reporting.


use thiserror::Error;

/// Command execution errors with minimal allocations
#[derive(Error, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CommandError {
    /// Command name not recognized
    #[error("Unknown command: {command}")]
    UnknownCommand { 
        /// The unrecognized command name
        command: String 
    },
    /// Invalid or malformed arguments provided
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
    /// Syntax error in command structure
    #[error("Invalid syntax: {detail}")]
    InvalidSyntax { 
        /// Details about the syntax error
        detail: String 
    },
    /// Command execution failed
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    /// User lacks permission to execute command
    #[error("Permission denied")]
    PermissionDenied,
    /// Error parsing command parameters
    #[error("Parse error: {message}")]
    ParseError { 
        /// Error message
        message: String,
        /// Input that failed to parse
        input: String,
    },
    /// Command already exists
    #[error("Command already exists: {command}")]
    CommandAlreadyExists { 
        /// Command name that already exists
        command: String 
    },
    /// Alias already exists
    #[error("Alias already exists: {alias}")]
    AliasAlreadyExists { 
        /// Alias name that already exists
        alias: String 
    },
    /// Configuration is invalid or missing
    #[error("Configuration error: {detail}")]
    ConfigurationError { 
        /// Details about the configuration error
        detail: String 
    },
    /// Input/output operation failed
    #[error("IO error: {0}")]
    IoError(String),
    /// Network communication error
    #[error("Network error: {0}")]
    NetworkError(String),
    /// Command execution timed out
    #[error("Command timeout")]
    Timeout,
    /// Requested resource not found
    #[error("Resource not found")]
    NotFound,
    /// Internal system error
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type for command operations
pub type CommandResult<T> = Result<T, CommandError>;

impl CommandError {
    /// Create a new UnknownCommand error
    pub fn unknown_command(command: impl Into<String>) -> Self {
        Self::UnknownCommand { command: command.into() }
    }

    /// Create a new InvalidArguments error  
    pub fn invalid_arguments(msg: impl Into<String>) -> Self {
        Self::InvalidArguments(msg.into())
    }

    /// Create a new InvalidSyntax error
    pub fn invalid_syntax(detail: impl Into<String>) -> Self {
        Self::InvalidSyntax { detail: detail.into() }
    }

    /// Create a new ExecutionFailed error
    pub fn execution_failed(msg: impl Into<String>) -> Self {
        Self::ExecutionFailed(msg.into())
    }

    /// Create a new ParseError
    pub fn parse_error(msg: impl Into<String>, input: impl Into<String>) -> Self {
        Self::ParseError { message: msg.into(), input: input.into() }
    }

    /// Create a new ConfigurationError
    pub fn configuration_error(detail: impl Into<String>) -> Self {
        Self::ConfigurationError { detail: detail.into() }
    }

    /// Create a new IoError
    pub fn io_error(msg: impl Into<String>) -> Self {
        Self::IoError(msg.into())
    }

    /// Create a new NetworkError
    pub fn network_error(msg: impl Into<String>) -> Self {
        Self::NetworkError(msg.into())
    }

    /// Create a new InternalError
    pub fn internal_error(msg: impl Into<String>) -> Self {
        Self::InternalError(msg.into())
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            CommandError::UnknownCommand { .. } => false,
            CommandError::InvalidArguments(_) => true,
            CommandError::InvalidSyntax { .. } => true,
            CommandError::ExecutionFailed(_) => false,
            CommandError::PermissionDenied => false,
            CommandError::ParseError { .. } => true,
            CommandError::ConfigurationError { .. } => true,
            CommandError::IoError(_) => true,
            CommandError::NetworkError(_) => true,
            CommandError::Timeout => true,
            CommandError::NotFound => false,
            CommandError::InternalError(_) => false,
            CommandError::CommandAlreadyExists { .. } => false,
            CommandError::AliasAlreadyExists { .. } => false,
        }
    }

    /// Get error category for logging/metrics
    pub fn category(&self) -> &'static str {
        match self {
            CommandError::UnknownCommand { .. } => "unknown_command",
            CommandError::InvalidArguments(_) => "invalid_arguments",
            CommandError::InvalidSyntax { .. } => "invalid_syntax",
            CommandError::ExecutionFailed(_) => "execution_failed",
            CommandError::PermissionDenied => "permission_denied",
            CommandError::ParseError { .. } => "parse_error", 
            CommandError::ConfigurationError { .. } => "configuration_error",
            CommandError::IoError(_) => "io_error",
            CommandError::NetworkError(_) => "network_error",
            CommandError::Timeout => "timeout",
            CommandError::NotFound => "not_found",
            CommandError::InternalError(_) => "internal_error",
            CommandError::CommandAlreadyExists { .. } => "command_already_exists",
            CommandError::AliasAlreadyExists { .. } => "alias_already_exists",
        }
    }
}