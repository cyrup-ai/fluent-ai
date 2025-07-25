//! Command execution errors with minimal allocations
//!
//! Provides comprehensive error handling for command processing with zero-allocation
//! error propagation patterns and efficient error reporting.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Command execution errors with minimal allocations
///
/// All errors use owned strings to avoid lifetime issues while maintaining
/// efficient memory usage through strategic allocation patterns.
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
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

impl CommandError {
    /// Create a new UnknownCommand error
    #[inline(always)]
    pub fn unknown_command(command: impl Into<String>) -> Self {
        Self::UnknownCommand {
            command: command.into(),
        }
    }

    /// Create a new InvalidArguments error
    #[inline(always)]
    pub fn invalid_arguments(detail: impl Into<String>) -> Self {
        Self::InvalidArguments(detail.into())
    }

    /// Create a new InvalidSyntax error
    #[inline(always)]
    pub fn invalid_syntax(detail: impl Into<String>) -> Self {
        Self::InvalidSyntax {
            detail: detail.into(),
        }
    }

    /// Create a new ExecutionFailed error
    #[inline(always)]
    pub fn execution_failed(detail: impl Into<String>) -> Self {
        Self::ExecutionFailed(detail.into())
    }

    /// Create a new ParseError
    #[inline(always)]
    pub fn parse_error(detail: impl Into<String>) -> Self {
        Self::ParseError(detail.into())
    }

    /// Create a new ConfigurationError
    #[inline(always)]
    pub fn configuration_error(detail: impl Into<String>) -> Self {
        Self::ConfigurationError {
            detail: detail.into(),
        }
    }

    /// Create a new IoError
    #[inline(always)]
    pub fn io_error(detail: impl Into<String>) -> Self {
        Self::IoError(detail.into())
    }

    /// Create a new NetworkError
    #[inline(always)]
    pub fn network_error(detail: impl Into<String>) -> Self {
        Self::NetworkError(detail.into())
    }

    /// Create a new InternalError
    #[inline(always)]
    pub fn internal_error(detail: impl Into<String>) -> Self {
        Self::InternalError(detail.into())
    }

    /// Check if error is recoverable
    #[inline(always)]
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::Timeout
                | Self::NetworkError(_)
                | Self::IoError(_)
        )
    }

    /// Check if error is a user error
    #[inline(always)]
    pub fn is_user_error(&self) -> bool {
        matches!(
            self,
            Self::UnknownCommand { .. }
                | Self::InvalidArguments(_)
                | Self::InvalidSyntax { .. }
                | Self::ParseError(_)
        )
    }

    /// Check if error is a system error
    #[inline(always)]
    pub fn is_system_error(&self) -> bool {
        matches!(
            self,
            Self::InternalError(_)
                | Self::ConfigurationError { .. }
                | Self::PermissionDenied
                | Self::NotFound
        )
    }

    /// Get error category as string
    #[inline(always)]
    pub fn category(&self) -> &'static str {
        match self {
            Self::UnknownCommand { .. } => "unknown_command",
            Self::InvalidArguments(_) => "invalid_arguments", 
            Self::InvalidSyntax { .. } => "invalid_syntax",
            Self::ExecutionFailed(_) => "execution_failed",
            Self::PermissionDenied => "permission_denied",
            Self::ParseError(_) => "parse_error",
            Self::ConfigurationError { .. } => "configuration_error",
            Self::IoError(_) => "io_error", 
            Self::NetworkError(_) => "network_error",
            Self::Timeout => "timeout",
            Self::NotFound => "not_found",
            Self::InternalError(_) => "internal_error",
        }
    }
}

/// Result type for command operations
///
/// Provides ergonomic error handling for all command-related operations
/// with efficient Result propagation patterns.
pub type CommandResult<T> = Result<T, CommandError>;

/// Extension trait for converting standard errors to CommandError
pub trait ToCommandError {
    /// Convert to CommandError
    fn to_command_error(self) -> CommandError;
}

impl ToCommandError for std::io::Error {
    fn to_command_error(self) -> CommandError {
        CommandError::io_error(self.to_string())
    }
}

impl ToCommandError for serde_json::Error {
    fn to_command_error(self) -> CommandError {
        CommandError::parse_error(self.to_string())
    }
}

impl From<std::io::Error> for CommandError {
    fn from(err: std::io::Error) -> Self {
        err.to_command_error()
    }
}

impl From<serde_json::Error> for CommandError {
    fn from(err: serde_json::Error) -> Self {
        err.to_command_error()
    }
}