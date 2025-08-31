//! Error types for MCP client operations
//!
//! This module defines comprehensive error handling for MCP client implementations,
//! with proper integration with sweet-mcp-type error structures.

use sweet_mcp_type::{McpError, JsonRpcError, JsonValue};

/// Comprehensive error type for MCP client operations
///
/// This enum covers all possible error conditions that can occur during
/// MCP client operations, with proper context preservation and error chaining.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// MCP protocol-level error from sweet-mcp-type
    #[error("MCP protocol error: {0}")]
    Protocol(#[from] McpError),

    /// JSON-RPC error response from server
    #[error("JSON-RPC error {code}: {message}")]
    JsonRpc {
        /// Error code from JSON-RPC response
        code: i64,
        /// Error message from JSON-RPC response  
        message: String,
        /// Optional additional error data
        data: Option<JsonValue>,
    },

    /// Network transport error
    #[error("Transport error: {0}")]
    Transport(#[from] reqwest::Error),

    /// Tool execution error with context
    #[error("Tool execution failed for '{tool}': {message}")]
    ToolExecution {
        /// The tool that failed
        tool: String,
        /// Error message describing the failure
        message: String,
        /// Optional error code
        code: Option<i64>,
        /// Optional additional context data
        context: Option<JsonValue>,
    },

    /// Request building/validation error
    #[error("Request building error: {0}")]
    RequestBuild(String),

    /// Response parsing error with context
    #[error("Response parsing error: {reason} - {context}")]
    ResponseParse {
        /// Reason for parsing failure
        reason: String,
        /// Additional context about what was being parsed
        context: String,
    },

    /// Invalid argument error with detailed information
    #[error("Invalid argument '{arg}': {reason}")]
    InvalidArgument {
        /// The argument name that was invalid
        arg: String,
        /// Detailed reason why the argument is invalid
        reason: String,
        /// Optional suggested valid values
        suggestions: Option<Vec<String>>,
    },

    /// Authentication/authorization error
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Server capability error
    #[error("Server capability error: {capability} - {reason}")]
    Capability {
        /// The capability that caused the error
        capability: String,
        /// Reason for the capability error
        reason: String,
    },

    /// Timeout error with operation context
    #[error("Operation timed out after {timeout_ms}ms: {operation}")]
    Timeout {
        /// The operation that timed out
        operation: String,
        /// Timeout duration in milliseconds
        timeout_ms: u64,
    },

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] anyhow::Error),
}

impl ClientError {
    /// Create a new tool execution error
    ///
    /// # Arguments
    /// * `tool` - The tool name that failed
    /// * `message` - Error message
    /// * `code` - Optional error code
    /// * `context` - Optional context data
    pub fn tool_execution(
        tool: impl Into<String>,
        message: impl Into<String>,
        code: Option<i64>,
        context: Option<JsonValue>,
    ) -> Self {
        Self::ToolExecution {
            tool: tool.into(),
            message: message.into(),
            code,
            context,
        }
    }

    /// Create a new invalid argument error
    ///
    /// # Arguments
    /// * `arg` - The argument name
    /// * `reason` - Why the argument is invalid
    /// * `suggestions` - Optional valid alternatives
    pub fn invalid_argument(
        arg: impl Into<String>,
        reason: impl Into<String>,
        suggestions: Option<Vec<String>>,
    ) -> Self {
        Self::InvalidArgument {
            arg: arg.into(),
            reason: reason.into(),
            suggestions,
        }
    }

    /// Create a new response parsing error
    ///
    /// # Arguments
    /// * `reason` - Why parsing failed
    /// * `context` - What was being parsed
    pub fn response_parse(
        reason: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        Self::ResponseParse {
            reason: reason.into(),
            context: context.into(),
        }
    }

    /// Create a new timeout error
    ///
    /// # Arguments
    /// * `operation` - The operation that timed out
    /// * `timeout_ms` - Timeout duration in milliseconds
    pub fn timeout(
        operation: impl Into<String>,
        timeout_ms: u64,
    ) -> Self {
        Self::Timeout {
            operation: operation.into(),
            timeout_ms,
        }
    }

    /// Create a capability error
    ///
    /// # Arguments
    /// * `capability` - The capability name
    /// * `reason` - Why the capability failed
    pub fn capability(
        capability: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::Capability {
            capability: capability.into(),
            reason: reason.into(),
        }
    }

    /// Check if this error is retryable
    ///
    /// # Returns
    /// True if the operation can be safely retried
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Transport(reqwest_err) => {
                // Retry on timeout, connection errors, but not on 4xx client errors
                reqwest_err.is_timeout() || reqwest_err.is_connect()
            }
            Self::Timeout { .. } => true,
            Self::JsonRpc { code, .. } => {
                // Retry on server errors (5xx equivalent), not client errors (4xx equivalent)
                *code >= -32099 && *code <= -32000 // JSON-RPC server error range
            }
            Self::Protocol(_) => false, // Protocol errors are usually not retryable
            Self::ToolExecution { code, .. } => {
                // Only retry if it's a server-side error
                code.map_or(false, |c| c >= 500)
            }
            _ => false,
        }
    }

    /// Get error severity level
    ///
    /// # Returns
    /// Error severity: "critical", "error", "warning", or "info"
    pub fn severity(&self) -> &'static str {
        match self {
            Self::Authentication(_) => "critical",
            Self::Protocol(_) => "critical", 
            Self::ToolExecution { .. } => "error",
            Self::JsonRpc { code, .. } => {
                if *code >= -32099 && *code <= -32000 {
                    "error" // Server error
                } else {
                    "warning" // Client error
                }
            }
            Self::Transport(_) => "error",
            Self::Timeout { .. } => "warning",
            Self::ResponseParse { .. } => "error",
            Self::InvalidArgument { .. } => "warning",
            Self::Capability { .. } => "warning",
            Self::Configuration(_) => "error",
            Self::RequestBuild(_) => "warning",
            Self::Serialization(_) => "error",
        }
    }
}

impl From<JsonRpcError> for ClientError {
    fn from(error: JsonRpcError) -> Self {
        Self::JsonRpc {
            code: error.code,
            message: error.message,
            data: error.data,
        }
    }
}

/// Result type alias for MCP client operations
pub type ClientResult<T> = Result<T, ClientError>;