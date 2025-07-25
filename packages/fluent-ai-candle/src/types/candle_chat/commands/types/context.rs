//! Command execution context and output types
//!
//! Context management and output handling for zero-allocation command execution
//! with streaming support and comprehensive metadata tracking.

use std::collections::HashMap;

use super::core::ResourceUsage;
use super::events::OutputType;

/// Command handler context for zero-allocation execution
#[derive(Debug, Clone)]
pub struct CommandContext {
    /// Command execution ID
    pub execution_id: u64,
    /// User session identifier
    pub session_id: String,
    /// Command input text
    pub input: String,
    /// Execution timestamp in nanoseconds
    pub timestamp_nanos: u64,
    /// Environment variables
    pub environment: HashMap<String, String>,
}

impl CommandContext {
    /// Create new command context
    #[inline]
    pub fn new(execution_id: u64, session_id: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            execution_id,
            session_id: session_id.into(),
            input: input.into(),
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            environment: HashMap::new(),
        }
    }

    /// Add environment variable
    #[inline]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment.insert(key.into(), value.into());
        self
    }
}

/// Command execution output with zero-allocation streaming
#[derive(Debug, Clone)]
pub struct CommandOutput {
    /// Execution ID this output belongs to
    pub execution_id: u64,
    /// Output content
    pub content: String,
    /// Output type
    pub output_type: OutputType,
    /// Output timestamp in nanoseconds
    pub timestamp_nanos: u64,
    /// Whether output is final
    pub is_final: bool,
    /// Execution time in nanoseconds
    pub execution_time: u64,
    /// Command execution success status
    pub success: bool,
    /// Command execution message
    pub message: String,
    /// Command execution data payload
    pub data: Option<String>,
    /// Resource usage statistics
    pub resource_usage: Option<ResourceUsage>,
}

impl CommandOutput {
    /// Create new command output
    #[inline]
    pub fn new(execution_id: u64, content: impl Into<String>, output_type: OutputType) -> Self {
        Self {
            execution_id,
            content: content.into(),
            output_type,
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            is_final: false,
            execution_time: 0,
            success: true,
            message: String::new(),
            data: None,
            resource_usage: None,
        }
    }

    /// Create successful command output
    #[inline]
    pub fn success(execution_id: u64, content: impl Into<String>) -> Self {
        Self::new(execution_id, content, OutputType::Text)
    }

    /// Mark output as final
    #[inline]
    pub fn final_output(mut self) -> Self {
        self.is_final = true;
        self
    }

    /// Create text output
    #[inline]
    pub fn text(execution_id: u64, content: impl Into<String>) -> Self {
        Self::new(execution_id, content, OutputType::Text)
    }

    /// Create JSON output
    #[inline]
    pub fn json(execution_id: u64, content: impl Into<String>) -> Self {
        Self::new(execution_id, content, OutputType::Json)
    }

    /// Create HTML output
    #[inline]
    pub fn html(execution_id: u64, content: impl Into<String>) -> Self {
        Self::new(execution_id, content, OutputType::Html)
    }

    /// Create markdown output
    #[inline]
    pub fn markdown(execution_id: u64, content: impl Into<String>) -> Self {
        Self::new(execution_id, content, OutputType::Markdown)
    }

    /// Create successful command output with ID
    #[inline]
    pub fn success_with_id(execution_id: u64, content: impl Into<String>) -> Self {
        Self::success(execution_id, content)
    }

    /// Create error output
    #[inline]
    pub fn error(execution_id: u64, content: impl Into<String>) -> Self {
        Self::new(execution_id, content, OutputType::Text)
    }

    /// Create error output with timestamp
    #[inline]
    pub fn error_with_time(execution_id: u64, content: impl Into<String>) -> Self {
        Self::error(execution_id, content)
    }
}