//! MCP Tool domain traits - pure interfaces only
//!
//! Contains only trait definitions and basic data structures.
//! Implementation logic is in fluent_ai package.

use std::fmt;
use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

/// Core tool trait - base interface for all tools
pub trait Tool: Send + Sync + fmt::Debug + Clone {
    /// Get the name of the tool
    fn name(&self) -> &str;

    /// Get the description of the tool
    fn description(&self) -> &str;

    /// Get the JSON schema for the tool's input parameters
    fn parameters(&self) -> &Value;

    /// Execute the tool with given arguments
    fn execute(&self, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>;
}

/// MCP tool trait - extends Tool with MCP-specific functionality
pub trait McpTool: Tool {
    /// Get the optional server identifier this tool belongs to
    fn server(&self) -> Option<&str>;

    /// Create a new MCP tool with the given name, description, and parameters
    fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self;
}
