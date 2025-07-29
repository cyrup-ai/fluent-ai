//! CandleTool trait definition - EXACT REPLICA of fluent-ai-domain Tool trait
//!
//! This trait provides the core tool interface exactly matching domain architecture.

use std::fmt;

use fluent_ai_async::AsyncStream;
use serde_json::Value;

/// Core tool trait - EXACT REPLICA of domain Tool trait (without Clone for dyn compatibility)
pub trait CandleTool: Send + Sync + fmt::Debug {
    /// Get the name of the tool
    fn name(&self) -> &str;

    /// Get the description of the tool
    fn description(&self) -> &str;

    /// Get the JSON schema for the tool's input parameters
    fn parameters(&self) -> &Value;

    /// Execute the tool with given arguments
    fn execute(&self, args: Value) -> AsyncStream<Value>;
}

/// MCP tool trait - EXACT REPLICA of domain McpTool trait
pub trait CandleMcpTool: CandleTool {
    /// Get the optional server identifier this tool belongs to
    fn server(&self) -> Option<&str>;

    /// Create a new MCP tool with the given name, description, and parameters
    fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self;
}