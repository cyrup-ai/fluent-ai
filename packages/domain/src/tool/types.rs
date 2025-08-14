//! Tool type definitions and data structures
//!
//! Contains data structures for MCP tools and tool definitions.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Basic MCP tool data structure - implementation is in fluent_ai
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolData {
    /// Name of the tool
    pub name: String,
    /// Description of what the tool does
    pub description: String,
    /// JSON schema for the tool's input parameters
    pub parameters: Value,
    /// Optional server identifier this tool belongs to
    pub server: Option<String>,
}

impl McpToolData {
    /// Create a new MCP tool data structure
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            server: None,
        }
    }

    /// Create a new MCP tool with server identifier
    pub fn with_server(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        server: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            server: Some(server.into()),
        }
    }
}

/// Tool definition from MCP protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// The unique name identifier for this tool
    pub name: String,
    /// Human-readable description of what this tool does
    pub description: String,
    /// JSON schema defining the expected input parameters for this tool
    pub input_schema: Value,
}
