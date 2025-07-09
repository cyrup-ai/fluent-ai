//! MCP (Model Context Protocol) tool integration

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents a tool available through the Model Context Protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    /// Name of the tool
    pub name: String,
    /// Description of what the tool does
    pub description: String,
    /// JSON schema for the tool's input parameters
    pub parameters: Value,
    /// Optional server identifier this tool belongs to
    pub server: Option<String>,
}

impl McpTool {
    /// Create a new MCP tool
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

    /// Create a new MCP tool with parameter map
    pub fn with_params<F>(name: impl Into<String>, description: impl Into<String>, f: F) -> Self
    where
        F: FnOnce() -> hashbrown::HashMap<String, Value>,
    {
        let params = f();
        let json_params: serde_json::Map<String, Value> = params.into_iter().collect();
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Value::Object(json_params),
            server: None,
        }
    }
}
