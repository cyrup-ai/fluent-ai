//! MCP Tool domain trait and implementation
//! 
//! Provides trait-based MCP tool management with builder pattern

use crate::async_task::{AsyncTask, NotResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::future::Future;
use std::pin::Pin;

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

/// Default implementation of the McpTool trait
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolImpl {
    /// Name of the tool
    name: String,
    /// Description of what the tool does
    description: String,
    /// JSON schema for the tool's input parameters
    parameters: Value,
    /// Optional server identifier this tool belongs to
    server: Option<String>,
}

impl Tool for McpToolImpl {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn parameters(&self) -> &Value {
        &self.parameters
    }
    
    fn execute(&self, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> {
        // Default implementation for MCP tools - actual execution would be handled by MCP server
        let name = self.name.clone();
        Box::pin(async move {
            // This is a placeholder - real MCP tools would delegate to the MCP server
            Ok(serde_json::json!({
                "tool": name,
                "args": args,
                "result": "MCP tool execution not implemented yet"
            }))
        })
    }
}

impl McpTool for McpToolImpl {
    fn server(&self) -> Option<&str> {
        self.server.as_deref()
    }
    
    fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            server: None,
        }
    }
}

impl McpToolImpl {
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

/// Builder for creating and configuring MCP tools
pub struct McpToolBuilder {
    name: Option<String>,
    description: Option<String>,
    parameters: Option<Value>,
    server: Option<String>,
    error_handler: Option<Box<dyn FnMut(String) + Send + 'static>>,
}

/// Builder with error handler - exposes terminal methods
pub struct McpToolBuilderWithHandler {
    name: Option<String>,
    description: Option<String>,
    parameters: Option<Value>,
    server: Option<String>,
    error_handler: Box<dyn FnMut(String) + Send + 'static>,
    result_handler: Option<Box<dyn FnOnce(McpToolImpl) -> McpToolImpl + Send + 'static>>,
    chunk_handler: Option<Box<dyn FnMut(McpToolImpl) -> McpToolImpl + Send + 'static>>,
}

impl McpToolBuilder {
    /// Create a new MCP tool builder
    pub fn new() -> Self {
        Self {
            name: None,
            description: None,
            parameters: None,
            server: None,
            error_handler: None,
        }
    }
    
    /// Set the tool name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    /// Set the tool description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    
    /// Set the tool parameters
    pub fn parameters(mut self, parameters: Value) -> Self {
        self.parameters = Some(parameters);
        self
    }
    
    /// Set the server identifier
    pub fn server(mut self, server: impl Into<String>) -> Self {
        self.server = Some(server.into());
        self
    }
    
    /// Add error handler to enable terminal methods
    pub fn on_error<F>(self, error_handler: F) -> McpToolBuilderWithHandler
    where
        F: FnMut(String) + Send + 'static,
    {
        McpToolBuilderWithHandler {
            name: self.name,
            description: self.description,
            parameters: self.parameters,
            server: self.server,
            error_handler: Box::new(error_handler),
            result_handler: None,
            chunk_handler: None,
        }
    }

    pub fn on_result<F>(self, handler: F) -> McpToolBuilderWithHandler
    where
        F: FnOnce(McpToolImpl) -> McpToolImpl + Send + 'static,
    {
        McpToolBuilderWithHandler {
            name: self.name,
            description: self.description,
            parameters: self.parameters,
            server: self.server,
            error_handler: Box::new(|e| eprintln!("McpTool error: {}", e)),
            result_handler: Some(Box::new(handler)),
            chunk_handler: None,
        }
    }

    pub fn on_chunk<F>(self, handler: F) -> McpToolBuilderWithHandler
    where
        F: FnMut(McpToolImpl) -> McpToolImpl + Send + 'static,
    {
        McpToolBuilderWithHandler {
            name: self.name,
            description: self.description,
            parameters: self.parameters,
            server: self.server,
            error_handler: Box::new(|e| eprintln!("McpTool chunk error: {}", e)),
            result_handler: None,
            chunk_handler: Some(Box::new(handler)),
        }
    }
}

impl McpToolBuilderWithHandler {
    /// Build and return an MCP tool implementation
    pub fn build(self) -> impl McpTool {
        let name = self.name.unwrap_or_else(|| "unnamed_tool".to_string());
        let description = self.description.unwrap_or_else(|| "No description".to_string());
        let parameters = self.parameters.unwrap_or_else(|| serde_json::json!({}));
        
        if let Some(server) = self.server {
            McpToolImpl::with_server(name, description, parameters, server)
        } else {
            McpToolImpl::new(name, description, parameters)
        }
    }
    
    /// Build asynchronously and return an MCP tool implementation
    pub fn build_async(self) -> AsyncTask<impl McpTool> {
        let name = self.name.unwrap_or_else(|| "unnamed_tool".to_string());
        let description = self.description.unwrap_or_else(|| "No description".to_string());
        let parameters = self.parameters.unwrap_or_else(|| serde_json::json!({}));
        let server = self.server;
        
        AsyncTask::from_future(async move {
            if let Some(server) = server {
                McpToolImpl::with_server(name, description, parameters, server)
            } else {
                McpToolImpl::new(name, description, parameters)
            }
        })
    }
}

impl Default for McpToolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// From trait implementation removed - conflicts with blanket impl From<T> for T

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_tool_impl_new() {
        let tool = McpToolImpl::new("test_tool", "A test tool", serde_json::json!({"type": "object"}));
        assert_eq!(tool.name(), "test_tool");
        assert_eq!(tool.description(), "A test tool");
        assert_eq!(tool.server(), None);
    }

    #[test]
    fn test_mcp_tool_with_server() {
        let tool = McpToolImpl::with_server(
            "server_tool", 
            "A server tool", 
            serde_json::json!({"type": "object"}),
            "test_server"
        );
        assert_eq!(tool.name(), "server_tool");
        assert_eq!(tool.server(), Some("test_server"));
    }
    
    #[test]
    fn test_mcp_tool_builder() {
        let tool = McpToolBuilder::new()
            .name("built_tool")
            .description("A built tool")
            .parameters(serde_json::json!({"type": "string"}))
            .on_error(|_| {})
            .build();
            
        assert_eq!(tool.name(), "built_tool");
        assert_eq!(tool.description(), "A built tool");
    }
}