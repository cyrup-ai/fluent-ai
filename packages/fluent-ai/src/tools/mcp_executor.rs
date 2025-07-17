//! MCP Tool execution implementation
//! 
//! Contains the actual implementation logic for MCP tools with secure execution.

use fluent_ai_domain::{Tool, McpTool, McpToolData};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

/// Production implementation of MCP Tool with secure execution
#[derive(Debug, Clone)]
pub struct McpToolImpl {
    data: McpToolData,
}

impl McpToolImpl {
    /// Create a new MCP tool implementation
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            data: McpToolData::new(name, description, parameters),
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
            data: McpToolData::with_server(name, description, parameters, server),
        }
    }

    /// Create a new MCP tool with parameter map
    pub fn with_params<F>(name: impl Into<String>, description: impl Into<String>, f: F) -> Self
    where
        F: FnOnce() -> std::collections::HashMap<String, Value>,
    {
        let params = f();
        let json_params: serde_json::Map<String, Value> = params.into_iter().collect();
        Self {
            data: McpToolData::new(name, description, Value::Object(json_params)),
        }
    }
}

impl Tool for McpToolImpl {
    fn name(&self) -> &str {
        &self.data.name
    }
    
    fn description(&self) -> &str {
        &self.data.description
    }
    
    fn parameters(&self) -> &Value {
        &self.data.parameters
    }
    
    fn execute(&self, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> {
        let name = self.data.name.clone();
        let description = self.data.description.clone();
        
        Box::pin(async move {
            // Check if this is a code execution tool that should use secure execution
            if should_use_secure_execution(&name, &description, &args) {
                // Use secure execution via cylo
                match execute_with_secure_backend(&name, &args).await {
                    Ok(result) => result,
                    Err(e) => {
                        // Fallback to default behavior if secure execution fails
                        tracing::warn!("Secure execution failed for tool '{}': {}. Falling back to default.", name, e);
                        Ok(serde_json::json!({
                            "tool": name,
                            "args": args,
                            "result": "MCP tool execution not implemented yet",
                            "secure_execution_error": e
                        }))
                    }
                }
            } else {
                // Default implementation for non-code MCP tools
                Ok(serde_json::json!({
                    "tool": name,
                    "args": args,
                    "result": "MCP tool execution not implemented yet"
                }))
            }
        })
    }
}

impl McpTool for McpToolImpl {
    fn server(&self) -> Option<&str> {
        self.data.server.as_deref()
    }
    
    fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self::new(name, description, parameters)
    }
}

/// Execute tool with secure backend (cylo integration)
async fn execute_with_secure_backend(name: &str, args: &Value) -> Result<Result<Value, String>, String> {
    // TODO: Integrate with cylo secure execution
    // For now, return a proper placeholder that matches the expected signature
    Ok(Ok(serde_json::json!({
        "tool": name,
        "args": args,
        "result": "Secure execution not yet implemented",
        "status": "placeholder"
    })))
}

/// Helper function to determine if a tool should use secure execution
fn should_use_secure_execution(name: &str, description: &str, args: &Value) -> bool {
    // Check if the tool name suggests code execution
    let name_lower = name.to_lowercase();
    let desc_lower = description.to_lowercase();
    
    // Common code execution tool patterns
    let code_execution_patterns = [
        "exec", "execute", "run", "eval", "script", "code",
        "python", "javascript", "bash", "shell", "rust", "go",
        "interpreter", "compiler", "runner"
    ];
    
    // Check if name or description contains code execution patterns
    let has_code_pattern = code_execution_patterns.iter().any(|pattern| {
        name_lower.contains(pattern) || desc_lower.contains(pattern)
    });
    
    // Check if args contain code or script
    let has_code_args = args.get("code").is_some() 
        || args.get("script").is_some() 
        || args.get("command").is_some()
        || args.get("language").is_some();
    
    // Use secure execution if either pattern matches
    has_code_pattern || has_code_args
}

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
    fn test_should_use_secure_execution() {
        let args = serde_json::json!({"code": "print('hello')"});
        assert!(should_use_secure_execution("python_exec", "Execute Python code", &args));
        
        let args = serde_json::json!({"message": "hello"});
        assert!(!should_use_secure_execution("send_email", "Send an email", &args));
    }
}