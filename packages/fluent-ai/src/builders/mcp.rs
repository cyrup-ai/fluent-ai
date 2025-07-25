//! MCP client builder implementation moved from domain
//! Builders are behavioral/construction logic, separate from core domain models

use std::sync::Arc;

use fluent_ai_domain::{AsyncTask, Client, McpClient, McpError, Tool, Transport, spawn_async};
use serde_json::Value;

pub struct McpClientBuilder<T: Transport> {
    client: Arc<Client<T>>,
    name: Option<String>,
    description: Option<String>,
    input_schema: Option<Value>}

impl<T: Transport> McpClient<T> {
    #[inline]
    pub fn define(name: impl Into<String>, client: Client<T>) -> McpClientBuilder<T> {
        McpClientBuilder {
            client: Arc::new(client),
            name: Some(name.into()),
            description: None,
            input_schema: None}
    }
}

impl<T: Transport> McpClientBuilder<T> {
    #[inline]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    #[inline]
    pub fn input_schema(mut self, schema: Value) -> Self {
        self.input_schema = Some(schema);
        self
    }

    #[inline]
    pub fn parameters(mut self, schema: Value) -> Self {
        self.input_schema = Some(schema);
        self
    }

    #[inline]
    pub fn register(self) -> McpClient<T> {
        McpClient {
            definition: Tool {
                name: self.name.unwrap_or_else(|| "unnamed_tool".to_string()),
                description: self
                    .description
                    .unwrap_or_else(|| "No description provided".to_string()),
                input_schema: self
                    .input_schema
                    .unwrap_or(Value::Object(Default::default()))},
            client: self.client}
    }

    #[inline]
    pub fn execute(self, args: Value) -> AsyncTask<Value> {
        let tool = self.register();
        let client = tool.client.clone();
        let name = tool.definition.name.clone();

        spawn_async(async move {
            match client.call_tool(&name, args).await {
                Ok(result) => result,
                Err(McpError::ToolNotFound) => Value::String(format!("Tool '{}' not found", name)),
                Err(McpError::ExecutionFailed(msg)) => {
                    Value::String(format!("Execution failed: {}", msg))
                }
                Err(McpError::Transport(msg)) => Value::String(format!("Transport error: {}", msg)),
                Err(McpError::Protocol(msg)) => Value::String(format!("Protocol error: {}", msg)),
                Err(McpError::Timeout) => Value::String("Request timeout".to_string()),
                Err(McpError::InvalidResponse) => {
                    Value::String("Invalid response from server".to_string())
                }
                Err(McpError::ConnectionFailed(msg)) => {
                    Value::String(format!("Connection failed: {}", msg))
                }
                Err(McpError::Authentication(msg)) => {
                    Value::String(format!("Authentication failed: {}", msg))
                }
                Err(McpError::ResourceNotFound) => Value::String("Resource not found".to_string()),
                Err(McpError::Internal(msg)) => Value::String(format!("Internal error: {}", msg))}
        })
    }
}
