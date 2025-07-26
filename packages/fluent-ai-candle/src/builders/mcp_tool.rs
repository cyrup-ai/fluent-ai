use serde_json::Value;

use crate::domain::mcp_tool::Tool;

/// Builder for MCP Tool objects
pub struct McpToolBuilder {
    name: String,
    description: String,
    parameters: Value}

impl McpToolBuilder {
    /// Create a new McpToolBuilder
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            parameters: Value::Object(Default::default())}
    }

    /// Set the parameters schema
    pub fn parameters(mut self, parameters: Value) -> Self {
        self.parameters = parameters;
        self
    }

    /// Build the Tool object
    pub fn build(self) -> impl Tool {
        McpToolImpl {
            name: self.name,
            description: self.description,
            parameters: self.parameters}
    }
}

/// Implementation of Tool for McpToolBuilder
pub struct McpToolImpl {
    name: String,
    description: String,
    parameters: Value}

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

    fn execute(
        &self,
        _args: Value,
    ) -> fluent_ai_http3::async_task::AsyncStream<Result<Value, String>> {
        let (tx, stream) = fluent_ai_http3::async_task::AsyncStream::channel();
        tokio::spawn(async move {
            let result = Ok(Value::Null);
            let _ = tx.send(result);
        });
        stream
    }
}
