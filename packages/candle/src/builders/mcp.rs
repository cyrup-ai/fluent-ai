//! MCP client builder implementation moved from domain
//! Builders are behavioral/construction logic, separate from core domain models

use std::sync::Arc;

use crate::domain::{tool::{CandleClient as Client, CandleMcpError as McpError, CandleTool as Tool, CandleTransport as Transport, types::CandleMcpToolType as McpClient}, AsyncTask};
use fluent_ai_async::AsyncStream;
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
    pub fn execute(self, args: Value) -> AsyncStream<Value> {
        let tool = self.register();
        let client = tool.client.clone();
        let name = tool.definition.name.clone();

        AsyncStream::with_channel(|stream_sender| {
            std::thread::spawn(move || {
                let mut call_stream = client.call_tool(&name, args);
                if let Some(result) = call_stream.try_next() {
                    let final_result = match result {
                        Ok(value) => value,
                        Err(McpError::ToolNotFound) => Value::String(format!("Tool '{}' not found", name)),
                        Err(McpError::ExecutionFailed(msg)) => {
                            Value::String(format!("Execution failed: {}", msg))
                        }
                        Err(McpError::Timeout) => Value::String("Request timeout".to_string()),
                        Err(McpError::InvalidResponse) => {
                            Value::String("Invalid response from server".to_string())
                        }
                        Err(McpError::TransportClosed) => {
                            Value::String("Transport connection closed".to_string())
                        }
                        Err(McpError::SerializationFailed) => {
                            Value::String("Serialization failed".to_string())
                        }
                    };
                    let _ = stream_sender.send(final_result);
                }
            });
        })
    }
}
