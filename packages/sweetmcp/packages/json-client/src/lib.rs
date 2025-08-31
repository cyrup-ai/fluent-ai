//! SweetMCP JSON-RPC 2.0 Client Library
//!
//! This library provides a direct JSON-RPC 2.0 client implementation for SweetMCP
//! using `sweet-mcp-type` exclusively for maximum performance and zero-allocation
//! serialization with simd-json.
//!
//! # Example
//!
//! ```rust,no_run
//! use sweetmcp_json_client::JsonClient;
//! use mcp_client_traits::{McpClient, RequestBuilder};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = JsonClient::new("https://localhost:8443")?;
//!     
//!     // Use fluent API to call time tool
//!     let response = client
//!         .tool("time")
//!         .with_arg("name", "get_time_utc")
//!         .execute()
//!         .await?;
//!         
//!     println!("Time result: {:?}", response);
//!     Ok(())
//! }
//! ```

use std::pin::Pin;
use std::future::Future;
use std::collections::HashMap;

use reqwest::Client;
use tracing::{info, debug};
use uuid::Uuid;
use anyhow::{Context, Result};

// Import sweet-mcp-type and client traits
use sweet_mcp_type::{Request, Response, RequestId, JsonValue, Message, Implementation};
use mcp_client_traits::{
    McpClient, ProtocolClient, ClientError, ToolInfo,
};

use value_trait::prelude::*;

/// Direct JSON-RPC 2.0 client for SweetMCP protocol
///
/// This client communicates directly with SweetMCP servers using JSON-RPC 2.0
/// protocol with `sweet-mcp-type` structures for optimal performance.
#[derive(Debug, Clone)]
pub struct JsonClient {
    /// Base URL of the SweetMCP server
    base_url: String,
    /// HTTP client for making requests
    http_client: Client,
    /// Default timeout for requests (in milliseconds)
    default_timeout_ms: u64,
}

impl JsonClient {
    /// Create a new JSON client
    ///
    /// # Arguments
    /// * `base_url` - The base URL of the SweetMCP server (e.g., "https://localhost:8443")
    ///
    /// # Returns
    /// A new `JsonClient` instance or an error if the URL is invalid
    pub fn new(base_url: &str) -> Result<Self> {
        let url = reqwest::Url::parse(base_url)
            .with_context(|| format!("Invalid base URL: {}", base_url))?;
        
        info!("Creating JSON client for SweetMCP server at {}", url);
        
        Ok(Self {
            base_url: base_url.to_string(),
            http_client: Client::new(),
            default_timeout_ms: 30000, // 30 seconds default
        })
    }

    /// Set default timeout for requests
    ///
    /// # Arguments
    /// * `timeout_ms` - Timeout in milliseconds
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.default_timeout_ms = timeout_ms;
        self
    }

    /// Send a JSON-RPC 2.0 request to the SweetMCP server
    ///
    /// # Arguments
    /// * `method` - The JSON-RPC method name
    /// * `params` - The method parameters as JsonValue
    /// * `request_id` - Optional request ID (generates UUID if None)
    ///
    /// # Returns
    /// The MCP Response or an error
    async fn send_request(
        &self,
        method: &str,
        params: JsonValue,
        request_id: Option<RequestId>,
    ) -> Result<Response, ClientError> {
        let id = request_id.unwrap_or_else(|| RequestId::Str(Uuid::new_v4().to_string()));
        
        // Build JSON-RPC 2.0 request using sweet-mcp-type
        let request = Request {
            id: id.clone(),
            method: method.to_string(),
            params,
            meta: None,
        };

        debug!("Sending JSON-RPC request: method={}, id={:?}", method, id);

        // Serialize request using sweet-mcp-type JSON serialization
        let request_body = self.serialize_request(&request)
            .map_err(|e| ClientError::RequestBuild(
                format!("Failed to serialize request: {}", e)
            ))?;

        // Send HTTP request
        let response = self
            .http_client
            .post(&format!("{}/mcp", self.base_url))
            .header("Content-Type", "application/json")
            .body(request_body)
            .timeout(std::time::Duration::from_millis(self.default_timeout_ms))
            .send()
            .await
            .map_err(ClientError::Transport)?;

        // Check HTTP status
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Failed to read error response".to_string());
            return Err(ClientError::RequestBuild(
                format!("Server returned HTTP {}: {}", status, error_text)
            ));
        }

        // Parse response
        let response_bytes = response.bytes().await
            .map_err(ClientError::Transport)?;

        self.parse_response(&response_bytes)
    }

    /// Serialize a Request to JSON bytes using sweet-mcp-type
    fn serialize_request(&self, request: &Request) -> Result<Vec<u8>, anyhow::Error> {
        // Use sweet-mcp-type's Message JSON serialization
        let message = Message::Req(request.clone());
        let json_string = message.to_json();
        Ok(json_string.into_bytes())
    }

    /// Parse response bytes into sweet-mcp-type Response
    fn parse_response(&self, bytes: &[u8]) -> Result<Response, ClientError> {
        // Parse using sweet-mcp-type Message parsing
        let json_str = std::str::from_utf8(bytes)
            .map_err(|e| ClientError::response_parse(
                format!("Invalid UTF-8 in response: {}", e),
                "JSON-RPC response encoding",
            ))?;

        let message = Message::from_json(json_str)
            .map_err(|e| ClientError::response_parse(
                format!("Invalid JSON-RPC response: {:?}", e),
                "JSON-RPC response parsing",
            ))?;

        match message {
            Message::Res(response) => Ok(response),
            _ => Err(ClientError::response_parse(
                "Expected response, got request or notification".to_string(),
                "JSON-RPC message type",
            )),
        }
    }
}

// Implement core MCP client trait
impl McpClient for JsonClient {
    fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
    ) -> Pin<Box<dyn Future<Output = Result<Response, ClientError>> + Send + '_>> {
        let name = name.to_string();
        Box::pin(async move {
            let mut params = HashMap::new();
            params.insert("name".to_string(), JsonValue::from(name));
            params.insert("arguments".to_string(), args);
            
            self.send_request("tools/call", JsonValue::from(params), None).await
        })
    }

    fn list_tools(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ToolInfo>, ClientError>> + Send + '_>> {
        Box::pin(async move {
            let response = self.send_request("tools/list", JsonValue::from(HashMap::<String, JsonValue>::new()), None).await?;
            
            // Extract tools from response
            if let Some(result) = response.result {
                if let Some(tools_array) = result.get("tools").and_then(|t| t.as_array()) {
                    let mut tools = Vec::new();
                    for tool_value in tools_array {
                        if let (Some(name), Some(input_schema)) = (
                            tool_value.get("name").and_then(|n| n.as_str()),
                            tool_value.get("inputSchema")
                        ) {
                            tools.push(ToolInfo {
                                name: name.to_string(),
                                description: tool_value.get("description")
                                    .and_then(|d| d.as_str())
                                    .map(|s| s.to_string()),
                                input_schema: input_schema.clone(),
                            });
                        }
                    }
                    return Ok(tools);
                }
            }
            
            Err(ClientError::response_parse(
                "Invalid tools/list response format".to_string(),
                "tools list extraction",
            ))
        })
    }

    fn initialize(
        &self,
        client_capabilities: JsonValue,
        client_info: Implementation,
    ) -> Pin<Box<dyn Future<Output = Result<Response, ClientError>> + Send + '_>> {
        Box::pin(async move {
            let mut params = HashMap::new();
            params.insert("capabilities".to_string(), client_capabilities);
            params.insert("clientInfo".to_string(), JsonValue::from([
                ("name", JsonValue::from(client_info.name)),
                ("version", JsonValue::from(client_info.version)),
            ].iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<HashMap<String, JsonValue>>()));
            
            self.send_request("initialize", JsonValue::from(params), None).await
        })
    }

    fn ping(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Response, ClientError>> + Send + '_>> {
        Box::pin(async move {
            self.send_request("ping", JsonValue::from(HashMap::<String, JsonValue>::new()), None).await
        })
    }
}

// Tool operations are automatically implemented via blanket impl

// Implement protocol client trait
impl ProtocolClient for JsonClient {
    type Request = Request;
    type Response = Response;

    fn send(
        &self,
        request: Self::Request,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Response, ClientError>> + Send + '_>> {
        Box::pin(async move {
            self.send_request(&request.method, request.params, Some(request.id)).await
        })
    }

    fn to_mcp_response(&self, response: Self::Response) -> Result<Response, ClientError> {
        // Already an MCP Response
        Ok(response)
    }

    fn from_mcp_request(&self, request: Request) -> Result<Self::Request, ClientError> {
        // Already an MCP Request
        Ok(request)
    }
}

// Request builder is automatically implemented via blanket impl

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;
    
    #[test]
    fn test_client_creation() {
        let client = JsonClient::new("https://localhost:8443").unwrap();
        assert_eq!(client.protocol_name(), "JSON-RPC 2.0");
        assert_eq!(client.server_url(), "https://localhost:8443");
        assert!(client.is_connected());
    }

    #[test]
    fn test_invalid_url() {
        let result = JsonClient::new("not-a-url");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_request_serialization() {
        let client = JsonClient::new("https://localhost:8443").unwrap();
        
        let request = Request {
            id: RequestId::Str("test-123".to_string()),
            method: "tools/call".to_string(),
            params: JsonValue::from([("name", "time"), ("arguments", "{}")]
                .iter()
                .map(|(k, v)| (k.to_string(), JsonValue::from(*v)))
                .collect::<HashMap<String, JsonValue>>()),
            meta: None,
        };

        let serialized = client.serialize_request(&request).unwrap();
        assert!(!serialized.is_empty());
    }
}