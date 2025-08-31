//! SweetMCP GraphQL Client Library
//!
//! This library provides a GraphQL client implementation for SweetMCP that
//! generates GraphQL schemas dynamically based on available MCP tools and
//! provides a type-safe query interface.
//!
//! # Example
//!
//! ```rust,no_run
//! use sweetmcp_graphql_client::GraphQLClient;
//! use mcp_client_traits::{McpClient, RequestBuilder};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = GraphQLClient::new("https://localhost:8443").await?;
//!     
//!     // Execute GraphQL query for time tool
//!     let query = r#"
//!         mutation {
//!             timeResult: time_tool(operation: "get_time_utc") {
//!                 ... on TimeResult {
//!                     utc_time
//!                     formatted_time
//!                     timezone
//!                 }
//!             }
//!         }
//!     "#;
//!     
//!     let response = client.execute_query(query, None).await?;
//!     println!("GraphQL result: {}", response);
//!     Ok(())
//! }
//! ```

use std::pin::Pin;
use std::future::Future;
use std::collections::HashMap;


use async_graphql::{Schema, Request as GraphQLRequest, Variables, EmptySubscription};
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

// Import our GraphQL schema
pub mod schema;
use schema::{Query, Mutation, McpClientContext, create_schema};

/// GraphQL client for SweetMCP protocol
///
/// This client provides a GraphQL interface to MCP tools, automatically
/// generating schema based on available tools and providing type-safe queries.
#[derive(Clone)]
pub struct GraphQLClient {
    /// Base URL of the SweetMCP server
    base_url: String,
    /// HTTP client for making requests
    http_client: Client,
    /// Default timeout for requests (in milliseconds)
    default_timeout_ms: u64,
    /// GraphQL schema with MCP tool mappings
    schema: Schema<Query, Mutation, EmptySubscription>,
}

impl GraphQLClient {
    /// Create a new GraphQL client
    ///
    /// # Arguments
    /// * `base_url` - The base URL of the SweetMCP server
    ///
    /// # Returns
    /// A new `GraphQLClient` instance
    pub async fn new(base_url: &str) -> Result<Self> {
        let url = reqwest::Url::parse(base_url)
            .with_context(|| format!("Invalid base URL: {}", base_url))?;
        
        info!("Creating GraphQL client for SweetMCP server at {}", url);
        
        let client = Self {
            base_url: base_url.to_string(),
            http_client: Client::new(),
            default_timeout_ms: 30000, // 30 seconds default
            schema: create_schema(),
        };

        Ok(client)
    }

    /// Set default timeout for requests
    ///
    /// # Arguments
    /// * `timeout_ms` - Timeout in milliseconds
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.default_timeout_ms = timeout_ms;
        self
    }

    /// Execute a GraphQL query against the MCP server
    ///
    /// # Arguments
    /// * `query` - GraphQL query string
    /// * `variables` - Optional query variables
    ///
    /// # Returns
    /// JSON response from GraphQL execution
    pub async fn execute_query(
        &self,
        query: &str,
        variables: Option<Variables>,
    ) -> Result<String, ClientError> {
        let request = GraphQLRequest::new(query);
        let request = if let Some(vars) = variables {
            request.variables(vars)
        } else {
            request
        };

        // Execute with context data
        let response = self.schema
            .execute(request.data(McpClientContext::new(self.clone())))
            .await;

        // Convert GraphQL response to JSON
        Ok(serde_json::to_string_pretty(&response)
            .map_err(|e| ClientError::response_parse(
                format!("Failed to serialize GraphQL response: {}", e),
                "GraphQL response serialization",
            ))?)
    }

    /// Get the GraphQL schema SDL (Schema Definition Language)
    ///
    /// # Returns
    /// String containing the complete GraphQL schema
    pub fn get_schema_sdl(&self) -> String {
        self.schema.sdl()
    }

    /// Send a JSON-RPC 2.0 request to the SweetMCP server (internal method)
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

        debug!("Sending JSON-RPC request via GraphQL client: method={}, id={:?}", method, id);

        // Serialize request using sweet-mcp-type Message serialization
        let message = Message::Req(request);
        let request_body = message.to_json().into_bytes();

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
impl McpClient for GraphQLClient {
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

// Implement protocol client trait
impl ProtocolClient for GraphQLClient {
    type Request = GraphQLRequest;
    type Response = async_graphql::Response;

    fn send(
        &self,
        request: Self::Request,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Response, ClientError>> + Send + '_>> {
        Box::pin(async move {
            Ok(self.schema
                .execute(request.data(McpClientContext::new(self.clone())))
                .await)
        })
    }

    fn to_mcp_response(&self, response: Self::Response) -> Result<Response, ClientError> {
        // Convert GraphQL response to MCP Response
        // This is a simplified conversion - in practice, you might want more sophisticated mapping
        let result_json = serde_json::to_string(&response)
            .map_err(|e| ClientError::response_parse(
                format!("Failed to serialize GraphQL response: {}", e),
                "GraphQL to MCP conversion",
            ))?;

        let mut result_bytes = result_json.as_bytes().to_vec();
        let result_value = simd_json::to_owned_value(&mut result_bytes)
            .map_err(|e| ClientError::response_parse(
                format!("Failed to parse GraphQL response as JSON: {}", e),
                "GraphQL response parsing",
            ))?;

        Ok(Response {
            id: RequestId::Str(Uuid::new_v4().to_string()),
            result: Some(result_value),
            error: None,
        })
    }

    fn from_mcp_request(&self, request: Request) -> Result<Self::Request, ClientError> {
        // Convert MCP Request to GraphQL Request
        // This is a simplified conversion - you would typically need more sophisticated mapping
        let query = format!(
            "mutation {{ execute_tool(tool_name: \"{}\", args_json: \"{}\") {{ ... on GenericResult {{ success content }} }} }}",
            request.method,
            request.params.encode().replace('"', "\\\"")
        );
        
        Ok(GraphQLRequest::new(query))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;
    
    #[tokio::test]
    async fn test_client_creation() {
        let client = GraphQLClient::new("https://localhost:8443").await.unwrap();
        assert!(!client.get_schema_sdl().is_empty());
    }

    #[test]
    fn test_invalid_url() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(GraphQLClient::new("not-a-url"));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_schema_generation() {
        let client = GraphQLClient::new("https://localhost:8443").await.unwrap();
        let schema_sdl = client.get_schema_sdl();
        
        // Verify schema contains expected types
        assert!(schema_sdl.contains("type Query"));
        assert!(schema_sdl.contains("type Mutation"));
        assert!(schema_sdl.contains("TimeResult"));
        assert!(schema_sdl.contains("HashResult"));
    }
}