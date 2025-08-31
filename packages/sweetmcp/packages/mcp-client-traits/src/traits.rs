//! Core trait definitions for MCP client implementations
//!
//! This module defines the fundamental traits that all MCP client implementations
//! must provide. All traits use `sweet-mcp-type` structures exclusively.

use std::future::Future;
use std::pin::Pin;
use sweet_mcp_type::{Request, Response, JsonValue, ToolInfo, Implementation, ServerCapabilities};
use crate::errors::ClientError;

/// Core MCP client interface that all protocol implementations must provide
///
/// This trait defines the fundamental operations for interacting with MCP servers,
/// using `sweet-mcp-type` structures throughout for optimal performance.
pub trait McpClient: Send + Sync {
    /// Execute a tool call with the specified name and arguments
    ///
    /// # Arguments
    /// * `name` - The tool name to execute
    /// * `args` - Tool arguments as sweet-mcp-type JsonValue
    ///
    /// # Returns
    /// A `Response` from sweet-mcp-type containing the tool execution result
    ///
    /// # Example
    /// ```rust,no_run
    /// use sweet_mcp_type::mcp::JsonValue;
    /// use std::collections::HashMap;
    /// 
    /// # async fn example(client: impl mcp_client_traits::McpClient) -> Result<(), mcp_client_traits::ClientError> {
    /// let mut args = HashMap::new();
    /// args.insert("data".to_string(), JsonValue::from("Hello World"));
    /// args.insert("algorithm".to_string(), JsonValue::from("sha256"));
    /// 
    /// let response = client.call_tool("hash", JsonValue::Object(args)).await?;
    /// # Ok(())
    /// # }
    /// ```
    fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
    ) -> Pin<Box<dyn Future<Output = Result<Response, ClientError>> + Send + '_>>;

    /// List all available tools from the MCP server
    ///
    /// # Returns
    /// A vector of `ToolInfo` structures from sweet-mcp-type
    fn list_tools(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ToolInfo>, ClientError>> + Send + '_>>;

    /// Initialize the MCP session with server capability negotiation
    ///
    /// # Arguments
    /// * `client_capabilities` - Client capabilities as JsonValue
    /// * `client_info` - Client implementation information
    ///
    /// # Returns
    /// Server response containing negotiated capabilities
    fn initialize(
        &self,
        client_capabilities: JsonValue,
        client_info: Implementation,
    ) -> Pin<Box<dyn Future<Output = Result<Response, ClientError>> + Send + '_>>;

    /// Send a ping request to test server connectivity
    ///
    /// # Returns
    /// Response confirming server is reachable
    fn ping(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Response, ClientError>> + Send + '_>>;
}

/// Convenience trait providing high-level operations for common MCP tools
///
/// This trait builds on top of `McpClient` to provide type-safe, convenient
/// methods for frequently used tools like time and hash operations.
pub trait McpToolOperations: McpClient {
    /// Execute time tool operations
    ///
    /// # Arguments
    /// * `operation` - The time operation ("get_time_utc", "parse_time")
    /// * `time_string` - Optional time string for parse_time operation
    ///
    /// # Example
    /// ```rust,no_run
    /// # async fn example(client: impl mcp_client_traits::McpToolOperations) -> Result<(), mcp_client_traits::ClientError> {
    /// let response = client.time_tool("get_time_utc", None).await?;
    /// # Ok(())
    /// # }
    /// ```
    fn time_tool(
        &self,
        operation: &str,
        time_string: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<Response, ClientError>> + Send + '_>> {
        let mut args = std::collections::HashMap::new();
        args.insert("name".to_string(), JsonValue::from(operation));
        
        if let Some(time_str) = time_string {
            args.insert("time_string".to_string(), JsonValue::from(time_str));
        }

        self.call_tool("time", JsonValue::from(args))
    }

    /// Execute hash tool operations  
    ///
    /// # Arguments
    /// * `data` - The data to hash or encode
    /// * `algorithm` - The algorithm ("sha256", "sha512", "md5", "base64", etc.)
    ///
    /// # Example
    /// ```rust,no_run
    /// # async fn example(client: impl mcp_client_traits::McpToolOperations) -> Result<(), mcp_client_traits::ClientError> {
    /// let response = client.hash_tool("Hello World", "sha256").await?;
    /// # Ok(())
    /// # }
    /// ```
    fn hash_tool(
        &self,
        data: &str,
        algorithm: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Response, ClientError>> + Send + '_>> {
        let mut args = std::collections::HashMap::new();
        args.insert("data".to_string(), JsonValue::from(data));
        args.insert("algorithm".to_string(), JsonValue::from(algorithm));

        self.call_tool("hash", JsonValue::from(args))
    }
}

/// Protocol-specific client implementation interface
///
/// This trait allows different protocols (GraphQL, JSON-RPC, Cap'n Proto) to 
/// implement their own request/response handling while maintaining a common interface.
pub trait ProtocolClient: Send + Sync {
    /// The protocol-specific request type
    type Request: Send;
    
    /// The protocol-specific response type  
    type Response: Send;

    /// Send a protocol-specific request and receive a response
    ///
    /// # Arguments
    /// * `request` - The protocol-specific request structure
    ///
    /// # Returns
    /// Protocol-specific response that can be converted to MCP Response
    fn send(
        &self,
        request: Self::Request,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Response, ClientError>> + Send + '_>>;

    /// Convert a protocol-specific response to MCP Response
    ///
    /// # Arguments
    /// * `response` - Protocol-specific response
    ///
    /// # Returns
    /// Standardized MCP Response from sweet-mcp-type
    fn to_mcp_response(&self, response: Self::Response) -> Result<Response, ClientError>;

    /// Convert MCP Request to protocol-specific request
    ///
    /// # Arguments  
    /// * `request` - MCP Request from sweet-mcp-type
    ///
    /// # Returns
    /// Protocol-specific request structure
    fn from_mcp_request(&self, request: Request) -> Result<Self::Request, ClientError>;
}

/// Client capabilities and configuration interface
///
/// This trait handles MCP capability negotiation and client configuration.
pub trait ClientCapabilities {
    /// Get the client's supported capabilities
    ///
    /// # Returns
    /// JsonValue representing client capabilities for MCP negotiation
    fn get_capabilities(&self) -> JsonValue;

    /// Get client implementation information
    ///
    /// # Returns  
    /// Implementation details (name, version) for MCP handshake
    fn get_client_info(&self) -> Implementation;

    /// Update client configuration based on server capabilities
    ///
    /// # Arguments
    /// * `server_capabilities` - Server capabilities from initialization response
    fn configure_from_server(&mut self, server_capabilities: ServerCapabilities) -> Result<(), ClientError>;

    /// Check if a specific capability is supported
    ///
    /// # Arguments
    /// * `capability` - The capability name to check
    ///
    /// # Returns
    /// True if the capability is supported by both client and server
    fn supports_capability(&self, capability: &str) -> bool;
}

// Auto-implement McpToolOperations for any type that implements McpClient
impl<T: McpClient> McpToolOperations for T {}