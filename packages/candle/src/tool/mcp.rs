//! MCP (Model Context Protocol) Client and Transport
//!
//! This module provides MCP client functionality including:
//! - JSON-RPC transport layer (StdioTransport)
//! - MCP client for tool execution
//! - Error handling and response management

use hashbrown::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{RwLock, mpsc};

// Removed unused imports AsyncTask and spawn_async

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    method: String,
    params: Value,
    id: u64}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    result: Option<Value>,
    error: Option<JsonRpcError>,
    id: u64}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    data: Option<Value>}

/// Error types for MCP (Model Context Protocol) operations.
///
/// This enum represents all possible errors that can occur during MCP tool execution,
/// transport operations, and protocol communication.
#[derive(Debug)]
pub enum McpError {
    /// The transport connection has been closed or is unavailable.
    TransportClosed,
    /// Failed to serialize or deserialize MCP protocol messages.
    SerializationFailed,
    /// The requested tool was not found in the MCP server.
    ToolNotFound,
    /// Tool execution failed with the provided error message.
    ExecutionFailed(String),
    /// Operation timed out waiting for response.
    Timeout,
    /// Received an invalid or malformed response from the MCP server.
    InvalidResponse}

/// Transport layer abstraction for MCP (Model Context Protocol) communication.
///
/// This trait defines the interface for sending and receiving data over various transport
/// mechanisms (stdio, TCP, WebSocket, etc.) used by MCP servers and clients.
///
/// Implementations must be thread-safe and support async operations without blocking.
pub trait Transport: Send + Sync + 'static {
    /// Send data to the transport endpoint.
    ///
    /// # Arguments
    ///
    /// * `data` - The raw bytes to send over the transport
    ///
    /// # Returns
    ///
    /// A future that resolves to `Ok(())` on successful send, or `McpError` on failure.
    fn send(&self, data: &[u8]) -> impl std::future::Future<Output = Result<(), McpError>> + Send;

    /// Receive data from the transport endpoint.
    ///
    /// # Returns
    ///
    /// A future that resolves to the received bytes on success, or `McpError` on failure.
    /// May block until data is available or an error occurs.
    fn receive(&self) -> impl std::future::Future<Output = Result<Vec<u8>, McpError>> + Send;
}

/// Standard input/output transport implementation for MCP communication.
///
/// This transport uses stdin/stdout for bidirectional communication with MCP servers,
/// which is the most common transport method for MCP tools. It uses async channels
/// to handle the communication without blocking.
pub struct StdioTransport {
    stdin_tx: mpsc::UnboundedSender<Vec<u8>>,
    stdout_rx: Arc<RwLock<mpsc::UnboundedReceiver<Vec<u8>>>>}

impl StdioTransport {
    /// Create a new StdioTransport instance.
    ///
    /// This method sets up bidirectional communication channels using stdin/stdout
    /// and spawns async tasks to handle the I/O operations. The transport is ready
    /// to use immediately after creation.
    ///
    /// # Returns
    ///
    /// A new `StdioTransport` instance ready for MCP communication.
    #[inline]
    pub fn new() -> Self {
        let (stdin_tx, mut stdin_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (stdout_tx, stdout_rx) = mpsc::unbounded_channel::<Vec<u8>>();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let mut stdout = tokio::io::stdout();

            while let Some(mut data) = stdin_rx.recv().await {
                data.push(b'\n');
                if stdout.write_all(&data).await.is_err() {
                    break;
                }
                if stdout.flush().await.is_err() {
                    break;
                }
            }
        });

        tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let stdin = tokio::io::stdin();
            let mut reader = BufReader::new(stdin);
            let mut line_buffer = String::with_capacity(8192);

            loop {
                line_buffer.clear();
                match reader.read_line(&mut line_buffer).await {
                    Ok(0) => break,
                    Ok(_) => {
                        let trimmed = line_buffer.trim_end();
                        if !trimmed.is_empty() {
                            if stdout_tx.send(trimmed.as_bytes().to_vec()).is_err() {
                                break;
                            }
                        }
                    }
                    Err(_) => break}
            }
        });

        Self {
            stdin_tx,
            stdout_rx: Arc::new(RwLock::new(stdout_rx))}
    }
}

impl Transport for StdioTransport {
    #[inline]
    async fn send(&self, data: &[u8]) -> Result<(), McpError> {
        self.stdin_tx
            .send(data.to_vec())
            .map_err(|_| McpError::TransportClosed)
    }

    #[inline]
    async fn receive(&self) -> Result<Vec<u8>, McpError> {
        let mut rx = self.stdout_rx.write().await;
        rx.recv().await.ok_or(McpError::TransportClosed)
    }
}

/// MCP client for communicating with MCP servers over various transports.
///
/// This client handles JSON-RPC communication with MCP servers, including request/response
/// matching, timeout handling, and response caching. It supports any transport that
/// implements the `Transport` trait.
///
/// # Type Parameters
///
/// * `T` - The transport implementation to use for communication
pub struct Client<T: Transport> {
    transport: Arc<T>,
    request_id: AtomicU64,
    response_cache: Arc<RwLock<HashMap<u64, Value>>>,
    request_timeout: Duration}

impl<T: Transport> Client<T> {
    /// Create a new MCP client with the specified transport.
    ///
    /// # Arguments
    ///
    /// * `transport` - The transport implementation to use for communication
    ///
    /// # Returns
    ///
    /// A new `Client` instance ready to communicate with MCP servers.
    #[inline]
    pub fn new(transport: T) -> Self {
        Self {
            transport: Arc::new(transport),
            request_id: AtomicU64::new(1),
            response_cache: Arc::new(RwLock::new(HashMap::with_capacity(256))),
            request_timeout: Duration::from_secs(30)}
    }

    /// Call a tool on the MCP server with the specified arguments.
    ///
    /// This method sends a JSON-RPC request to execute a tool and waits for the response.
    /// It handles request/response matching and timeout management automatically.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the tool to execute
    /// * `args` - JSON arguments to pass to the tool
    ///
    /// # Returns
    ///
    /// Returns the tool's result as a JSON `Value` on success, or `McpError` on failure.
    ///
    /// # Errors
    ///
    /// * `McpError::Timeout` - If the request times out
    /// * `McpError::ExecutionFailed` - If the tool execution fails
    /// * `McpError::SerializationFailed` - If JSON serialization/deserialization fails
    /// * `McpError::TransportClosed` - If the transport connection is closed
    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value, McpError> {
        let id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let start_time = Instant::now();

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            method: "tools/call".to_string(),
            params: serde_json::json!({
                "name": name,
                "arguments": args
            }),
            id};

        let mut buffer = Vec::with_capacity(1024);
        serde_json::to_writer(&mut buffer, &request).map_err(|_| McpError::SerializationFailed)?;

        self.transport.send(&buffer).await?;

        loop {
            if start_time.elapsed() > self.request_timeout {
                return Err(McpError::Timeout);
            }

            let response_data = self.transport.receive().await?;

            let response: JsonRpcResponse = serde_json::from_slice(&response_data)
                .map_err(|_| McpError::SerializationFailed)?;

            if response.id == id {
                if let Some(error) = response.error {
                    return Err(McpError::ExecutionFailed(error.message));
                }

                return response.result.ok_or(McpError::InvalidResponse);
            }

            {
                let mut cache = self.response_cache.write().await;
                if let Some(result) = response.result {
                    cache.insert(response.id, result);
                }
            }
        }
    }

    /// List all available tools from the MCP server.
    ///
    /// This method queries the MCP server for its available tools and returns
    /// a list of tool definitions including their names, descriptions, and schemas.
    ///
    /// # Returns
    ///
    /// Returns a vector of `Tool` definitions on success, or `McpError` on failure.
    /// An empty vector is returned if no tools are available.
    ///
    /// # Errors
    ///
    /// * `McpError::Timeout` - If the request times out
    /// * `McpError::SerializationFailed` - If JSON parsing fails
    /// * `McpError::TransportClosed` - If the transport connection is closed
    #[inline]
    pub async fn list_tools(&self) -> Result<Vec<super::types::Tool>, McpError> {
        let result = self.call_tool_internal("tools/list", Value::Null).await?;

        if let Value::Object(obj) = result {
            if let Some(Value::Array(tools)) = obj.get("tools") {
            let mut parsed_tools = Vec::with_capacity(tools.len());
            for tool in tools {
                if let Ok(parsed) = serde_json::from_value::<super::types::Tool>(tool.clone()) {
                    parsed_tools.push(parsed);
                }
            }
            return Ok(parsed_tools);
            }
        }
        Ok(Vec::new())
    }

    #[inline]
    async fn call_tool_internal(&self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let start_time = Instant::now();

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
            id};

        let mut buffer = Vec::with_capacity(512);
        serde_json::to_writer(&mut buffer, &request).map_err(|_| McpError::SerializationFailed)?;

        self.transport.send(&buffer).await?;

        loop {
            if start_time.elapsed() > self.request_timeout {
                return Err(McpError::Timeout);
            }

            let response_data = self.transport.receive().await?;

            let response: JsonRpcResponse = serde_json::from_slice(&response_data)
                .map_err(|_| McpError::SerializationFailed)?;

            if response.id == id {
                if let Some(error) = response.error {
                    return Err(McpError::ExecutionFailed(error.message));
                }

                return response.result.ok_or(McpError::InvalidResponse);
            }
        }
    }
}
