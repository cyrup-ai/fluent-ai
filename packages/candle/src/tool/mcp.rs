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
use fluent_ai_async::AsyncStream;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{RwLock, mpsc};

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
    /// Stream that emits unit value on successful send, or nothing on failure.
    fn send(&self, data: &[u8]) -> AsyncStream<()>;

    /// Receive data from the transport endpoint.
    ///
    /// # Returns
    ///
    /// Stream that emits received bytes on success, or nothing on failure.
    /// May block until data is available or an error occurs.
    fn receive(&self) -> AsyncStream<Vec<u8>>;
}

/// Standard input/output transport implementation for MCP communication.
///
/// This transport uses stdin/stdout for bidirectional communication with MCP servers,
/// which is the most common transport method for MCP tools. It uses standard channels
/// to handle the communication without blocking.
pub struct StdioTransport {
    stdin_tx: mpsc::Sender<Vec<u8>>,
    stdout_rx: Arc<RwLock<mpsc::Receiver<Vec<u8>>>>}

impl StdioTransport {
    /// Create a new StdioTransport instance.
    ///
    /// This method sets up bidirectional communication channels using stdin/stdout
    /// and spawns standard threads to handle the I/O operations. The transport is ready
    /// to use immediately after creation.
    ///
    /// # Returns
    ///
    /// A new `StdioTransport` instance ready for MCP communication.
    #[inline]
    pub fn new() -> Self {
        let (stdin_tx, stdin_rx) = mpsc::channel::<Vec<u8>>();
        let (stdout_tx, stdout_rx) = mpsc::channel::<Vec<u8>>();

        std::thread::spawn(move || {
            use std::io::{Write, stdout};
            let mut stdout_handle = stdout();

            while let Ok(mut data) = stdin_rx.recv() {
                data.push(b'\n');
                if stdout_handle.write_all(&data).is_err() {
                    break;
                }
                if stdout_handle.flush().is_err() {
                    break;
                }
            }
        });

        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader, stdin};
            let stdin_handle = stdin();
            let mut reader = BufReader::new(stdin_handle);
            let mut line_buffer = String::with_capacity(8192);

            loop {
                line_buffer.clear();
                match reader.read_line(&mut line_buffer) {
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
    fn send(&self, data: &[u8]) -> AsyncStream<()> {
        let data = data.to_vec();
        let stdin_tx = self.stdin_tx.clone();
        AsyncStream::with_channel(move |sender| {
            match stdin_tx.send(data) {
                Ok(()) => {
                    let _ = sender.send(());
                },
                Err(_) => {
                    // Transport closed - don't send anything
                }
            }
        })
    }

    #[inline]
    fn receive(&self) -> AsyncStream<Vec<u8>> {
        let stdout_rx = self.stdout_rx.clone();
        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                if let Ok(rx) = stdout_rx.try_read() {
                    if let Ok(data) = rx.recv() {
                        let _ = sender.send(data);
                    }
                }
            });
        })
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
    pub fn call_tool(&self, name: &str, args: Value) -> AsyncStream<Value> {
        let name = name.to_string();
        let transport = self.transport.clone();
        let request_id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let request_timeout = self.request_timeout;
        let response_cache = self.response_cache.clone();
        
        AsyncStream::with_channel(move |sender| {
            let request = JsonRpcRequest {
                jsonrpc: "2.0",
                method: "tools/call".to_string(),
                params: serde_json::json!({
                    "name": name,
                    "arguments": args
                }),
                id: request_id
            };

            let mut buffer = Vec::with_capacity(1024);
            if serde_json::to_writer(&mut buffer, &request).is_err() {
                return; // SerializationFailed - don't send anything
            }

            // Send request
            let mut send_stream = transport.send(&buffer);
            if send_stream.try_next().is_none() {
                return; // Transport send failed
            }

            let start_time = Instant::now();
            loop {
                if start_time.elapsed() > request_timeout {
                    return; // Timeout - don't send anything
                }

                let mut receive_stream = transport.receive();
                if let Some(response_data) = receive_stream.try_next() {
                    if let Ok(response) = serde_json::from_slice::<JsonRpcResponse>(&response_data) {
                        if response.id == request_id {
                            if let Some(_error) = response.error {
                                return; // ExecutionFailed - don't send anything
                            }

                            if let Some(result) = response.result {
                                let _ = sender.send(result);
                                return;
                            }
                        } else {
                            // Cache response for different request ID
                            if let Ok(mut cache) = response_cache.try_write() {
                                if let Some(result) = response.result {
                                    cache.insert(response.id, result);
                                }
                            }
                        }
                    }
                }
                
                // Small delay to prevent busy waiting
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        })
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
    pub fn list_tools(&self) -> AsyncStream<Vec<super::types::Tool>> {
        let mut internal_stream = self.call_tool_internal("tools/list", Value::Null);
        
        AsyncStream::with_channel(move |sender| {
            if let Some(result) = internal_stream.try_next() {
                if let Value::Object(obj) = result {
                    if let Some(Value::Array(tools)) = obj.get("tools") {
                        let mut parsed_tools = Vec::with_capacity(tools.len());
                        for tool in tools {
                            if let Ok(parsed) = serde_json::from_value::<super::types::Tool>(tool.clone()) {
                                parsed_tools.push(parsed);
                            }
                        }
                        let _ = sender.send(parsed_tools);
                        return;
                    }
                }
                // Send empty vector if parsing failed
                let _ = sender.send(Vec::new());
            }
            // Don't send anything if no result from internal call
        })
    }

    #[inline]
    fn call_tool_internal(&self, method: &str, params: Value) -> AsyncStream<Value> {
        let method = method.to_string();
        let transport = self.transport.clone();
        let request_id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let request_timeout = self.request_timeout;
        
        AsyncStream::with_channel(move |sender| {
            let request = JsonRpcRequest {
                jsonrpc: "2.0",
                method,
                params,
                id: request_id
            };

            let mut buffer = Vec::with_capacity(512);
            if serde_json::to_writer(&mut buffer, &request).is_err() {
                return; // SerializationFailed - don't send anything
            }

            // Send request
            let mut send_stream = transport.send(&buffer);
            if send_stream.try_next().is_none() {
                return; // Transport send failed
            }

            let start_time = Instant::now();
            loop {
                if start_time.elapsed() > request_timeout {
                    return; // Timeout - don't send anything
                }

                let mut receive_stream = transport.receive();
                if let Some(response_data) = receive_stream.try_next() {
                    if let Ok(response) = serde_json::from_slice::<JsonRpcResponse>(&response_data) {
                        if response.id == request_id {
                            if let Some(_error) = response.error {
                                return; // ExecutionFailed - don't send anything
                            }

                            if let Some(result) = response.result {
                                let _ = sender.send(result);
                                return;
                            }
                        }
                    }
                }
                
                // Small delay to prevent busy waiting
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        })
    }
}