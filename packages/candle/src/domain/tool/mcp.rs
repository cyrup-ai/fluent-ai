//! MCP (Model Context Protocol) Client and Transport
//!
//! This module provides MCP client functionality including:
//! - JSON-RPC transport layer (StdioTransport)
//! - MCP client for tool execution
//! - Error handling and response management

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use crossbeam_channel;
use fluent_ai_async::AsyncStream;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::domain::tool::CandleMcpToolData;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    method: String,
    params: Value,
    id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    result: Option<Value>,
    error: Option<JsonRpcError>,
    id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    data: Option<Value>,
}

/// Error types for MCP (Model Context Protocol) operations.
///
/// This enum represents all possible errors that can occur during MCP tool execution,
/// transport operations, and protocol communication.
#[derive(Debug)]
pub enum CandleMcpError {
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
    InvalidResponse,
}

// Type alias for ergonomics within the domain module
type McpError = CandleMcpError;

/// Transport layer abstraction for MCP (Model Context Protocol) communication.
///
/// This trait defines the interface for sending and receiving data over various transport
/// mechanisms (stdio, TCP, WebSocket, etc.) used by MCP servers and clients.
///
/// Implementations must be thread-safe and support async operations without blocking.
pub trait CandleTransport: Send + Sync + 'static {
    /// Send data to the transport endpoint.
    ///
    /// # Arguments
    ///
    /// * `data` - The raw bytes to send over the transport
    ///
    /// # Returns
    ///
    /// Stream that emits Result on send attempt.
    fn send(&self, data: &[u8]) -> AsyncStream<Result<(), CandleMcpError>>;

    /// Receive data from the transport endpoint.
    ///
    /// # Returns
    ///
    /// Stream that emits received bytes on success, or error on failure.
    /// May block until data is available or an error occurs.
    fn receive(&self) -> AsyncStream<Result<Vec<u8>, CandleMcpError>>;
}

/// Standard input/output transport implementation for MCP communication.
///
/// This transport uses stdin/stdout for bidirectional communication with MCP servers,
/// which is the most common transport method for MCP tools. It uses standard channels
/// to handle the communication without blocking.
pub struct CandleStdioTransport {
    stdin_tx: crossbeam_channel::Sender<Vec<u8>>,
    stdout_rx: Arc<RwLock<crossbeam_channel::Receiver<Vec<u8>>>>,
}

impl CandleStdioTransport {
    /// Create a new StdioTransport instance.
    ///
    /// This method sets up bidirectional communication channels using stdin/stdout
    /// and spawns standard threads to handle the I/O operations. The transport is ready
    /// to use immediately after creation.
    ///
    /// # Returns
    ///
    /// A new `CandleStdioTransport` instance ready for MCP communication.
    #[inline]
    pub fn new() -> Self {
        let (stdin_tx, stdin_rx) = crossbeam_channel::bounded::<Vec<u8>>(1024);
        let (stdout_tx, stdout_rx) = crossbeam_channel::bounded::<Vec<u8>>(1024);

        std::thread::spawn(move || {
            use std::io::Write;
            let mut stdout = std::io::stdout();

            while let Ok(mut data) = stdin_rx.recv() {
                data.push(b'\n');
                if stdout.write_all(&data).is_err() {
                    break;
                }
                if stdout.flush().is_err() {
                    break;
                }
            }
        });

        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            let stdin = std::io::stdin();
            let mut reader = BufReader::new(stdin);
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
                    Err(_) => break,
                }
            }
        });

        Self {
            stdin_tx,
            stdout_rx: Arc::new(RwLock::new(stdout_rx)),
        }
    }
}

impl CandleTransport for CandleStdioTransport {
    #[inline]
    fn send(&self, data: &[u8]) -> AsyncStream<Result<(), CandleMcpError>> {
        let stdin_tx = self.stdin_tx.clone();
        let data = data.to_vec();
        AsyncStream::with_channel(|stream_sender| {
            std::thread::spawn(move || {
                let result = stdin_tx
                    .send(data)
                    .map_err(|_| CandleMcpError::TransportClosed);
                let _ = stream_sender.send(result);
            });
        })
    }

    #[inline]
    fn receive(&self) -> AsyncStream<Result<Vec<u8>, CandleMcpError>> {
        let stdout_rx = self.stdout_rx.clone();
        AsyncStream::with_channel(|stream_sender| {
            std::thread::spawn(move || {
                if let Ok(rx) = stdout_rx.try_read() {
                    let result = rx.recv().map_err(|_| CandleMcpError::TransportClosed);
                    let _ = stream_sender.send(result);
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
pub struct CandleClient<T: CandleTransport> {
    transport: Arc<T>,
    request_id: AtomicU64,
    response_cache: Arc<RwLock<HashMap<u64, Value>>>,
    request_timeout: Duration,
}

impl<T: CandleTransport> Clone for CandleClient<T> {
    fn clone(&self) -> Self {
        Self {
            transport: self.transport.clone(),
            request_id: AtomicU64::new(self.request_id.load(Ordering::Relaxed)),
            response_cache: self.response_cache.clone(),
            request_timeout: self.request_timeout,
        }
    }
}

impl<T: CandleTransport> CandleClient<T> {
    /// Create a new MCP client with the specified transport.
    ///
    /// # Arguments
    ///
    /// * `transport` - The transport implementation to use for communication
    ///
    /// # Returns
    ///
    /// A new `CandleClient` instance ready to communicate with MCP servers.
    #[inline]
    pub fn new(transport: T) -> Self {
        Self {
            transport: Arc::new(transport),
            request_id: AtomicU64::new(1),
            response_cache: Arc::new(RwLock::new(HashMap::with_capacity(256))),
            request_timeout: Duration::from_secs(30),
        }
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
    /// Returns the tool's result as a JSON `Value` on success, or `CandleMcpError` on failure.
    ///
    /// # Errors
    ///
    /// * `McpError::Timeout` - If the request times out
    /// * `McpError::ExecutionFailed` - If the tool execution fails
    /// * `McpError::SerializationFailed` - If JSON serialization/deserialization fails
    /// * `McpError::TransportClosed` - If the transport connection is closed
    pub fn call_tool(&self, name: &str, args: Value) -> AsyncStream<Result<Value, McpError>> {
        let transport = self.transport.clone();
        let request_id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let response_cache = self.response_cache.clone();
        let request_timeout = self.request_timeout;
        let name = name.to_string();

        AsyncStream::with_channel(move |stream_sender| {
            std::thread::spawn(move || {
                let start_time = Instant::now();

                let request = JsonRpcRequest {
                    jsonrpc: "2.0",
                    method: "tools/call".to_string(),
                    params: serde_json::json!({
                        "name": name,
                        "arguments": args
                    }),
                    id: request_id,
                };

                let mut buffer = Vec::with_capacity(1024);
                if let Err(_) = serde_json::to_writer(&mut buffer, &request) {
                    let _ = stream_sender.send(Err(McpError::SerializationFailed));
                    return;
                }

                let mut send_stream = transport.send(&buffer);
                if let Some(send_result) = send_stream.try_next() {
                    if let Err(e) = send_result {
                        let _ = stream_sender.send(Err(e));
                        return;
                    }
                }

                loop {
                    if start_time.elapsed() > request_timeout {
                        let _ = stream_sender.send(Err(McpError::Timeout));
                        return;
                    }

                    let mut receive_stream = transport.receive();
                    if let Some(response_result) = receive_stream.try_next() {
                        match response_result {
                            Ok(response_data) => {
                                let response: JsonRpcResponse =
                                    match serde_json::from_slice(&response_data) {
                                        Ok(r) => r,
                                        Err(_) => {
                                            let _ = stream_sender
                                                .send(Err(McpError::SerializationFailed));
                                            return;
                                        }
                                    };

                                if response.id == request_id {
                                    if let Some(error) = response.error {
                                        let _ = stream_sender
                                            .send(Err(McpError::ExecutionFailed(error.message)));
                                        return;
                                    }

                                    let result = response.result.ok_or(McpError::InvalidResponse);
                                    let _ = stream_sender.send(result);
                                    return;
                                }

                                // Cache response for other requests
                                if let Ok(mut cache) = response_cache.try_write() {
                                    if let Some(result) = response.result {
                                        cache.insert(response.id, result);
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = stream_sender.send(Err(e));
                                return;
                            }
                        }
                    }

                    // Small delay to prevent busy waiting
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            });
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
    pub fn list_tools(&self) -> AsyncStream<Result<Vec<CandleMcpToolData>, McpError>> {
        let client = self.clone();
        AsyncStream::with_channel(|stream_sender| {
            std::thread::spawn(move || {
                let mut internal_stream = client.call_tool_internal("tools/list", Value::Null);
                if let Some(result) = internal_stream.try_next() {
                    match result {
                        Ok(result_value) => {
                            if let Value::Object(obj) = result_value {
                                if let Some(Value::Array(tools)) = obj.get("tools") {
                                    let mut parsed_tools = Vec::with_capacity(tools.len());
                                    for tool in tools {
                                        if let Ok(parsed) =
                                            serde_json::from_value::<CandleMcpToolData>(
                                                tool.clone(),
                                            )
                                        {
                                            parsed_tools.push(parsed);
                                        }
                                    }
                                    let _ = stream_sender.send(Ok(parsed_tools));
                                    return;
                                }
                            }
                            let _ = stream_sender.send(Ok(Vec::new()));
                        }
                        Err(e) => {
                            let _ = stream_sender.send(Err(e));
                        }
                    }
                }
            });
        })
    }

    #[inline]
    fn call_tool_internal(
        &self,
        method: &str,
        params: Value,
    ) -> AsyncStream<Result<Value, CandleMcpError>> {
        let transport = self.transport.clone();
        let request_id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let request_timeout = self.request_timeout;
        let method = method.to_string();

        AsyncStream::with_channel(move |stream_sender| {
            std::thread::spawn(move || {
                let start_time = Instant::now();

                let request = JsonRpcRequest {
                    jsonrpc: "2.0",
                    method,
                    params,
                    id: request_id,
                };

                let mut buffer = Vec::with_capacity(512);
                if let Err(_) = serde_json::to_writer(&mut buffer, &request) {
                    let _ = stream_sender.send(Err(CandleMcpError::SerializationFailed));
                    return;
                }

                let mut send_stream = transport.send(&buffer);
                if let Some(send_result) = send_stream.try_next() {
                    if let Err(e) = send_result {
                        let _ = stream_sender.send(Err(e));
                        return;
                    }
                }

                loop {
                    if start_time.elapsed() > request_timeout {
                        let _ = stream_sender.send(Err(CandleMcpError::Timeout));
                        return;
                    }

                    let mut receive_stream = transport.receive();
                    if let Some(response_result) = receive_stream.try_next() {
                        match response_result {
                            Ok(response_data) => {
                                let response: JsonRpcResponse =
                                    match serde_json::from_slice(&response_data) {
                                        Ok(r) => r,
                                        Err(_) => {
                                            let _ = stream_sender
                                                .send(Err(CandleMcpError::SerializationFailed));
                                            return;
                                        }
                                    };

                                if response.id == request_id {
                                    if let Some(error) = response.error {
                                        let _ = stream_sender.send(Err(
                                            CandleMcpError::ExecutionFailed(error.message),
                                        ));
                                        return;
                                    }

                                    let result =
                                        response.result.ok_or(CandleMcpError::InvalidResponse);
                                    let _ = stream_sender.send(result);
                                    return;
                                }
                            }
                            Err(e) => {
                                let _ = stream_sender.send(Err(e));
                                return;
                            }
                        }
                    }

                    // Small delay to prevent busy waiting
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            });
        })
    }
}
