use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{RwLock, mpsc};

use crate::{AsyncTask, spawn_async};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

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

#[derive(Debug)]
pub enum McpError {
    TransportClosed,
    SerializationFailed,
    ToolNotFound,
    ExecutionFailed(String),
    Timeout,
    InvalidResponse,
}

pub trait Transport: Send + Sync + 'static {
    fn send(&self, data: &[u8]) -> impl std::future::Future<Output = Result<(), McpError>> + Send;
    fn receive(&self) -> impl std::future::Future<Output = Result<Vec<u8>, McpError>> + Send;
}

pub struct StdioTransport {
    stdin_tx: mpsc::UnboundedSender<Vec<u8>>,
    stdout_rx: Arc<RwLock<mpsc::UnboundedReceiver<Vec<u8>>>>,
}

impl StdioTransport {
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

pub struct Client<T: Transport> {
    transport: Arc<T>,
    request_id: AtomicU64,
    response_cache: Arc<RwLock<HashMap<u64, Value>>>,
    request_timeout: Duration,
}

impl<T: Transport> Client<T> {
    #[inline]
    pub fn new(transport: T) -> Self {
        Self {
            transport: Arc::new(transport),
            request_id: AtomicU64::new(1),
            response_cache: Arc::new(RwLock::new(HashMap::with_capacity(256))),
            request_timeout: Duration::from_secs(30),
        }
    }

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
            id,
        };

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

    #[inline]
    pub async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        let result = self.call_tool_internal("tools/list", Value::Null).await?;

        if let Value::Object(obj) = result {
            if let Some(Value::Array(tools)) = obj.get("tools") {
                let mut parsed_tools = Vec::with_capacity(tools.len());
                for tool in tools {
                    if let Ok(parsed) = serde_json::from_value::<Tool>(tool.clone()) {
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
            id,
        };

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

pub struct McpClient<T: Transport> {
    pub definition: Tool,
    pub client: Arc<Client<T>>,
}

// McpClientBuilder moved to fluent-ai/src/builders/mcp.rs

// McpClient::define moved to fluent-ai/src/builders/mcp.rs

// McpClientBuilder implementation moved to fluent-ai/src/builders/mcp.rs
