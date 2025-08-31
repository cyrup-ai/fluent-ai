//! SweetMCP Cap'n Proto Client Library
//! 
//! This library provides a Cap'n Proto client for demonstrating the SweetMCP protocol
//! extension that supports GraphQL and Cap'n Proto requests converted to JSON-RPC.

use anyhow::{Context, Result};
use capnp::{message, serialize_packed};
use serde_json::Value;
use std::io::Cursor;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

// Generated Cap'n Proto code
capnp::generated_code!(pub mod mcp_request_capnp);

pub use mcp_request_capnp::{mcp_tool_request, mcp_tool_response};

/// Cap'n Proto client for MCP tool requests
pub struct McpCapnProtoClient {
    base_url: String,
    client: reqwest::Client,
}

impl McpCapnProtoClient {
    /// Create a new client with the SweetMCP server base URL
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a Cap'n Proto request for the time tool
    pub fn create_time_request() -> Result<Vec<u8>> {
        let mut message = message::Builder::new_default();
        let mut request = message.init_root::<mcp_tool_request::Builder>();
        
        // Set request details
        let request_id = Uuid::new_v4().to_string();
        request.set_request_id(&request_id);
        request.set_tool_name("time");
        
        // Set metadata first (before we move request for arguments)
        {
            let mut metadata = request.reborrow().init_metadata();
            metadata.set_client_name("sweetmcp-capnp-client");
            metadata.set_client_version("0.1.0");
            metadata.set_protocol_version("2025-03-26");
            metadata.set_timestamp(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            );
        }
        
        // Set arguments for get_time_utc operation
        let mut arguments = request.init_arguments(1);
        let mut arg = arguments.reborrow().get(0);
        arg.set_key("name");
        arg.get_value()?.set_text("get_time_utc");
        
        // Serialize to binary
        let mut buffer = Vec::new();
        serialize_packed::write_message(&mut buffer, &message)
            .context("Failed to serialize Cap'n Proto message")?;
        
        Ok(buffer)
    }

    /// Create a Cap'n Proto request for the hash tool
    pub fn create_hash_request(data: &str, algorithm: &str) -> Result<Vec<u8>> {
        let mut message = message::Builder::new_default();
        let mut request = message.init_root::<mcp_tool_request::Builder>();
        
        // Set request details
        let request_id = Uuid::new_v4().to_string();
        request.set_request_id(&request_id);
        request.set_tool_name("hash");
        
        // Set metadata first (before we move request for arguments)
        {
            let mut metadata = request.reborrow().init_metadata();
            metadata.set_client_name("sweetmcp-capnp-client");
            metadata.set_client_version("0.1.0");
            metadata.set_protocol_version("2025-03-26");
            metadata.set_timestamp(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            );
        }
        
        // Set arguments for hash operation
        let mut arguments = request.init_arguments(2);
        
        // Data argument
        let mut data_arg = arguments.reborrow().get(0);
        data_arg.set_key("data");
        data_arg.get_value()?.set_text(data);
        
        // Algorithm argument
        let mut algo_arg = arguments.reborrow().get(1);
        algo_arg.set_key("algorithm");
        algo_arg.get_value()?.set_text(algorithm);
        
        // Serialize to binary
        let mut buffer = Vec::new();
        serialize_packed::write_message(&mut buffer, &message)
            .context("Failed to serialize Cap'n Proto message")?;
        
        Ok(buffer)
    }

    /// Send a Cap'n Proto request to the SweetMCP server and get response
    pub async fn send_request(&self, capnp_request: Vec<u8>) -> Result<McpResponse> {
        // Send Cap'n Proto binary data to SweetMCP server
        let response = self
            .client
            .post(&format!("{}/mcp", self.base_url))
            .header("Content-Type", "application/capnp")
            .body(capnp_request)
            .send()
            .await
            .context("Failed to send request to SweetMCP server")?;

        // Check if request was successful
        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Failed to read error response".to_string());
            return Err(anyhow::anyhow!(
                "Server returned error status: {} - {}",
                response.status(),
                error_text
            ));
        }

        // Parse response - should be Cap'n Proto binary back
        let response_bytes = response
            .bytes()
            .await
            .context("Failed to read response bytes")?;

        // Try to parse as Cap'n Proto
        if let Ok(capnp_response) = Self::parse_capnp_response(&response_bytes) {
            return Ok(capnp_response);
        }

        // Fallback: try to parse as JSON for debugging
        if let Ok(json_str) = std::str::from_utf8(&response_bytes) {
            if let Ok(json_value) = serde_json::from_str::<Value>(json_str) {
                return Ok(McpResponse::Json(json_value));
            }
        }

        Err(anyhow::anyhow!("Could not parse response as Cap'n Proto or JSON"))
    }

    /// Parse Cap'n Proto response
    fn parse_capnp_response(data: &[u8]) -> Result<McpResponse> {
        let mut cursor = Cursor::new(data);
        let message_reader = serialize_packed::read_message(
            &mut cursor,
            message::ReaderOptions::new(),
        )
        .context("Failed to read Cap'n Proto response")?;

        let response = message_reader
            .get_root::<mcp_tool_response::Reader>()
            .context("Failed to get response root")?;

        let request_id = response
            .get_request_id()
            .and_then(|id| id.to_string().map_err(|e| capnp::Error::failed(format!("UTF-8 error: {}", e))))
            .context("Failed to get request ID")?;

        let status = match response.get_status() {
            Ok(status_enum) => {
                match status_enum {
                    mcp_tool_response::ResponseStatus::Success => "success".to_string(),
                    mcp_tool_response::ResponseStatus::Error => "error".to_string(), 
                    mcp_tool_response::ResponseStatus::Timeout => "timeout".to_string(),
                }
            }
            Err(_) => "unknown".to_string(),
        };

        // Parse result content
        let mut content_items = Vec::new();
        if let Ok(result) = response.get_result() {
            if let Ok(content_list) = result.get_content() {
                for content_item in content_list {
                    let content_type = content_item
                        .get_content_type()
                        .and_then(|t| t.to_string())
                        .unwrap_or_else(|_| "text".to_string());
                    let data = content_item
                        .get_data()
                        .and_then(|d| d.to_string())
                        .unwrap_or_else(|_| String::new());
                    let mime_type = content_item
                        .get_mime_type()
                        .and_then(|m| m.to_string())
                        .unwrap_or_else(|_| String::new());

                    content_items.push(ContentItem {
                        content_type,
                        data,
                        mime_type,
                    });
                }
            }
        }

        Ok(McpResponse::CapnProto {
            request_id,
            status,
            content: content_items,
        })
    }
}

/// MCP Response representation
#[derive(Debug, Clone)]
pub enum McpResponse {
    CapnProto {
        request_id: String,
        status: String,
        content: Vec<ContentItem>,
    },
    Json(Value),
}

/// Content item from MCP tool response
#[derive(Debug, Clone)]
pub struct ContentItem {
    pub content_type: String,
    pub data: String,
    pub mime_type: String,
}

impl McpResponse {
    /// Get the main content as text
    pub fn get_text_content(&self) -> Option<String> {
        match self {
            McpResponse::CapnProto { content, .. } => {
                content
                    .iter()
                    .find(|item| item.content_type == "text")
                    .map(|item| item.data.clone())
            }
            McpResponse::Json(value) => {
                // Try to extract content from JSON-RPC response
                value
                    .get("result")
                    .and_then(|r| r.get("content"))
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|item| item.get("text"))
                    .and_then(|text| text.as_str())
                    .map(|s| s.to_string())
            }
        }
    }

    /// Check if response indicates success
    pub fn is_success(&self) -> bool {
        match self {
            McpResponse::CapnProto { status, .. } => status == "success",
            McpResponse::Json(value) => !value.get("error").is_some(),
        }
    }
}