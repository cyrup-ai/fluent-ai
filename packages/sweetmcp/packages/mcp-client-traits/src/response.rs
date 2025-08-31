//! Response adapters and utilities for MCP client implementations
//!
//! This module provides utilities for working with MCP responses and extracting
//! content in a standardized way across different protocol implementations.

use sweet_mcp_type::{Response, JsonValue, JsonRpcError};
use value_trait::prelude::*;
use simd_json;
use crate::errors::ClientError;

/// Adapter trait for converting protocol-specific responses to MCP Response
///
/// Different protocols (GraphQL, JSON-RPC, Cap'n Proto) may have their own
/// response formats that need to be adapted to the standard MCP Response structure.
pub trait ResponseAdapter<T> {
    /// Convert a protocol-specific response to MCP Response
    ///
    /// # Arguments
    /// * `response` - The protocol-specific response
    ///
    /// # Returns
    /// Standardized MCP Response from sweet-mcp-type
    fn to_mcp_response(&self, response: T) -> Result<Response, ClientError>;

    /// Convert MCP Response to protocol-specific format
    ///
    /// # Arguments
    /// * `response` - MCP Response from sweet-mcp-type
    ///
    /// # Returns
    /// Protocol-specific response format
    fn from_mcp_response(&self, response: Response) -> Result<T, ClientError>;
}

/// Content extraction utilities for MCP responses
///
/// This trait provides convenient methods for extracting specific types of
/// content from MCP tool responses.
pub trait ContentExtractor {
    /// Extract text content from the response
    ///
    /// # Returns
    /// First text content item, or None if not found
    fn extract_text(&self) -> Option<&str>;

    /// Extract all text content from the response
    ///
    /// # Returns
    /// Vector of all text content items
    fn extract_all_text(&self) -> Vec<&str>;

    /// Extract binary data content
    ///
    /// # Returns
    /// First binary data content (base64 encoded), or None if not found
    fn extract_data(&self) -> Option<&str>;

    /// Extract content by MIME type
    ///
    /// # Arguments
    /// * `mime_type` - The MIME type to search for
    ///
    /// # Returns
    /// Content with matching MIME type, or None if not found
    fn extract_by_mime_type(&self, mime_type: &str) -> Option<&str>;

    /// Check if the response indicates success
    ///
    /// # Returns
    /// True if the response indicates successful execution
    fn is_success(&self) -> bool;

    /// Extract error information if present
    ///
    /// # Returns
    /// Error details if the response contains an error
    fn extract_error(&self) -> Option<&JsonRpcError>;

    /// Get response metadata
    ///
    /// # Returns
    /// Optional metadata from the response
    fn extract_metadata(&self) -> Option<&JsonValue>;
}

impl ContentExtractor for Response {
    fn extract_text(&self) -> Option<&str> {
        if let Some(result) = &self.result {
            if let Some(content_array) = result.get("content").and_then(|c| c.as_array()) {
                for content_item in content_array {
                    if let Some(content_type) = content_item.get("type").and_then(|t| t.as_str()) {
                        if content_type == "text" {
                            if let Some(text) = content_item.get("text").and_then(|t| t.as_str()) {
                                return Some(text);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn extract_all_text(&self) -> Vec<&str> {
        let mut texts = Vec::new();
        
        if let Some(result) = &self.result {
            if let Some(content_array) = result.get("content").and_then(|c| c.as_array()) {
                for content_item in content_array {
                    if let Some(content_type) = content_item.get("type").and_then(|t| t.as_str()) {
                        if content_type == "text" {
                            if let Some(text) = content_item.get("text").and_then(|t| t.as_str()) {
                                texts.push(text);
                            }
                        }
                    }
                }
            }
        }
        
        texts
    }

    fn extract_data(&self) -> Option<&str> {
        if let Some(result) = &self.result {
            if let Some(content_array) = result.get("content").and_then(|c| c.as_array()) {
                for content_item in content_array {
                    if let Some(content_type) = content_item.get("type").and_then(|t| t.as_str()) {
                        if content_type == "image" || content_type == "audio" {
                            if let Some(data) = content_item.get("data").and_then(|d| d.as_str()) {
                                return Some(data);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn extract_by_mime_type(&self, mime_type: &str) -> Option<&str> {
        if let Some(result) = &self.result {
            if let Some(content_array) = result.get("content").and_then(|c| c.as_array()) {
                for content_item in content_array {
                    if let Some(item_mime) = content_item.get("mimeType").and_then(|m| m.as_str()) {
                        if item_mime == mime_type {
                            // Return text content for text MIME types
                            if mime_type.starts_with("text/") {
                                if let Some(text) = content_item.get("text").and_then(|t| t.as_str()) {
                                    return Some(text);
                                }
                            } else {
                                // Return data content for binary MIME types
                                if let Some(data) = content_item.get("data").and_then(|d| d.as_str()) {
                                    return Some(data);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn is_success(&self) -> bool {
        self.error.is_none() && self.result.is_some()
    }

    fn extract_error(&self) -> Option<&JsonRpcError> {
        self.error.as_ref()
    }

    fn extract_metadata(&self) -> Option<&JsonValue> {
        // MCP responses may include metadata in the result
        if let Some(result) = &self.result {
            result.get("_meta")
        } else {
            None
        }
    }
}

/// Utility functions for working with tool responses
pub struct ToolResponseUtils;

impl ToolResponseUtils {
    /// Parse a time tool response
    ///
    /// # Arguments
    /// * `response` - The MCP response from time tool
    ///
    /// # Returns
    /// Parsed time information or error
    pub fn parse_time_response(response: &Response) -> Result<TimeResult, ClientError> {
        if !response.is_success() {
            if let Some(error) = response.extract_error() {
                return Err(ClientError::tool_execution(
                    "time",
                    &error.message,
                    Some(error.code),
                    error.data.clone(),
                ));
            }
            return Err(ClientError::tool_execution(
                "time",
                "Unknown error occurred",
                None,
                None,
            ));
        }

        let text_content = response.extract_text()
            .ok_or_else(|| ClientError::response_parse(
                "No text content found",
                "time tool response",
            ))?;

        // Parse JSON response from time tool
        let mut time_data_bytes = text_content.as_bytes().to_vec();
        let time_data: JsonValue = simd_json::to_owned_value(&mut time_data_bytes)
            .map_err(|e| ClientError::response_parse(
                format!("Invalid JSON: {}", e),
                "time tool response content",
            ))?;

        Ok(TimeResult {
            utc_time: time_data.get("utc_time")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            utc_time_rfc2822: time_data.get("utc_time_rfc2822")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            parsed_time: time_data.get("parsed_time")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            formatted: time_data.get("formatted")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        })
    }

    /// Parse a hash tool response
    ///
    /// # Arguments
    /// * `response` - The MCP response from hash tool
    ///
    /// # Returns
    /// Hash result or error
    pub fn parse_hash_response(response: &Response) -> Result<String, ClientError> {
        if !response.is_success() {
            if let Some(error) = response.extract_error() {
                return Err(ClientError::tool_execution(
                    "hash",
                    &error.message,
                    Some(error.code),
                    error.data.clone(),
                ));
            }
            return Err(ClientError::tool_execution(
                "hash",
                "Unknown error occurred",
                None,
                None,
            ));
        }

        response.extract_text()
            .ok_or_else(|| ClientError::response_parse(
                "No text content found",
                "hash tool response",
            ))
            .map(|s| s.to_string())
    }

    /// Validate response structure according to MCP specification
    ///
    /// # Arguments
    /// * `response` - The response to validate
    ///
    /// # Returns
    /// Result indicating if response is valid
    pub fn validate_response(response: &Response) -> Result<(), ClientError> {
        // Check that we have either result or error, but not both
        match (&response.result, &response.error) {
            (Some(_), Some(_)) => Err(ClientError::response_parse(
                "Response contains both result and error",
                "MCP response validation",
            )),
            (None, None) => Err(ClientError::response_parse(
                "Response contains neither result nor error",
                "MCP response validation",
            )),
            _ => Ok(()),
        }
    }
}

/// Parsed time tool result
#[derive(Debug, Clone)]
pub struct TimeResult {
    pub utc_time: Option<String>,
    pub utc_time_rfc2822: Option<String>,
    pub parsed_time: Option<String>,
    pub formatted: Option<String>,
}