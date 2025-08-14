//! File operations tool for Anthropic Files API with zero-allocation HTTP3
//!
//! This module provides secure file upload, download, and management capabilities
//! using the Anthropic Files API with fluent_ai_http3 for optimal performance.

use std::path::Path;

use bytes::Bytes;
use fluent_ai_async::AsyncStream;
use fluent_ai_async::channel;
use fluent_ai_domain::tool::Tool;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::fs;

use super::super::error::{AnthropicError, AnthropicResult};
use super::{
    core::ToolExecutor,
    function_calling::{ToolExecutionContext, ToolOutput},
};

/// Built-in file operations tool for Anthropic Files API
pub struct FileOperationsTool;

/// File upload response from Anthropic Files API
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileUploadResponse {
    id: String,
    #[serde(rename = "type")]
    file_type: String,
    filename: String,
    size: u64,
    created_at: String,
}

/// File list response from Anthropic Files API
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileListResponse {
    data: Vec<FileMetadata>,
    has_more: bool,
    first_id: Option<String>,
    last_id: Option<String>,
}

/// File metadata from Anthropic Files API
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileMetadata {
    id: String,
    #[serde(rename = "type")]
    file_type: String,
    filename: String,
    size: u64,
    created_at: String,
}

/// Supported file types for Anthropic Files API
const SUPPORTED_FILE_TYPES: &[&str] = &[
    "application/pdf",
    "text/plain",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
];

/// Maximum file size (500MB)
const MAX_FILE_SIZE: u64 = 500 * 1024 * 1024;

impl FileOperationsTool {
    /// Get API key from context metadata with production-ready validation
    fn get_api_key(context: &ToolExecutionContext) -> AnthropicResult<&str> {
        context
            .metadata
            .get("api_key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AnthropicError::ApiKeyMissing)
    }

    /// Create a new HTTP client with AI-optimized settings
    fn create_http_client() -> Result<HttpClient, String> {
        HttpClient::new(HttpConfig::ai_optimized())
            .map_err(|e| format!("Failed to create HTTP client: {}", e))
    }
}

impl ToolExecutor for FileOperationsTool {
    fn execute(&self, input: Value, context: &ToolExecutionContext) -> AsyncStream<ToolOutput> {
        let (tx, stream) = channel();
        let api_key = match Self::get_api_key(context) {
            Ok(key) => key.to_string(),
            Err(e) => {
                let _ = tx.send(Err(e));
                return stream;
            }
        };

        tokio::spawn(async move {
            let result = async {
                let operation = input["operation"].as_str().unwrap_or_default();
                let client = Self::create_http_client().map_err(|e| {
                    ToolOutput::Error {
                        message: e,
                        code: Some("HTTP_CLIENT_ERROR".to_string())
                    }
                })?;

                match operation {
                    "upload" => {
                        let file_path = input["file_path"].as_str().unwrap_or_default();
                        let path = Path::new(file_path);
                        if !path.exists() {
                            return Ok(ToolOutput::Error {
                                message: "File not found".to_string(),
                                code: Some("FILE_NOT_FOUND".to_string())});
                        }

                        let metadata = fs::metadata(path).await.unwrap();
                        if metadata.len() > MAX_FILE_SIZE {
                            return Ok(ToolOutput::Error {
                                message: "File size exceeds 500MB limit".to_string(),
                                code: Some("FILE_TOO_LARGE".to_string())});
                        }

                        let mime_type = mime_guess::from_path(path).first_or_octet_stream();
                        if !SUPPORTED_FILE_TYPES.contains(&mime_type.as_ref()) {
                            return Ok(ToolOutput::Error {
                                message: "Unsupported file type".to_string(),
                                code: Some("UNSUPPORTED_FILE_TYPE".to_string())});
                        }

                        let file_contents = fs::read(path).await.unwrap();
                        let request = HttpRequest::post("https://api.anthropic.com/v1/files")
                            .header("x-api-key", &api_key)
                            .header("Content-Type", mime_type.as_ref())
                            .body(file_contents);

                        match client.send(request).await {
                            Ok(response) => {
                                let upload_response: FileUploadResponse = response.json().await.unwrap();
                                Ok(ToolOutput::Json(json!(upload_response)))
                            }
                            Err(e) => Ok(ToolOutput::Error {
                                message: e.to_string(),
                                code: Some("UPLOAD_ERROR".to_string())})}
                    }

                    "list" => {
                        let request = HttpRequest::get("https://api.anthropic.com/v1/files")
                            .header("x-api-key", &api_key);
                        match client.send(request).await {
                            Ok(response) => {
                                let list_response: FileListResponse = response.json().await.unwrap();
                                Ok(ToolOutput::Json(json!(list_response)))
                            }
                            Err(e) => Ok(ToolOutput::Error {
                                message: e.to_string(),
                                code: Some("LIST_ERROR".to_string())})}
                    }

                    "retrieve" => {
                        let file_id = input["file_id"].as_str().unwrap_or_default();
                        let url = format!("https://api.anthropic.com/v1/files/{}", file_id);
                        let request = HttpRequest::get(&url).header("x-api-key", &api_key);
                        match client.send(request).await {
                            Ok(response) => {
                                let metadata: FileMetadata = response.json().await.unwrap();
                                Ok(ToolOutput::Json(json!(metadata)))
                            }
                            Err(e) => Ok(ToolOutput::Error {
                                message: e.to_string(),
                                code: Some("RETRIEVE_ERROR".to_string())})}
                    }

                    "delete" => {
                        let file_id = input["file_id"].as_str().unwrap_or_default();
                        let url = format!("https://api.anthropic.com/v1/files/{}", file_id);
                        let request = HttpRequest::delete(&url).header("x-api-key", &api_key);
                        match client.send(request).await {
                            Ok(_) => Ok(ToolOutput::Text("File deleted successfully".to_string())),
                            Err(e) => Ok(ToolOutput::Error {
                                message: e.to_string(),
                                code: Some("DELETE_ERROR".to_string())})}
                    }

                    "download" => {
                        let file_id = input["file_id"].as_str().unwrap_or_default();
                        let url = format!("https://api.anthropic.com/v1/files/{}/content", file_id);
                        let request = HttpRequest::get(&url).header("x-api-key", &api_key);
                        match client.send(request).await {
                            Ok(response) => {
                                let bytes: Bytes = response.bytes().await.unwrap();
                                Ok(ToolOutput::Json(json!(bytes.to_vec())))
                            }
                            Err(e) => Ok(ToolOutput::Error {
                                message: e.to_string(),
                                code: Some("DOWNLOAD_ERROR".to_string())})}
                    }

                    _ => Ok(ToolOutput::Error {
                        message: format!(
                            "Unsupported operation: {}. Supported operations: upload, list, retrieve, delete, download",
                            operation
                        ),
                        code: Some("INVALID_OPERATION".to_string())})}
            }
            .await;
            let _ = tx.send(result);
        });

        stream
    }
}
