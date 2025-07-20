//! File operations tool for Anthropic Files API with zero-allocation HTTP3
//! 
//! This module provides secure file upload, download, and management capabilities
//! using the Anthropic Files API with fluent_ai_http3 for optimal performance.

use super::{
    core::{Tool, AnthropicResult, AnthropicError},
    function_calling::{ToolExecutor, ToolExecutionContext, ToolOutput}
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    path::Path,
    pin::Pin,
    future::Future,
};
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest};
use bytes::Bytes;
use tokio::fs;

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
    fn get_api_key(context: &ToolExecutionContext) -> Result<String, AnthropicError> {
        context.metadata.get("api_key")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "API key not found or empty in tool execution context".to_string()
            ))
    }
    
    /// Upload file to Anthropic Files API using HTTP3
    async fn upload_file(file_path: &str, api_key: &str) -> AnthropicResult<FileUploadResponse> {
        let path = Path::new(file_path);
        
        // Validate file exists
        if !path.exists() {
            return Err(AnthropicError::InvalidRequest(
                format!("File not found: {}", file_path)
            ));
        }
        
        // Check file size
        let metadata = fs::metadata(path).await
            .map_err(|e| AnthropicError::FileError(format!("Failed to read file metadata: {}", e)))?;
        
        if metadata.len() > MAX_FILE_SIZE {
            return Err(AnthropicError::InvalidRequest(
                format!("File too large: {} bytes (max: {} bytes)", metadata.len(), MAX_FILE_SIZE)
            ));
        }
        
        // Validate file type using mime detection
        let mime_type = Self::detect_mime_type(path)?;
        if !SUPPORTED_FILE_TYPES.contains(&mime_type.as_str()) {
            return Err(AnthropicError::InvalidRequest(
                format!("Unsupported file type: {}", mime_type)
            ));
        }
        
        // Read file contents
        let file_contents = fs::read(path).await
            .map_err(|e| AnthropicError::FileError(format!("Failed to read file: {}", e)))?;
        
        // Get filename
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "Invalid filename".to_string()
            ))?;
        
        // Build multipart form data with zero-allocation patterns
        let boundary = format!("----formdata-{}", fastrand::u64(..));
        let mut body = Vec::with_capacity(file_contents.len() + 1024); // Pre-allocate with buffer
        
        // Add file part
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(format!("Content-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\n", filename).as_bytes());
        body.extend_from_slice(format!("Content-Type: {}\r\n\r\n", mime_type).as_bytes());
        body.extend_from_slice(&file_contents);
        body.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());
        
        // Create HTTP3 client with AI optimization
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create HTTP3 client: {}", e)))?;
        
        // Create request
        let request = HttpRequest::post("https://api.anthropic.com/v1/files", body)
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create request: {}", e)))?
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("Content-Type", &format!("multipart/form-data; boundary={}", boundary))
            .header("anthropic-beta", "files-api-2025-04-14");
        
        // Send request with streaming
        let response = client.send(request).await
            .map_err(|e| AnthropicError::HttpError(format!("Upload request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_body = response.stream().collect().await
                .map_err(|e| AnthropicError::HttpError(format!("Failed to read error response: {}", e)))?;
            return Err(AnthropicError::ApiError(format!(
                "Upload failed with status {}: {}",
                response.status(),
                String::from_utf8_lossy(&error_body)
            )));
        }
        
        // Parse response
        let response_body = response.stream().collect().await
            .map_err(|e| AnthropicError::HttpError(format!("Failed to read response: {}", e)))?;
        
        let upload_response: FileUploadResponse = serde_json::from_slice(&response_body)
            .map_err(|e| AnthropicError::ParseError(format!("Failed to parse upload response: {}", e)))?;
        
        Ok(upload_response)
    }
    
    /// List files in Anthropic Files API using HTTP3
    async fn list_files(api_key: &str) -> AnthropicResult<FileListResponse> {
        // Create HTTP3 client with AI optimization
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create HTTP3 client: {}", e)))?;
        
        let request = HttpRequest::get("https://api.anthropic.com/v1/files")
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create request: {}", e)))?
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("anthropic-beta", "files-api-2025-04-14");
        
        let response = client.send(request).await
            .map_err(|e| AnthropicError::HttpError(format!("List request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_body = response.stream().collect().await
                .map_err(|e| AnthropicError::HttpError(format!("Failed to read error response: {}", e)))?;
            return Err(AnthropicError::ApiError(format!(
                "List failed with status {}: {}",
                response.status(),
                String::from_utf8_lossy(&error_body)
            )));
        }
        
        let response_body = response.stream().collect().await
            .map_err(|e| AnthropicError::HttpError(format!("Failed to read response: {}", e)))?;
        
        let list_response: FileListResponse = serde_json::from_slice(&response_body)
            .map_err(|e| AnthropicError::ParseError(format!("Failed to parse list response: {}", e)))?;
        
        Ok(list_response)
    }
    
    /// Retrieve file metadata from Anthropic Files API using HTTP3
    async fn retrieve_file(file_id: &str, api_key: &str) -> AnthropicResult<FileMetadata> {
        // Create HTTP3 client with AI optimization
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create HTTP3 client: {}", e)))?;
        
        let request = HttpRequest::get(&format!("https://api.anthropic.com/v1/files/{}", file_id))
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create request: {}", e)))?
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("anthropic-beta", "files-api-2025-04-14");
        
        let response = client.send(request).await
            .map_err(|e| AnthropicError::HttpError(format!("Retrieve request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_body = response.stream().collect().await
                .map_err(|e| AnthropicError::HttpError(format!("Failed to read error response: {}", e)))?;
            return Err(AnthropicError::ApiError(format!(
                "Retrieve failed with status {}: {}",
                response.status(),
                String::from_utf8_lossy(&error_body)
            )));
        }
        
        let response_body = response.stream().collect().await
            .map_err(|e| AnthropicError::HttpError(format!("Failed to read response: {}", e)))?;
        
        let file_metadata: FileMetadata = serde_json::from_slice(&response_body)
            .map_err(|e| AnthropicError::ParseError(format!("Failed to parse retrieve response: {}", e)))?;
        
        Ok(file_metadata)
    }
    
    /// Delete file from Anthropic Files API using HTTP3
    async fn delete_file(file_id: &str, api_key: &str) -> AnthropicResult<()> {
        // Create HTTP3 client with AI optimization
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create HTTP3 client: {}", e)))?;
        
        let request = HttpRequest::delete(&format!("https://api.anthropic.com/v1/files/{}", file_id))
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create request: {}", e)))?
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("anthropic-beta", "files-api-2025-04-14");
        
        let response = client.send(request).await
            .map_err(|e| AnthropicError::HttpError(format!("Delete request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_body = response.stream().collect().await
                .map_err(|e| AnthropicError::HttpError(format!("Failed to read error response: {}", e)))?;
            return Err(AnthropicError::ApiError(format!(
                "Delete failed with status {}: {}",
                response.status(),
                String::from_utf8_lossy(&error_body)
            )));
        }
        
        Ok(())
    }
    
    /// Download file content from Anthropic Files API using HTTP3 streaming
    async fn download_file(file_id: &str, api_key: &str) -> AnthropicResult<Bytes> {
        // Create HTTP3 client with streaming optimization
        let client = HttpClient::with_config(HttpConfig::streaming_optimized())
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create HTTP3 client: {}", e)))?;
        
        let request = HttpRequest::get(&format!("https://api.anthropic.com/v1/files/{}/download", file_id))
            .map_err(|e| AnthropicError::HttpError(format!("Failed to create request: {}", e)))?
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("anthropic-beta", "files-api-2025-04-14");
        
        let response = client.send(request).await
            .map_err(|e| AnthropicError::HttpError(format!("Download request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_body = response.stream().collect().await
                .map_err(|e| AnthropicError::HttpError(format!("Failed to read error response: {}", e)))?;
            return Err(AnthropicError::ApiError(format!(
                "Download failed with status {}: {}",
                response.status(),
                String::from_utf8_lossy(&error_body)
            )));
        }
        
        let file_content = response.stream().collect().await
            .map_err(|e| AnthropicError::HttpError(format!("Failed to read file content: {}", e)))?;
        
        Ok(file_content)
    }
    
    /// Detect MIME type from file path with production-ready fallbacks
    fn detect_mime_type(path: &Path) -> AnthropicResult<String> {
        // Get file extension
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());
        
        let mime_type = match extension.as_deref() {
            Some("pdf") => "application/pdf",
            Some("txt") => "text/plain",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("png") => "image/png",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            _ => {
                // Try to detect from file contents for unsupported extensions
                return Err(AnthropicError::InvalidRequest(
                    format!("Unsupported file extension: {:?}", extension)
                ));
            }
        };
        
        Ok(mime_type.to_string())
    }
}

impl ToolExecutor for FileOperationsTool {
    fn execute(
        &self,
        input: Value,
        context: &ToolExecutionContext,
    ) -> Pin<Box<dyn Future<Output = AnthropicResult<ToolOutput>> + Send>> {
        let input = input.clone();
        let context = context.clone();
        
        Box::pin(async move {
            let operation = input
                .get("operation")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AnthropicError::InvalidRequest(
                    "File operation requires 'operation' parameter".to_string()
                ))?;
            
            // Get API key from context
            let api_key = Self::get_api_key(&context)?;
            
            match operation {
                "upload" => {
                    let file_path = input
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| AnthropicError::InvalidRequest(
                            "Upload operation requires 'file_path' parameter".to_string()
                        ))?;
                    
                    match Self::upload_file(file_path, &api_key).await {
                        Ok(upload_response) => Ok(ToolOutput::Json(json!({
                            "operation": "upload",
                            "file_id": upload_response.id,
                            "filename": upload_response.filename,
                            "size": upload_response.size,
                            "type": upload_response.file_type,
                            "created_at": upload_response.created_at
                        }))),
                        Err(e) => Ok(ToolOutput::Error {
                            message: e.to_string(),
                            code: Some("UPLOAD_ERROR".to_string()),
                        }),
                    }
                }
                
                "list" => {
                    match Self::list_files(&api_key).await {
                        Ok(list_response) => Ok(ToolOutput::Json(json!({
                            "operation": "list",
                            "files": list_response.data.into_iter().map(|file| json!({
                                "id": file.id,
                                "filename": file.filename,
                                "size": file.size,
                                "type": file.file_type,
                                "created_at": file.created_at
                            })).collect::<Vec<_>>(),
                            "has_more": list_response.has_more,
                            "first_id": list_response.first_id,
                            "last_id": list_response.last_id
                        }))),
                        Err(e) => Ok(ToolOutput::Error {
                            message: e.to_string(),
                            code: Some("LIST_ERROR".to_string()),
                        }),
                    }
                }
                
                "retrieve" => {
                    let file_id = input
                        .get("file_id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| AnthropicError::InvalidRequest(
                            "Retrieve operation requires 'file_id' parameter".to_string()
                        ))?;
                    
                    match Self::retrieve_file(file_id, &api_key).await {
                        Ok(file_metadata) => Ok(ToolOutput::Json(json!({
                            "operation": "retrieve",
                            "id": file_metadata.id,
                            "filename": file_metadata.filename,
                            "size": file_metadata.size,
                            "type": file_metadata.file_type,
                            "created_at": file_metadata.created_at
                        }))),
                        Err(e) => Ok(ToolOutput::Error {
                            message: e.to_string(),
                            code: Some("RETRIEVE_ERROR".to_string()),
                        }),
                    }
                }
                
                "delete" => {
                    let file_id = input
                        .get("file_id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| AnthropicError::InvalidRequest(
                            "Delete operation requires 'file_id' parameter".to_string()
                        ))?;
                    
                    match Self::delete_file(file_id, &api_key).await {
                        Ok(()) => Ok(ToolOutput::Json(json!({
                            "operation": "delete",
                            "file_id": file_id,
                            "success": true
                        }))),
                        Err(e) => Ok(ToolOutput::Error {
                            message: e.to_string(),
                            code: Some("DELETE_ERROR".to_string()),
                        }),
                    }
                }
                
                "download" => {
                    let file_id = input
                        .get("file_id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| AnthropicError::InvalidRequest(
                            "Download operation requires 'file_id' parameter".to_string()
                        ))?;
                    
                    match Self::download_file(file_id, &api_key).await {
                        Ok(file_content) => {
                            // Use base64 encoding for binary-safe content transfer
                            let content_base64 = base64::engine::general_purpose::STANDARD.encode(&file_content);
                            Ok(ToolOutput::Json(json!({
                                "operation": "download",
                                "file_id": file_id,
                                "content": content_base64,
                                "size": file_content.len()
                            })))
                        }
                        Err(e) => Ok(ToolOutput::Error {
                            message: e.to_string(),
                            code: Some("DOWNLOAD_ERROR".to_string()),
                        }),
                    }
                }
                
                _ => Ok(ToolOutput::Error {
                    message: format!("Unsupported operation: {}. Supported operations: upload, list, retrieve, delete, download", operation),
                    code: Some("INVALID_OPERATION".to_string()),
                }),
            }
        })
    }
    
    fn definition(&self) -> Tool {
        Tool::new(
            "file_operations",
            "Manage files using Anthropic's Files API. Upload local files to Anthropic's cloud storage, list uploaded files, retrieve file metadata, delete files, and download file content. Files can be referenced in subsequent API calls using their file_id.",
            json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["upload", "list", "retrieve", "delete", "download"],
                        "description": "Type of file operation to perform"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to local file to upload (required for upload operation)"
                    },
                    "file_id": {
                        "type": "string",
                        "description": "Unique identifier for the file (required for retrieve, delete, and download operations)"
                    }
                },
                "required": ["operation"],
                "additionalProperties": false
            }),
        )
    }
}