//! HTTP Response Chunk Types
//!
//! Proper MessageChunk implementations for HTTP responses following fluent_ai_async patterns.

use std::collections::HashMap;

use fluent_ai_async::prelude::*;
use http::StatusCode;

/// HTTP Response Chunk - the core streaming unit for HTTP operations
#[derive(Debug, Clone, Default)]
pub struct HttpResponseChunk {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub url: String,
    pub error_message: Option<String>,
}

impl MessageChunk for HttpResponseChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            status: 0,
            headers: HashMap::new(),
            body: Vec::new(),
            url: String::new(),
            error_message: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error_message.is_some() || self.status >= 400
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl HttpResponseChunk {
    pub fn new(status: u16, headers: HashMap<String, String>, body: Vec<u8>, url: String) -> Self {
        Self {
            status,
            headers,
            body,
            url,
            error_message: None,
        }
    }

    pub fn is_success(&self) -> bool {
        !self.is_error() && self.status >= 200 && self.status < 300
    }

    pub fn status_code(&self) -> StatusCode {
        StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }

    pub fn json<T>(&self) -> Result<T, serde_json::Error>
    where
        T: serde::de::DeserializeOwned,
    {
        serde_json::from_slice(&self.body)
    }

    /// Create a data chunk from bytes
    pub fn data(data: bytes::Bytes) -> Self {
        Self {
            status: 200,
            headers: HashMap::new(),
            body: data.to_vec(),
            url: String::new(),
            error_message: None,
        }
    }

    /// Create a connection chunk for H3 connections
    pub fn connection<T>(_conn: T) -> Self {
        Self {
            status: 200,
            headers: HashMap::new(),
            body: b"connection_established".to_vec(),
            url: String::new(),
            error_message: None,
        }
    }

    /// Create a response chunk from HTTP response
    pub fn from_response<T>(_response: T) -> Self {
        Self {
            status: 200,
            headers: HashMap::new(),
            body: b"response_received".to_vec(),
            url: String::new(),
            error_message: None,
        }
    }

    /// Create a HEAD response chunk (headers only, no body)
    pub fn head(status: u16, headers: HashMap<String, String>, url: String) -> Self {
        Self {
            status,
            headers,
            body: Vec::new(), // HEAD responses have no body
            url,
            error_message: None,
        }
    }
}

/// HTTP Request Chunk - for streaming request data
#[derive(Debug, Clone, Default)]
pub struct HttpRequestChunk {
    pub method: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub error_message: Option<String>,
}

impl MessageChunk for HttpRequestChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            method: String::new(),
            url: String::new(),
            headers: HashMap::new(),
            body: Vec::new(),
            error_message: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error_message.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

/// HTTP Download Chunk - for streaming file downloads
#[derive(Debug, Clone, Default)]
pub struct HttpDownloadChunk {
    pub chunk_id: usize,
    pub data: Vec<u8>,
    pub total_size: Option<u64>,
    pub bytes_downloaded: u64,
    pub url: String,
    pub error_message: Option<String>,
}

impl MessageChunk for HttpDownloadChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            chunk_id: 0,
            data: Vec::new(),
            total_size: None,
            bytes_downloaded: 0,
            url: String::new(),
            error_message: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error_message.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl HttpDownloadChunk {
    pub fn new(
        chunk_id: usize,
        data: Vec<u8>,
        total_size: Option<u64>,
        bytes_downloaded: u64,
        url: String,
    ) -> Self {
        Self {
            chunk_id,
            data,
            total_size,
            bytes_downloaded,
            url,
            error_message: None,
        }
    }

    pub fn progress_percentage(&self) -> Option<f64> {
        self.total_size.map(|total| {
            if total > 0 {
                (self.bytes_downloaded as f64 / total as f64) * 100.0
            } else {
                0.0
            }
        })
    }
}
