//! HTTP Request and Download Chunk Types
//!
//! This module contains non-response chunk types for HTTP operations.
//! The HttpResponseChunk has been moved to src/types/chunks.rs.

use std::collections::HashMap;

use fluent_ai_async::prelude::*;

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
