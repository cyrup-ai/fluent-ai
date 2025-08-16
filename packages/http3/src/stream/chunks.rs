//! HTTP streaming chunk types - Pure streams-first architecture
//! Zero futures, zero Result wrapping - all AsyncStream based

use bytes::Bytes;
use http::{HeaderMap, StatusCode};

/// Represents a chunk of an HTTP response stream.
#[derive(Debug, Clone)]
pub enum HttpChunk {
    /// The head of the response, containing status and headers.
    Head(StatusCode, HeaderMap),
    /// A chunk of the response body.
    Body(Bytes),
    /// Deserialized data from response body for pattern matching
    Deserialized(serde_json::Value),
    /// Error that occurred during processing - stores error message directly for fast access
    Error(String),
}

impl Default for HttpChunk {
    fn default() -> Self {
        HttpChunk::Head(StatusCode::OK, HeaderMap::new())
    }
}

impl cyrup_sugars::prelude::MessageChunk for HttpChunk {
    fn bad_chunk(error: String) -> Self {
        HttpChunk::Error(error)
    }

    fn is_error(&self) -> bool {
        matches!(self, HttpChunk::Error(_))
    }

    fn error(&self) -> Option<&str> {
        match self {
            HttpChunk::Error(msg) => Some(msg),
            _ => None,
        }
    }
}

impl fluent_ai_async::prelude::MessageChunk for HttpChunk {
    fn bad_chunk(error: String) -> Self {
        HttpChunk::Error(error)
    }

    fn error(&self) -> Option<&str> {
        match self {
            HttpChunk::Error(msg) => Some(msg),
            _ => None,
        }
    }
}

impl serde::Serialize for HttpChunk {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        match self {
            HttpChunk::Head(status, headers) => {
                let mut state = serializer.serialize_struct("HttpChunk", 3)?;
                state.serialize_field("type", "head")?;
                state.serialize_field("status", &status.as_u16())?;

                // Convert HeaderMap to serializable format
                let headers_map: std::collections::HashMap<String, String> = headers
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();
                state.serialize_field("headers", &headers_map)?;
                state.end()
            }
            HttpChunk::Body(bytes) => {
                let mut state = serializer.serialize_struct("HttpChunk", 2)?;
                state.serialize_field("type", "body")?;
                state.serialize_field("length", &bytes.len())?;
                state.end()
            }
            HttpChunk::Deserialized(value) => {
                let mut state = serializer.serialize_struct("HttpChunk", 2)?;
                state.serialize_field("type", "deserialized")?;
                state.serialize_field("data", value)?;
                state.end()
            }
            HttpChunk::Error(message) => {
                let mut state = serializer.serialize_struct("HttpChunk", 2)?;
                state.serialize_field("type", "error")?;
                state.serialize_field("message", message)?;
                state.end()
            }
        }
    }
}

/// Represents a bad/error chunk in HTTP response processing.
#[derive(Debug, Clone)]
pub enum BadChunk {
    /// An error chunk containing the HTTP error that occurred
    Error(crate::HttpError),
    /// A chunk that failed processing with context
    ProcessingFailed {
        /// The original error
        error: crate::HttpError,
        /// Additional context about the processing failure
        context: String,
    },
}

impl BadChunk {
    /// Create a BadChunk from an HttpError
    pub fn from_err(error: crate::HttpError) -> Self {
        BadChunk::Error(error)
    }

    /// Create a BadChunk from an HttpError with processing context
    pub fn from_processing_error(error: crate::HttpError, context: String) -> Self {
        BadChunk::ProcessingFailed { error, context }
    }

    /// Get the underlying HttpError
    pub fn error(&self) -> &crate::HttpError {
        match self {
            BadChunk::Error(err) => err,
            BadChunk::ProcessingFailed { error, .. } => error,
        }
    }
}

/// Represents a chunk of a file download stream.
#[derive(Debug, Clone)]
pub struct DownloadChunk {
    /// Raw bytes of this chunk
    pub data: Bytes,
    /// Sequential number of this chunk
    pub chunk_number: u64,
    /// Total file size if known from headers
    pub total_size: Option<u64>,
    /// Total bytes downloaded so far
    pub bytes_downloaded: u64,
    /// Error message if this is an error chunk
    pub error_message: Option<String>,
}

impl Default for DownloadChunk {
    fn default() -> Self {
        DownloadChunk {
            data: Bytes::new(),
            chunk_number: 0,
            total_size: None,
            bytes_downloaded: 0,
            error_message: None,
        }
    }
}

impl DownloadChunk {
    /// Calculate progress percentage if total size is known
    pub fn progress_percentage(&self) -> Option<f64> {
        self.total_size.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.bytes_downloaded as f64 / total as f64) * 100.0
            }
        })
    }

    /// Calculate download speed in bytes per second (requires timing context)
    pub fn download_speed(&self, elapsed_seconds: f64) -> Option<f64> {
        if elapsed_seconds > 0.0 {
            Some(self.bytes_downloaded as f64 / elapsed_seconds)
        } else {
            None
        }
    }
}

impl cyrup_sugars::prelude::MessageChunk for DownloadChunk {
    fn bad_chunk(error: String) -> Self {
        DownloadChunk {
            data: bytes::Bytes::new(),
            chunk_number: 0,
            total_size: None,
            bytes_downloaded: 0,
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

impl fluent_ai_async::prelude::MessageChunk for DownloadChunk {
    fn bad_chunk(error: String) -> Self {
        DownloadChunk {
            data: bytes::Bytes::new(),
            chunk_number: 0,
            total_size: None,
            bytes_downloaded: 0,
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

/// Server-Sent Event data
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// Event type
    pub event_type: Option<String>,
    /// Event data
    pub data: String,
    /// Event ID
    pub id: Option<String>,
    /// Retry timeout
    pub retry: Option<u64>,
}
