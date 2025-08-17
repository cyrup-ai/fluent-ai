//! Stream processing wrappers for implementing MessageChunk trait
//! Includes HTTP chunks, download chunks, and streaming-related types

use fluent_ai_async::prelude::MessageChunk;

/// Wrapper for streaming operations
#[derive(Debug, Clone, Default)]
pub struct StreamWrapper<T> {
    pub data: Option<T>,
    pub error_message: Option<String>,
}

impl<T> MessageChunk for StreamWrapper<T> {
    fn bad_chunk(error: String) -> Self {
        Self {
            data: None,
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

/// Wrapper for HTTP chunks
#[derive(Debug, Clone)]
pub struct HttpChunkWrapper {
    pub chunk: crate::stream::HttpChunk,
}

impl MessageChunk for HttpChunkWrapper {
    fn bad_chunk(error: String) -> Self {
        Self {
            chunk: crate::stream::HttpChunk::Error(error),
        }
    }

    fn is_error(&self) -> bool {
        matches!(self.chunk, crate::stream::HttpChunk::Error(_))
    }

    fn error(&self) -> Option<&str> {
        match &self.chunk {
            crate::stream::HttpChunk::Error(_) => Some("HTTP chunk error"),
            _ => None,
        }
    }
}

impl Default for HttpChunkWrapper {
    fn default() -> Self {
        Self {
            chunk: crate::stream::HttpChunk::Body(bytes::Bytes::new()),
        }
    }
}

impl From<crate::stream::HttpChunk> for HttpChunkWrapper {
    fn from(chunk: crate::stream::HttpChunk) -> Self {
        Self { chunk }
    }
}

/// Wrapper for download chunks
#[derive(Debug, Clone)]
pub struct DownloadChunkWrapper {
    pub chunk: crate::stream::DownloadChunk,
}

impl MessageChunk for DownloadChunkWrapper {
    fn bad_chunk(error: String) -> Self {
        Self {
            chunk: crate::stream::DownloadChunk {
                data: bytes::Bytes::new(),
                chunk_number: 0,
                total_size: None,
                bytes_downloaded: 0,
                error_message: Some(error),
            },
        }
    }

    fn is_error(&self) -> bool {
        self.chunk.error_message.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.chunk.error_message.as_deref()
    }
}

impl Default for DownloadChunkWrapper {
    fn default() -> Self {
        Self {
            chunk: crate::stream::DownloadChunk::default(),
        }
    }
}

impl From<crate::stream::DownloadChunk> for DownloadChunkWrapper {
    fn from(chunk: crate::stream::DownloadChunk) -> Self {
        Self { chunk }
    }
}
