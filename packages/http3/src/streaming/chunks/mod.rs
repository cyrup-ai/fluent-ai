//! Streaming chunk types for HTTP responses

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fluent_ai_async::prelude::MessageChunk;

static CHUNK_COUNTER: AtomicU64 = AtomicU64::new(0);

/// HTTP response chunk
#[derive(Debug, Clone)]
pub enum HttpChunk {
    Data { data: Vec<u8> },
    Headers { headers: Vec<(String, String)> },
    Status { code: u16, reason: String },
    Trailer { headers: Vec<(String, String)> },
    Error { message: Arc<str> },
}

impl MessageChunk for HttpChunk {
    #[inline]
    fn bad_chunk(error_message: String) -> Self {
        Self::Error {
            message: Arc::from(error_message),
        }
    }

    #[inline]
    fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    #[inline]
    fn error(&self) -> Option<&str> {
        match self {
            Self::Error { message } => Some(message),
            _ => None,
        }
    }
}

impl Default for HttpChunk {
    fn default() -> Self {
        Self::Error {
            message: Arc::from("Default chunk"),
        }
    }
}
/// Download chunk for file downloads
#[derive(Debug, Clone)]
pub enum DownloadChunk {
    Data {
        bytes: Vec<u8>,
        offset: u64,
    },
    Progress {
        bytes_downloaded: u64,
        total_bytes: Option<u64>,
    },
    Complete {
        total_bytes: u64,
    },
    Error {
        message: Arc<str>,
    },
}

impl MessageChunk for DownloadChunk {
    #[inline]
    fn bad_chunk(error_message: String) -> Self {
        Self::Error {
            message: Arc::from(error_message),
        }
    }

    #[inline]
    fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    #[inline]
    fn error(&self) -> Option<&str> {
        match self {
            Self::Error { message } => Some(message),
            _ => None,
        }
    }
}

impl Default for DownloadChunk {
    fn default() -> Self {
        Self::Error {
            message: Arc::from("Default download chunk"),
        }
    }
}

/// Server-Sent Events chunk
#[derive(Debug, Clone)]
pub enum SseEvent {
    Data { data: String },
    Event { event_type: String, data: String },
    Id { id: String },
    Retry { retry_ms: u64 },
    Comment { comment: String },
    Error { message: Arc<str> },
}

impl MessageChunk for SseEvent {
    #[inline]
    fn bad_chunk(error_message: String) -> Self {
        Self::Error {
            message: Arc::from(error_message),
        }
    }

    #[inline]
    fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    #[inline]
    fn error(&self) -> Option<&str> {
        match self {
            Self::Error { message } => Some(message),
            _ => None,
        }
    }
}

impl Default for SseEvent {
    fn default() -> Self {
        Self::Error {
            message: Arc::from("Default SSE event"),
        }
    }
}

/// Bad chunk for error handling
#[derive(Debug, Clone)]
pub struct BadChunk {
    pub message: Arc<str>,
    pub chunk_id: u64,
}

impl MessageChunk for BadChunk {
    #[inline]
    fn bad_chunk(error_message: String) -> Self {
        let chunk_id = CHUNK_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            message: Arc::from(error_message),
            chunk_id,
        }
    }

    #[inline]
    fn is_error(&self) -> bool {
        true
    }

    #[inline]
    fn error(&self) -> Option<&str> {
        Some(&self.message)
    }
}

impl Default for BadChunk {
    fn default() -> Self {
        Self::bad_chunk("Default bad chunk".to_string())
    }
}

// Re-export frame types for compatibility
pub use crate::protocols::frames::{FrameChunk, H2Frame, H3Frame};
