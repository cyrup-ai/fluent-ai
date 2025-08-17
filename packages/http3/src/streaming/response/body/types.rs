//! Core types and structures for HTTP response body processing
//!
//! Contains SSE event structures, JSON stream types, and core type definitions
//! for zero-allocation response body handling.

use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;

/// Server-Sent Event structure for streaming responses
/// Zero allocation design with unwrapped values
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// Event data payload
    pub data: Option<String>,
    /// Event type
    pub event_type: Option<String>,
    /// Event ID for last-event-id tracking
    pub id: Option<String>,
    /// Retry interval in milliseconds
    pub retry: Option<u64>,
}

impl SseEvent {
    /// Create new SSE event with data
    #[must_use]
    pub fn data(data: String) -> Self {
        Self {
            data: Some(data),
            event_type: None,
            id: None,
            retry: None,
        }
    }

    /// Create new SSE event with type and data
    #[must_use]
    pub fn typed(event_type: String, data: String) -> Self {
        Self {
            data: Some(data),
            event_type: Some(event_type),
            id: None,
            retry: None,
        }
    }
}

/// Type alias for the chunk handler function
pub type ChunkHandler<T> =
    Box<dyn FnMut(&T) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + 'static>;

/// JSON stream that yields unwrapped T values with user `on_chunk` error handling
/// Users get immediate values, error handling via `on_chunk` handler
pub struct JsonStream<T> {
    pub(super) body: Vec<u8>,
    pub(super) _phantom: PhantomData<T>,
    /// Optional handler for processing chunks
    #[allow(dead_code)]
    pub(super) handler: Option<ChunkHandler<T>>,
}

impl<T> Debug for JsonStream<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsonStream")
            .field("body", &self.body)
            .field(
                "handler",
                &if self.handler.is_some() {
                    "Some(<function>)"
                } else {
                    "None"
                },
            )
            .finish()
    }
}