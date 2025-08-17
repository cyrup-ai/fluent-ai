//! QUICHE protocol chunk types

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fluent_ai_async::prelude::MessageChunk;

static QUICHE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// QUICHE connection chunk for connection events
#[derive(Debug, Clone)]
pub enum QuicheConnectionChunk {
    Connected {
        connection_id: u64,
    },
    Disconnected {
        connection_id: u64,
        reason: String,
    },
    HandshakeComplete {
        connection_id: u64,
    },
    StreamOpened {
        connection_id: u64,
        stream_id: u64,
    },
    StreamClosed {
        connection_id: u64,
        stream_id: u64,
    },
    Error {
        connection_id: u64,
        message: Arc<str>,
    },
}

impl MessageChunk for QuicheConnectionChunk {
    #[inline]
    fn bad_chunk(error_message: String) -> Self {
        let connection_id = QUICHE_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self::Error {
            connection_id,
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
            Self::Error { message, .. } => Some(message),
            _ => None,
        }
    }
}

impl Default for QuicheConnectionChunk {
    fn default() -> Self {
        Self::bad_chunk("Default connection chunk".to_string())
    }
}
/// QUICHE packet chunk for packet-level events
#[derive(Debug, Clone)]
pub enum QuichePacketChunk {
    Received { packet_id: u64, size: usize },
    Sent { packet_id: u64, size: usize },
    Lost { packet_id: u64 },
    Acked { packet_id: u64 },
    Error { packet_id: u64, message: Arc<str> },
}

impl MessageChunk for QuichePacketChunk {
    #[inline]
    fn bad_chunk(error_message: String) -> Self {
        let packet_id = QUICHE_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self::Error {
            packet_id,
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
            Self::Error { message, .. } => Some(message),
            _ => None,
        }
    }
}

impl Default for QuichePacketChunk {
    fn default() -> Self {
        Self::bad_chunk("Default packet chunk".to_string())
    }
}

/// QUICHE stream chunk for stream-level events
#[derive(Debug, Clone)]
pub enum QuicheStreamChunk {
    Data { stream_id: u64, data: Vec<u8> },
    Opened { stream_id: u64 },
    Closed { stream_id: u64 },
    Reset { stream_id: u64, error_code: u64 },
    Error { stream_id: u64, message: Arc<str> },
}

impl MessageChunk for QuicheStreamChunk {
    #[inline]
    fn bad_chunk(error_message: String) -> Self {
        let stream_id = QUICHE_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self::Error {
            stream_id,
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
            Self::Error { message, .. } => Some(message),
            _ => None,
        }
    }
}

impl Default for QuicheStreamChunk {
    fn default() -> Self {
        Self::bad_chunk("Default stream chunk".to_string())
    }
}

/// Quiche readable chunk type alias
pub type QuicheReadableChunk = QuicheConnectionChunk;

/// Quiche write result type alias  
pub type QuicheWriteResult = QuichePacketChunk;
