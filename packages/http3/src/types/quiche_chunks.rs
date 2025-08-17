//! Quiche QUIC MessageChunk Implementations
//!
//! Provides MessageChunk trait implementations for all Quiche streaming types,
//! following fluent_ai_async error-as-data architecture patterns.
//!
//! ## Architecture
//! - All chunk types implement MessageChunk trait with bad_chunk() method
//! - Error-as-data pattern: errors are data, not exceptions
//! - Zero-allocation hot paths with pre-allocated buffers
//! - Direct integration with AsyncStream<T, CAP> producers

use std::net::SocketAddr;

use fluent_ai_async::prelude::MessageChunk;

/// Quiche packet processing chunk for UDP packet operations
#[derive(Debug, Clone)]
pub struct QuichePacketChunk {
    pub bytes_processed: usize,
    pub from_addr: Option<SocketAddr>,
    pub to_addr: Option<SocketAddr>,
    pub error: Option<String>,
}

impl QuichePacketChunk {
    #[inline]
    pub fn packet_processed(bytes: usize, from: SocketAddr, to: SocketAddr) -> Self {
        Self {
            bytes_processed: bytes,
            from_addr: Some(from),
            to_addr: Some(to),
            error: None,
        }
    }

    #[inline]
    pub fn connection_established(from: SocketAddr, to: SocketAddr) -> Self {
        Self {
            bytes_processed: 0,
            from_addr: Some(from),
            to_addr: Some(to),
            error: None,
        }
    }
}

impl MessageChunk for QuichePacketChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            bytes_processed: 0,
            from_addr: None,
            to_addr: None,
            error: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error.as_ref().map(|s| s.as_str())
    }
}

/// Quiche stream data chunk for HTTP/3 stream operations
#[derive(Debug, Clone)]
pub struct QuicheStreamChunk {
    pub data: Vec<u8>,
    pub stream_id: u64,
    pub fin: bool,
    pub is_complete: bool,
    pub error: Option<String>,
}

impl QuicheStreamChunk {
    #[inline]
    pub fn data_chunk(stream_id: u64, data: Vec<u8>, fin: bool) -> Self {
        Self {
            data,
            stream_id,
            fin,
            is_complete: fin,
            error: None,
        }
    }

    #[inline]
    pub fn stream_complete(stream_id: u64) -> Self {
        Self {
            data: Vec::new(),
            stream_id,
            fin: true,
            is_complete: true,
            error: None,
        }
    }

    #[inline]
    pub fn readable_stream(stream_id: u64) -> Self {
        Self {
            data: Vec::new(),
            stream_id,
            fin: false,
            is_complete: false,
            error: None,
        }
    }
}

impl MessageChunk for QuicheStreamChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            data: Vec::new(),
            stream_id: 0,
            fin: false,
            is_complete: false,
            error: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error.as_ref().map(|s| s.as_str())
    }
}

/// Quiche write result chunk for stream writing operations
#[derive(Debug, Clone)]
pub struct QuicheWriteResult {
    pub bytes_written: usize,
    pub stream_id: u64,
    pub is_complete: bool,
    pub fin_sent: bool,
    pub error: Option<String>,
}

impl QuicheWriteResult {
    #[inline]
    pub fn bytes_written(stream_id: u64, bytes: usize) -> Self {
        Self {
            bytes_written: bytes,
            stream_id,
            is_complete: false,
            fin_sent: false,
            error: None,
        }
    }

    #[inline]
    pub fn write_complete(stream_id: u64, total_bytes: usize) -> Self {
        Self {
            bytes_written: total_bytes,
            stream_id,
            is_complete: true,
            fin_sent: true,
            error: None,
        }
    }

    #[inline]
    pub fn stream_reset(stream_id: u64) -> Self {
        Self {
            bytes_written: 0,
            stream_id,
            is_complete: true,
            fin_sent: false,
            error: None,
        }
    }
}

impl MessageChunk for QuicheWriteResult {
    fn bad_chunk(error: String) -> Self {
        Self {
            bytes_written: 0,
            stream_id: 0,
            is_complete: false,
            fin_sent: false,
            error: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error.as_ref().map(|s| s.as_str())
    }
}

/// Quiche connection state chunk for connection lifecycle events
#[derive(Debug, Clone)]
pub struct QuicheConnectionChunk {
    pub local_addr: Option<SocketAddr>,
    pub peer_addr: Option<SocketAddr>,
    pub is_established: bool,
    pub is_closed: bool,
    pub timeout_ms: Option<u64>,
    pub error: Option<String>,
}

impl QuicheConnectionChunk {
    #[inline]
    pub fn established(local: SocketAddr, peer: SocketAddr) -> Self {
        Self {
            local_addr: Some(local),
            peer_addr: Some(peer),
            is_established: true,
            is_closed: false,
            timeout_ms: None,
            error: None,
        }
    }

    #[inline]
    pub fn connection_closed(local: SocketAddr, peer: SocketAddr) -> Self {
        Self {
            local_addr: Some(local),
            peer_addr: Some(peer),
            is_established: false,
            is_closed: true,
            timeout_ms: None,
            error: None,
        }
    }

    #[inline]
    pub fn timeout_event(timeout_ms: u64) -> Self {
        Self {
            local_addr: None,
            peer_addr: None,
            is_established: false,
            is_closed: false,
            timeout_ms: Some(timeout_ms),
            error: None,
        }
    }
}

impl MessageChunk for QuicheConnectionChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            local_addr: None,
            peer_addr: None,
            is_established: false,
            is_closed: false,
            timeout_ms: None,
            error: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error.as_ref().map(|s| s.as_str())
    }
}

/// Quiche readable streams chunk for stream iteration
#[derive(Debug, Clone)]
pub struct QuicheReadableChunk {
    pub readable_streams: Vec<u64>,
    pub writable_streams: Vec<u64>,
    pub connection_active: bool,
    pub error: Option<String>,
}

impl QuicheReadableChunk {
    #[inline]
    pub fn readable_stream(stream_id: u64) -> Self {
        Self {
            readable_streams: vec![stream_id],
            writable_streams: Vec::new(),
            connection_active: true,
            error: None,
        }
    }

    #[inline]
    pub fn writable_stream(stream_id: u64) -> Self {
        Self {
            readable_streams: Vec::new(),
            writable_streams: vec![stream_id],
            connection_active: true,
            error: None,
        }
    }

    #[inline]
    pub fn connection_closed() -> Self {
        Self {
            readable_streams: Vec::new(),
            writable_streams: Vec::new(),
            connection_active: false,
            error: None,
        }
    }

    #[inline]
    pub fn streams_available(readable: Vec<u64>, writable: Vec<u64>) -> Self {
        Self {
            readable_streams: readable,
            writable_streams: writable,
            connection_active: true,
            error: None,
        }
    }
}

impl MessageChunk for QuicheReadableChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            readable_streams: Vec::new(),
            writable_streams: Vec::new(),
            connection_active: false,
            error: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error.as_ref().map(|s| s.as_str())
    }
}
