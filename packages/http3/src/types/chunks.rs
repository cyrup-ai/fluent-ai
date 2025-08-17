//! HTTP Response Chunk Types and MessageChunk Implementations
//!
//! This module provides the canonical HttpResponseChunk enum and MessageChunk trait
//! implementations for zero-allocation HTTP streaming with fluent_ai_async architecture.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use fluent_ai_async::prelude::MessageChunk;
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode, Version};

/// Canonical HTTP response chunk type for all streaming operations.
///
/// This enum represents different states and data types that can occur
/// during HTTP response streaming, following the error-as-data pattern
/// required by fluent_ai_async architecture.
#[derive(Debug, Clone)]
pub enum HttpResponseChunk {
    /// Initial response status and headers received
    Status {
        status: StatusCode,
        headers: HeaderMap,
        version: Version,
    },

    /// Response body data chunk
    Data { bytes: Bytes, is_final: bool },

    /// HTTP trailers received (for chunked encoding)
    Trailers { headers: HeaderMap },

    /// Response completed successfully
    Complete,

    /// Connection-level error occurred
    ConnectionError {
        message: Arc<str>,
        recoverable: bool,
    },

    /// Protocol-level error occurred  
    ProtocolError {
        message: Arc<str>,
        status_code: Option<StatusCode>,
    },

    /// Timeout error occurred
    TimeoutError {
        message: Arc<str>,
        operation: Arc<str>,
    },

    /// Generic error chunk (for MessageChunk trait compliance)
    Error { message: Arc<str> },
}

impl MessageChunk for HttpResponseChunk {
    #[inline]
    fn bad_chunk(error: String) -> Self {
        HttpResponseChunk::Error {
            message: Arc::from(error.as_str()),
        }
    }

    #[inline]
    fn error(&self) -> Option<&str> {
        match self {
            HttpResponseChunk::ConnectionError { message, .. } => Some(message.as_ref()),
            HttpResponseChunk::ProtocolError { message, .. } => Some(message.as_ref()),
            HttpResponseChunk::TimeoutError { message, .. } => Some(message.as_ref()),
            HttpResponseChunk::Error { message } => Some(message.as_ref()),
            _ => None,
        }
    }

    #[inline]
    fn is_error(&self) -> bool {
        matches!(
            self,
            HttpResponseChunk::ConnectionError { .. }
                | HttpResponseChunk::ProtocolError { .. }
                | HttpResponseChunk::TimeoutError { .. }
                | HttpResponseChunk::Error { .. }
        )
    }
}

impl Default for HttpResponseChunk {
    #[inline]
    fn default() -> Self {
        HttpResponseChunk::Error {
            message: Arc::from("Default HttpResponseChunk"),
        }
    }
}

impl HttpResponseChunk {
    /// Create a status chunk with zero-allocation header optimization
    #[inline]
    pub fn status(status: StatusCode, headers: HeaderMap, version: Version) -> Self {
        HttpResponseChunk::Status {
            status,
            headers,
            version,
        }
    }

    /// Create a data chunk with final flag
    #[inline]
    pub fn data(bytes: Bytes, is_final: bool) -> Self {
        HttpResponseChunk::Data { bytes, is_final }
    }

    /// Create a trailers chunk
    #[inline]
    pub fn trailers(headers: HeaderMap) -> Self {
        HttpResponseChunk::Trailers { headers }
    }

    /// Create a completion marker
    #[inline]
    pub fn complete() -> Self {
        HttpResponseChunk::Complete
    }

    /// Create a connection error with recovery hint
    #[inline]
    pub fn connection_error(message: impl Into<Arc<str>>, recoverable: bool) -> Self {
        HttpResponseChunk::ConnectionError {
            message: message.into(),
            recoverable,
        }
    }

    /// Create a protocol error with optional status code
    #[inline]
    pub fn protocol_error(message: impl Into<Arc<str>>, status_code: Option<StatusCode>) -> Self {
        HttpResponseChunk::ProtocolError {
            message: message.into(),
            status_code,
        }
    }

    /// Create a timeout error with operation context
    #[inline]
    pub fn timeout_error(message: impl Into<Arc<str>>, operation: impl Into<Arc<str>>) -> Self {
        HttpResponseChunk::TimeoutError {
            message: message.into(),
            operation: operation.into(),
        }
    }

    /// Check if this chunk represents response completion
    #[inline]
    pub fn is_complete(&self) -> bool {
        matches!(self, HttpResponseChunk::Complete)
    }

    /// Check if this chunk contains response data
    #[inline]
    pub fn is_data(&self) -> bool {
        matches!(self, HttpResponseChunk::Data { .. })
    }

    /// Check if this chunk contains status information
    #[inline]
    pub fn is_status(&self) -> bool {
        matches!(self, HttpResponseChunk::Status { .. })
    }

    /// Check if this chunk contains trailers
    #[inline]
    pub fn is_trailers(&self) -> bool {
        matches!(self, HttpResponseChunk::Trailers { .. })
    }

    /// Extract data bytes if this is a data chunk
    #[inline]
    pub fn data_bytes(&self) -> Option<&Bytes> {
        match self {
            HttpResponseChunk::Data { bytes, .. } => Some(bytes),
            _ => None,
        }
    }

    /// Extract status code if this is a status chunk
    #[inline]
    pub fn status_code(&self) -> Option<StatusCode> {
        match self {
            HttpResponseChunk::Status { status, .. } => Some(*status),
            HttpResponseChunk::ProtocolError { status_code, .. } => *status_code,
            _ => None,
        }
    }

    /// Extract headers if this chunk contains them
    #[inline]
    pub fn headers(&self) -> Option<&HeaderMap> {
        match self {
            HttpResponseChunk::Status { headers, .. } => Some(headers),
            HttpResponseChunk::Trailers { headers } => Some(headers),
            _ => None,
        }
    }

    /// Check if this is a recoverable error
    #[inline]
    pub fn is_recoverable(&self) -> bool {
        match self {
            HttpResponseChunk::ConnectionError { recoverable, .. } => *recoverable,
            HttpResponseChunk::TimeoutError { .. } => true, // Timeouts are generally recoverable
            _ => false,
        }
    }
}

/// Zero-allocation helper for creating HeaderMap from iterator
///
/// This function provides an ergonomic way to create HeaderMaps
/// without unnecessary allocations during streaming operations.
#[inline]
pub fn headers_from_iter<I, K, V>(iter: I) -> Result<HeaderMap, http::Error>
where
    I: IntoIterator<Item = (K, V)>,
    K: TryInto<HeaderName>,
    V: TryInto<HeaderValue>,
    K::Error: Into<http::Error>,
    V::Error: Into<http::Error>,
{
    let mut headers = HeaderMap::new();
    for (key, value) in iter {
        let name = key.try_into().map_err(Into::into)?;
        let val = value.try_into().map_err(Into::into)?;
        headers.insert(name, val);
    }
    Ok(headers)
}

/// Zero-allocation helper for merging HeaderMaps
///
/// This function efficiently merges two HeaderMaps without
/// unnecessary cloning during streaming operations.
#[inline]
pub fn merge_headers(mut base: HeaderMap, additional: HeaderMap) -> HeaderMap {
    for (key, value) in additional {
        if let Some(key) = key {
            base.insert(key, value);
        }
    }
    base
}
