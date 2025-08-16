//! Core types and traits for H3 client connection
//!
//! Defines H3Connection wrapper and MessageChunk implementation
//! for HTTP/3 connection handling with error management.

use std::sync::Arc;

use bytes::Bytes;
use fluent_ai_async::prelude::MessageChunk;
use h3::client::SendRequest;
use h3_quinn::OpenStreams;

// Simplified types to avoid missing dependencies
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;
pub type PoolClient = ();
pub type DynResolver = ();

/// Custom H3 connection wrapper to avoid Default trait bound issues
pub struct H3Connection {
    pub connection: Option<h3::client::Connection<h3_quinn::Connection, Bytes>>,
    pub send_request: Option<SendRequest<OpenStreams, Bytes>>,
    pub error_message: Option<String>,
}

impl MessageChunk for H3Connection {
    fn bad_chunk(error: String) -> Self {
        Self {
            connection: None,
            send_request: None,
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

impl Default for H3Connection {
    fn default() -> Self {
        Self {
            connection: None,
            send_request: None,
            error_message: None,
        }
    }
}
