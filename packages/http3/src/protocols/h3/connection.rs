//! HTTP/3 connection management
//!
//! This module provides HTTP/3 connection types and stream management using quiche
//! integrated with fluent_ai_async streaming patterns.

use std::sync::Arc;

use crossbeam_utils::Backoff;
use fluent_ai_async::prelude::*;

use crate::streaming::chunks::HttpChunk;
use crate::types::{HttpVersion, TimeoutConfig};

/// HTTP/3 connection wrapper that integrates quiche with fluent_ai_async
#[derive(Debug)]
pub struct H3Connection {
    inner: Arc<std::sync::Mutex<quiche::Connection>>,
    config: TimeoutConfig,
}

impl H3Connection {
    /// Create a new H3Connection from quiche connection
    pub fn new(connection: quiche::Connection, config: TimeoutConfig) -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(connection)),
            config,
        }
    }

    /// Get the HTTP version
    pub fn version(&self) -> HttpVersion {
        HttpVersion::Http3
    }

    /// Get the timeout configuration
    pub fn config(&self) -> &TimeoutConfig {
        &self.config
    }

    /// Send an HTTP/3 request and return a stream of response chunks
    pub fn send_request(
        &self,
        request: &[u8],
        stream_id: u64,
    ) -> Result<AsyncStream<HttpChunk, 1024>, quiche::Error> {
        let connection = Arc::clone(&self.inner);

        // Create AsyncStream using elite polling pattern
        let stream = AsyncStream::<HttpChunk, 1024>::with_channel(move |sender| {
            let mut buffer = [0; 65535];
            let backoff = Backoff::new();

            loop {
                let mut conn = connection.lock().unwrap();

                // Try to read response data
                match conn.stream_recv(stream_id, &mut buffer) {
                    Ok((len, fin)) => {
                        if len > 0 {
                            let http_chunk = HttpChunk::new(buffer[..len].to_vec());
                            emit!(sender, http_chunk);
                            backoff.reset();
                        }

                        if fin {
                            // Stream finished
                            break;
                        }
                    }
                    Err(quiche::Error::Done) => {
                        // Elite backoff pattern - no data available
                        if backoff.is_completed() {
                            std::thread::yield_now();
                        } else {
                            backoff.snooze();
                        }
                        continue;
                    }
                    Err(e) => {
                        let error_chunk =
                            HttpChunk::bad_chunk(format!("Quiche stream error: {}", e));
                        emit!(sender, error_chunk);
                        break;
                    }
                }

                // Check if connection is closed
                if conn.is_closed() {
                    break;
                }
            }
        });

        Ok(stream)
    }

    /// Check if the connection is closed
    pub fn is_closed(&self) -> bool {
        self.inner.lock().unwrap().is_closed()
    }
}

/// HTTP/3 stream wrapper that bridges quiche streams to AsyncStream
#[derive(Debug)]
pub struct H3Stream {
    stream_id: u64,
    connection: Arc<std::sync::Mutex<quiche::Connection>>,
}

impl H3Stream {
    /// Create a new H3Stream from quiche connection and stream ID
    pub fn new(stream_id: u64, connection: Arc<std::sync::Mutex<quiche::Connection>>) -> Self {
        Self {
            stream_id,
            connection,
        }
    }

    /// Convert to AsyncStream for fluent_ai_async integration
    pub fn into_stream(self) -> AsyncStream<HttpChunk, 1024> {
        let stream_id = self.stream_id;
        let connection = self.connection;

        AsyncStream::<HttpChunk, 1024>::with_channel(move |sender| {
            let mut buffer = [0; 65535];
            let backoff = Backoff::new();

            loop {
                let mut conn = connection.lock().unwrap();

                match conn.stream_recv(stream_id, &mut buffer) {
                    Ok((len, fin)) => {
                        if len > 0 {
                            let http_chunk = HttpChunk::new(buffer[..len].to_vec());
                            emit!(sender, http_chunk);
                            backoff.reset();
                        }

                        if fin {
                            break;
                        }
                    }
                    Err(quiche::Error::Done) => {
                        // Elite backoff pattern - no data available
                        if backoff.is_completed() {
                            std::thread::yield_now();
                        } else {
                            backoff.snooze();
                        }
                        continue;
                    }
                    Err(e) => {
                        let error_chunk = HttpChunk::bad_chunk(format!("H3 stream error: {}", e));
                        emit!(sender, error_chunk);
                        break;
                    }
                }

                if conn.is_closed() {
                    break;
                }
            }
        })
    }

    /// Collect all chunks from the stream
    pub fn collect(self) -> Vec<HttpChunk> {
        self.into_stream().collect()
    }
}

impl MessageChunk for H3Connection {
    fn bad_chunk(_error: String) -> Self {
        // Create a minimal quiche config for error cases
        let mut config = quiche::Config::new(quiche::PROTOCOL_VERSION).unwrap();
        config.set_application_protos(&[b"h3"]).unwrap();

        // Create a dummy connection that's immediately closed
        let scid = quiche::ConnectionId::from_ref(&[0; 16]);
        let local_addr = "127.0.0.1:0".parse().unwrap();
        let peer_addr = "127.0.0.1:0".parse().unwrap();

        let conn =
            quiche::connect(None, &scid, local_addr, peer_addr, &mut config).unwrap_or_else(|_| {
                // If that fails, create with different parameters
                let mut fallback_config = quiche::Config::new(quiche::PROTOCOL_VERSION).unwrap();
                quiche::connect(None, &scid, local_addr, peer_addr, &mut fallback_config).unwrap()
            });

        Self {
            inner: Arc::new(std::sync::Mutex::new(conn)),
            config: TimeoutConfig::default(),
        }
    }

    fn is_error(&self) -> bool {
        self.is_closed()
    }

    fn error(&self) -> Option<&str> {
        if self.is_closed() {
            Some("H3 connection closed")
        } else {
            None
        }
    }
}
