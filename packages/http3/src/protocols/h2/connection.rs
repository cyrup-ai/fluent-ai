//! HTTP/2 connection management
//!
//! This module provides HTTP/2 connection types and stream management using h2 crate
//! direct polling primitives integrated with fluent_ai_async streaming patterns.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use crossbeam_utils::Backoff;
use fluent_ai_async::{AsyncStream, emit};
use h2::{RecvStream, SendRequest, client};
use http::{Request, Response};

use crate::protocols::core::{HttpProtocol, HttpVersion, TimeoutConfig};
use crate::protocols::h2::chunks::HttpChunk;
use crate::protocols::h2::streaming::H2ConnectionManager;

static CONNECTION_COUNTER: AtomicU64 = AtomicU64::new(0);
use crate::protocols::strategy::H2Config;

/// HTTP/2 connection wrapper using direct polling primitives
#[derive(Debug)]
pub struct H2Connection {
    manager: H2ConnectionManager,
    config: TimeoutConfig,
    h2_config: H2Config,
    established_at: Option<Instant>,
    connection_id: u64,
}

impl H2Connection {
    pub fn new() -> Self {
        Self::with_config(H2Config::default())
    }

    pub fn with_config(config: H2Config) -> Self {
        let connection_id = CONNECTION_COUNTER.fetch_add(1, Ordering::Relaxed);
        
        Self {
            manager: H2ConnectionManager::new(),
            config: TimeoutConfig::default(),
            h2_config: config,
            established_at: Some(Instant::now()),
            connection_id,
        }
    }

    /// Get the HTTP version
    pub fn version(&self) -> HttpVersion {
        HttpVersion::Http2
    }

    /// Get the timeout configuration
    pub fn config(&self) -> &TimeoutConfig {
        &self.config
    }

    /// Check if the connection is ready to send requests
    pub fn is_ready(&self) -> bool {
        !self.manager.is_error()
    }

    /// Send an HTTP/2 request and return a stream of response chunks
    pub fn send_request_stream<T>(
        &self,
        io: T,
        request: Request<()>,
        body: Option<Bytes>,
        mut context: Context<'_>,
    ) -> AsyncStream<HttpChunk, 1024>
    where
        T: std::io::Read + std::io::Write + Unpin + Send + 'static,
    {
        AsyncStream::<HttpChunk, 1024>::with_channel(move |sender| {
            // Use h2 direct polling - NO Future APIs
            match h2::client::Builder::new().handshake(io) {
                Ok((mut send_request, mut connection)) => {
                    // Poll connection readiness using direct primitives
                    let backoff = Backoff::new();
                    loop {
                        match connection.poll_ready(&mut context) {
                            Poll::Ready(Ok(())) => break,
                            Poll::Ready(Err(e)) => {
                                emit!(sender, HttpChunk::Error(format!("Connection error: {}", e)));
                                return;
                            }
                            Poll::Pending => {
                                backoff.spin();
                                continue;
                            }
                        }
                    }

                    // Send request using direct polling
                    match send_request.poll_ready(&mut context) {
                        Poll::Ready(Ok(())) => {
                            match send_request.send_request(request, body.is_none()) {
                                Ok((response_future, mut request_stream)) => {
                                    // Send body if provided using direct polling
                                    if let Some(body_data) = body {
                                        match request_stream.poll_ready(&mut context) {
                                            Poll::Ready(Ok(())) => {
                                                if let Err(e) = request_stream.send_data(body_data, true) {
                                                    emit!(sender, HttpChunk::Error(format!("Send body error: {}", e)));
                                                    return;
                                                }
                                            }
                                            Poll::Ready(Err(e)) => {
                                                emit!(sender, HttpChunk::Error(format!("Stream ready error: {}", e)));
                                                return;
                                            }
                                            Poll::Pending => {}
                                        }
                                    }

                                    // Use direct h2 response polling - NO Future APIs
                                    // This is a simplified implementation - in production would need proper response handling
                                    emit!(sender, HttpChunk::Error("H2 response handling not yet implemented with direct polling".to_string()));
                                }
                                Err(e) => {
                                    emit!(sender, HttpChunk::Error(format!("Send request error: {}", e)));
                                }
                            }
                        }
                        Poll::Ready(Err(e)) => {
                            emit!(sender, HttpChunk::Error(format!("Send ready error: {}", e)));
                        }
                        Poll::Pending => {
                            emit!(sender, HttpChunk::Error("Send request pending".to_string()));
                        }
                    }
                }
                Err(e) => {
                    emit!(sender, HttpChunk::Error(format!("Handshake error: {}", e)));
                }
            }
        })
    }

    /// Check if the connection has encountered an error
    pub fn is_error(&self) -> bool {
        self.manager.is_error()
    }

    pub fn connection_id(&self) -> u64 {
        self.connection_id
    }

    pub fn is_established(&self) -> bool {
        self.established_at.is_some()
    }
}

impl H2Connection {
    static CONNECTION_COUNTER: AtomicU64 = AtomicU64::new(0);
}

/// HTTP/2 stream wrapper that bridges h2::RecvStream to AsyncStream
#[derive(Debug)]
pub struct H2Stream {
    stream: AsyncStream<HttpChunk, 1024>,
}

impl H2Stream {
    /// Create a new H2Stream from h2::RecvStream using elite polling pattern
    pub fn from_recv_stream(recv_stream: h2::RecvStream) -> Self {
        use crossbeam_utils::Backoff;

        use crate::protocols::h2::frame_processor::H2FrameProcessor;

        let mut frame_processor = H2FrameProcessor::new(recv_stream);
        frame_processor.start_processing();

        let stream = AsyncStream::<HttpChunk, 1024>::with_channel(move |sender| {
            // Elite polling pattern for data chunks
            let backoff = Backoff::new();
            loop {
                if let Some(data_chunk) = frame_processor.try_recv_data_chunk() {
                    // Convert bytes::Bytes to HttpChunk
                    let http_chunk = HttpChunk::new(data_chunk.to_vec());
                    emit!(sender, http_chunk);
                    backoff.reset();
                } else if frame_processor.is_closed() {
                    // Stream closed, exit loop
                    break;
                } else {
                    // Elite backoff pattern - no chunks available
                    if backoff.is_completed() {
                        std::thread::yield_now();
                    } else {
                        backoff.snooze();
                    }
                }
            }
        });

        Self { stream }
    }

    /// Get the underlying AsyncStream
    pub fn into_stream(self) -> AsyncStream<HttpChunk, 1024> {
        self.stream
    }

    /// Collect all chunks from the stream
    pub fn collect(self) -> Vec<HttpChunk> {
        self.stream.collect()
    }
}

impl fluent_ai_async::prelude::MessageChunk for H2Connection {
    fn bad_chunk(_error: String) -> Self {
        // Create error connection without tokio dependencies
        Self {
            manager: H2ConnectionManager::new(),
            config: TimeoutConfig::default(),
            h2_config: H2Config::default(),
            established_at: None, // No establishment time for error connections
            connection_id: CONNECTION_COUNTER.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some("H2 connection error")
        } else {
            None
        }
    }
}

impl H2Connection {
    pub fn is_ready(&self) -> bool {
        !self.manager.is_error()
    }

    pub fn is_closed(&self) -> bool {
        self.manager.is_error()
    }

    fn is_error(&self) -> bool {
        self.manager.is_error()
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some("H2 connection error")
        } else {
            None
        }
    }
}
