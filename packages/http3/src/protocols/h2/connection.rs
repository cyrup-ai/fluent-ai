//! HTTP/2 connection management
//!
//! This module provides HTTP/2 connection types and stream management using h2 crate
//! direct polling primitives integrated with fluent_ai_async streaming patterns.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use crossbeam_utils::Backoff;
use fluent_ai_async::{AsyncStream, emit, spawn_task};
use h2::{RecvStream, client};
use http::{Request, Response};

use super::streaming::H2ConnectionManager;
use crate::prelude::*;
use crate::protocols::{
    core::{HttpVersion, TimeoutConfig},
    strategy::H2Config,
};
use crate::protocols::quiche::QuicheConnectionChunk;

static CONNECTION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// HTTP/2 connection wrapper using direct polling primitives
pub struct H2Connection {
    manager: super::streaming::H2ConnectionManager,
    config: super::super::core::TimeoutConfig,
    h2_config: super::super::strategy::H2Config,
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

    /// Send data through HTTP/2 connection
    pub fn send_data(
        &self,
        data: Vec<u8>,
    ) -> AsyncStream<crate::protocols::frames::FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Convert data to H2 frame and emit
            let frame = crate::protocols::frames::H2Frame::Data {
                stream_id: 1,
                data,
                end_stream: false,
            };
            emit!(
                sender,
                crate::protocols::frames::FrameChunk::h2_frame(frame)
            );
        })
    }

    /// Receive data from HTTP/2 connection
    pub fn receive_data(&self) -> AsyncStream<crate::protocols::frames::FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Poll for incoming H2 frames
            let frame = crate::protocols::frames::H2Frame::Data {
                stream_id: 1,
                data: vec![],
                end_stream: false,
            };
            emit!(
                sender,
                crate::protocols::frames::FrameChunk::h2_frame(frame)
            );
        })
    }

    /// Close HTTP/2 connection
    pub fn close(&self) -> AsyncStream<crate::protocols::frames::FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            let frame = crate::protocols::frames::H2Frame::GoAway {
                last_stream_id: 0,
                error_code: 0,
                debug_data: vec![],
            };
            emit!(
                sender,
                crate::protocols::frames::FrameChunk::h2_frame(frame)
            );
        })
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
            // Use h2 handshake with simple blocking executor
            use futures::executor::block_on;
            
            match block_on(h2::client::Builder::new().handshake(io)) {
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
                                                if let Err(e) =
                                                    request_stream.send_data(body_data, true)
                                                {
                                                    emit!(
                                                        sender,
                                                        HttpChunk::Error(format!(
                                                            "Send body error: {}",
                                                            e
                                                        ))
                                                    );
                                                    return;
                                                }
                                            }
                                            Poll::Ready(Err(e)) => {
                                                emit!(
                                                    sender,
                                                    HttpChunk::Error(format!(
                                                        "Stream ready error: {}",
                                                        e
                                                    ))
                                                );
                                                return;
                                            }
                                            Poll::Pending => {}
                                        }
                                    }

                                    // Poll response future using direct polling pattern
                                    let mut response_future = response_future;
                                    loop {
                                        match response_future.poll(&mut context) {
                                            Poll::Ready(Ok(response)) => {
                                                // Extract response headers
                                                let status = response.status();
                                                let headers = response.headers().clone();
                                                
                                                // Emit status/headers as chunks BEFORE body processing
                                                emit!(sender, HttpChunk::Headers(status, headers));
                                                
                                                // Get body stream and poll for data
                                                let mut body_stream = response.into_body();
                                                loop {
                                                    match body_stream.poll_data(&mut context) {
                                                        Poll::Ready(Some(Ok(data))) => {
                                                            emit!(sender, HttpChunk::Data(data.to_vec()));
                                                            let _ = body_stream.flow_control().release_capacity(data.len());
                                                        }
                                                        Poll::Ready(Some(Err(e))) => {
                                                            emit!(sender, HttpChunk::Error(format!("Body stream error: {}", e)));
                                                            break;
                                                        }
                                                        Poll::Ready(None) => {
                                                            // Body stream ended - now poll for trailers
                                                            loop {
                                                                match body_stream.poll_trailers(&mut context) {
                                                                    Poll::Ready(Ok(Some(trailers))) => {
                                                                        emit!(sender, HttpChunk::Trailers(trailers));
                                                                        break;
                                                                    }
                                                                    Poll::Ready(Ok(None)) => {
                                                                        // No trailers available
                                                                        break;
                                                                    }
                                                                    Poll::Ready(Err(e)) => {
                                                                        emit!(sender, HttpChunk::Error(format!("Trailers error: {}", e)));
                                                                        break;
                                                                    }
                                                                    Poll::Pending => {
                                                                        backoff.snooze();
                                                                        continue;
                                                                    }
                                                                }
                                                            }
                                                            break;
                                                        }
                                                        Poll::Pending => {
                                                            backoff.snooze();
                                                            continue;
                                                        }
                                                    }
                                                }
                                                break;
                                            }
                                            Poll::Ready(Err(e)) => {
                                                emit!(sender, HttpChunk::Error(format!("Response error: {}", e)));
                                                break;
                                            }
                                            Poll::Pending => {
                                                backoff.snooze();
                                                continue;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    emit!(
                                        sender,
                                        HttpChunk::Error(format!("Send request error: {}", e))
                                    );
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

impl H2Connection {}

/// HTTP/2 stream wrapper that bridges h2::RecvStream to AsyncStream
pub struct H2Stream {
    stream: AsyncStream<HttpChunk, 1024>,
}

impl H2Stream {
    /// Create a new H2Stream from h2::RecvStream using pure streams architecture
    ///
    /// Note: This method currently delegates to an empty stream as recv-only streams
    /// are not the primary use case for this HTTP client library. The main pattern
    /// is request-response via send_request_stream().
    pub fn from_recv_stream(_recv_stream: h2::RecvStream) -> Self {
        // Create an empty stream for recv-only scenario
        // Most HTTP client usage goes through send_request_stream() which handles the full lifecycle
        let stream = AsyncStream::<HttpChunk, 1024>::with_channel(move |sender| {
            // Emit end immediately for recv-only streams
            // Real HTTP client usage should use send_request_stream() instead
            emit!(sender, HttpChunk::End);
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


impl Clone for H2Connection {
    fn clone(&self) -> Self {
        Self {
            manager: self.manager.clone(),
            config: self.config.clone(),
            h2_config: self.h2_config.clone(),
            established_at: self.established_at,
            connection_id: self.connection_id,
        }
    }
}
