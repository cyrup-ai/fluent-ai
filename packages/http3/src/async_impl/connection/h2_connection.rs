//! H2 Connection Management - Direct Poll-Based Implementation
//!
//! Simple H2 connection management using direct poll-based primitives only,
//! following the exact specification requirements.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::task::{Context, Poll};

use fluent_ai_async::{AsyncStream, emit};
use h2::{client, server};

use crate::types::h2_chunks::{H2ConnectionChunk, H2DataChunk, H2RequestChunk, H2SendResult};

/// H2 Connection Manager with direct poll-based primitives only
///
/// Provides simple H2 connection management using only the patterns
/// specified in the task requirements.
pub struct H2ConnectionManager {
    is_connected: AtomicBool,
    connection_id: AtomicU64,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
}

impl H2ConnectionManager {
    /// Create a new H2 connection manager
    #[inline]
    pub const fn new() -> Self {
        Self {
            is_connected: AtomicBool::new(false),
            connection_id: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
        }
    }

    /// Create H2 connection establishment stream using direct poll-based primitives
    ///
    /// Integrates h2::client::handshake with direct polling
    /// within AsyncStream::with_channel pattern.
    #[inline]
    pub fn establish_connection_stream<T>(
        &self,
        io: T,
        polling_context: Context<'_>,
    ) -> AsyncStream<H2ConnectionChunk, 1024>
    where
        T: std::io::Read + std::io::Write + Unpin + Send + 'static,
    {
        let connection_id = self.connection_id.fetch_add(1, Ordering::SeqCst);
        let is_connected = &self.is_connected;

        AsyncStream::<H2ConnectionChunk, 1024>::with_channel(move |sender| {
            let mut context = polling_context;

            // Use h2's handshake with direct polling - NO Future-based APIs
            match h2::client::Builder::new().handshake(io) {
                Ok((send_request, connection)) => {
                    is_connected.store(true, Ordering::SeqCst);
                    emit!(sender, H2ConnectionChunk::ready());

                    // Monitor connection using direct poll_ready primitive
                    let mut conn = connection;
                    loop {
                        match conn.poll_ready(&mut context) {
                            Poll::Ready(Ok(())) => {
                                // Connection healthy, continue monitoring
                                continue;
                            }
                            Poll::Ready(Err(e)) => {
                                is_connected.store(false, Ordering::SeqCst);
                                emit!(
                                    sender,
                                    H2ConnectionChunk::bad_chunk(format!("Connection lost: {}", e))
                                );
                                break;
                            }
                            Poll::Pending => {
                                // Connection not ready, AsyncStream handles this
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    emit!(
                        sender,
                        H2ConnectionChunk::bad_chunk(format!("Handshake failed: {}", e))
                    );
                }
            }
        })
    }

    /// Create multiplexed H2 receive stream for multiple streams
    ///
    /// Handles multiple H2 receive streams using
    /// direct poll-based primitives within AsyncStream::with_channel.
    #[inline]
    pub fn multiplexed_receive_stream(
        recv_streams: Vec<h2::RecvStream>,
        polling_context: Context<'_>,
    ) -> AsyncStream<H2DataChunk, 1024> {
        AsyncStream::<H2DataChunk, 1024>::with_channel(move |sender| {
            let mut context = polling_context;
            let mut streams = recv_streams;

            // Poll all receive streams using direct poll_data primitives
            while !streams.is_empty() {
                let mut completed_indices = Vec::new();

                for (index, recv_stream) in streams.iter_mut().enumerate() {
                    match recv_stream.poll_data(&mut context) {
                        Poll::Ready(Some(Ok(data))) => {
                            emit!(sender, H2DataChunk::from_bytes(data));
                        }
                        Poll::Ready(Some(Err(e))) => {
                            emit!(sender, H2DataChunk::bad_chunk(format!("Data error: {}", e)));
                            completed_indices.push(index);
                        }
                        Poll::Ready(None) => {
                            emit!(sender, H2DataChunk::stream_complete());
                            completed_indices.push(index);
                        }
                        Poll::Pending => {
                            // Stream not ready, continue to next
                        }
                    }
                }

                // Remove completed streams (in reverse order to maintain indices)
                for &index in completed_indices.iter().rev() {
                    streams.swap_remove(index);
                }

                if streams.is_empty() {
                    break;
                }
            }
        })
    }

    /// Create flow-controlled H2 send stream with backpressure
    ///
    /// Implements flow control using h2's direct poll_ready primitive
    /// within AsyncStream::with_channel pattern.
    #[inline]
    pub fn flow_controlled_send_stream(
        send_stream: h2::SendStream<bytes::Bytes>,
        data_chunks: Vec<bytes::Bytes>,
        polling_context: Context<'_>,
    ) -> AsyncStream<H2SendResult, 1024> {
        AsyncStream::<H2SendResult, 1024>::with_channel(move |sender| {
            let mut stream = send_stream;
            let mut context = polling_context;
            let mut remaining_chunks = data_chunks;

            while let Some(chunk) = remaining_chunks.pop() {
                // Use h2's direct poll_ready primitive for flow control
                match stream.poll_ready(&mut context) {
                    Poll::Ready(Ok(())) => {
                        let is_last = remaining_chunks.is_empty();

                        match stream.send_data(chunk, is_last) {
                            Ok(()) => {
                                emit!(sender, H2SendResult::data_sent());

                                if is_last {
                                    emit!(sender, H2SendResult::send_complete());
                                    break;
                                }
                            }
                            Err(e) => {
                                emit!(
                                    sender,
                                    H2SendResult::bad_chunk(format!("Send error: {}", e))
                                );
                                break;
                            }
                        }
                    }
                    Poll::Ready(Err(e)) => {
                        emit!(
                            sender,
                            H2SendResult::bad_chunk(format!("Flow control error: {}", e))
                        );
                        break;
                    }
                    Poll::Pending => {
                        // Flow control window not available, AsyncStream handles this
                        remaining_chunks.push(chunk); // Put chunk back
                        break;
                    }
                }
            }
        })
    }

    /// Get connection status
    #[inline]
    pub fn is_connected(&self) -> bool {
        self.is_connected.load(Ordering::Acquire)
    }

    /// Get connection ID
    #[inline]
    pub fn connection_id(&self) -> u64 {
        self.connection_id.load(Ordering::Acquire)
    }

    /// Update bytes sent counter
    #[inline]
    pub fn add_bytes_sent(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::SeqCst);
    }

    /// Update bytes received counter
    #[inline]
    pub fn add_bytes_received(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::SeqCst);
    }

    /// Get bytes sent
    #[inline]
    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent.load(Ordering::Acquire)
    }

    /// Get bytes received
    #[inline]
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received.load(Ordering::Acquire)
    }
}

impl Default for H2ConnectionManager {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
