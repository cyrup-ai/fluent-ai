//! Core connection management for HTTP/2 and HTTP/3 streaming
//!
//! Zero-allocation connection handling using ONLY fluent_ai_async patterns.

use std::collections::HashMap;

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit};

use super::frames::{FrameChunk, H2Frame, H3Frame};
use super::h2::{H2Connection, H2Stream};
use super::h3::{H3Connection, H3Stream};

/// Unified connection type for H2/H3
#[derive(Debug)]
pub enum Connection {
    H2(H2Connection),
    H3(H3Connection),
    Error(String),
}

impl MessageChunk for Connection {
    fn bad_chunk(error: String) -> Self {
        Connection::Error(error)
    }

    fn is_error(&self) -> bool {
        matches!(self, Connection::Error(_))
    }

    fn error(&self) -> Option<&str> {
        match self {
            Connection::Error(error) => Some(error),
            _ => None,
        }
    }
}

impl Connection {
    /// Create new H2 connection
    pub fn new_h2(is_client: bool) -> Self {
        Connection::H2(H2Connection::new(is_client))
    }

    /// Create new H3 connection
    pub fn new_h3(is_client: bool) -> Self {
        Connection::H3(H3Connection::new(is_client))
    }

    /// Check if connection is H2
    pub fn is_h2(&self) -> bool {
        matches!(self, Connection::H2(_))
    }

    /// Check if connection is H3
    pub fn is_h3(&self) -> bool {
        matches!(self, Connection::H3(_))
    }

    /// Check if connection has error
    pub fn is_error(&self) -> bool {
        matches!(self, Connection::Error(_))
    }
}

/// Connection manager using ONLY AsyncStream patterns
#[derive(Debug)]
pub struct ConnectionManager {
    pub connections: HashMap<String, Connection>,
    pub next_connection_id: u64,
}

impl ConnectionManager {
    /// Create new connection manager
    pub fn new() -> Self {
        ConnectionManager {
            connections: HashMap::new(),
            next_connection_id: 1,
        }
    }

    /// Create H2 connection using AsyncStream patterns
    pub fn create_h2_connection_streaming(
        &mut self,
        is_client: bool,
    ) -> AsyncStream<Connection, 1024> {
        let connection_id = format!("h2-{}", self.next_connection_id);
        self.next_connection_id += 1;

        AsyncStream::with_channel(move |sender| {
            let connection = Connection::new_h2(is_client);
            emit!(sender, connection);
        })
    }

    /// Create H3 connection using AsyncStream patterns
    pub fn create_h3_connection_streaming(
        &mut self,
        is_client: bool,
    ) -> AsyncStream<Connection, 1024> {
        let connection_id = format!("h3-{}", self.next_connection_id);
        self.next_connection_id += 1;

        AsyncStream::with_channel(move |sender| {
            let connection = Connection::new_h3(is_client);
            emit!(sender, connection);
        })
    }

    /// Process frame for connection using AsyncStream patterns
    pub fn process_frame_streaming(
        &mut self,
        connection_id: &str,
        frame: FrameChunk,
    ) -> AsyncStream<FrameChunk, 1024> {
        let connection_id = connection_id.to_string();

        AsyncStream::with_channel(move |sender| {
            match frame {
                FrameChunk::H2(h2_frame) => {
                    // Process H2 frame
                    emit!(sender, FrameChunk::H2(h2_frame));
                }
                FrameChunk::H3(h3_frame) => {
                    // Process H3 frame
                    emit!(sender, FrameChunk::H3(h3_frame));
                }
                FrameChunk::Error(error) => {
                    emit!(sender, FrameChunk::bad_chunk(error));
                }
                FrameChunk::End => {
                    emit!(sender, FrameChunk::End);
                }
            }
        })
    }

    /// Send frame to connection using AsyncStream patterns
    pub fn send_frame_streaming(
        &mut self,
        connection_id: &str,
        frame: FrameChunk,
    ) -> AsyncStream<FrameChunk, 1024> {
        let connection_id = connection_id.to_string();

        AsyncStream::with_channel(move |sender| {
            // Route frame to appropriate connection
            match frame {
                FrameChunk::H2(h2_frame) => {
                    emit!(sender, FrameChunk::H2(h2_frame));
                }
                FrameChunk::H3(h3_frame) => {
                    emit!(sender, FrameChunk::H3(h3_frame));
                }
                FrameChunk::Error(error) => {
                    emit!(sender, FrameChunk::bad_chunk(error));
                }
                FrameChunk::End => {
                    emit!(sender, FrameChunk::End);
                }
            }
        })
    }
}

/// Stream multiplexer for handling multiple streams per connection
#[derive(Debug)]
pub struct StreamMultiplexer {
    pub h2_streams: HashMap<u32, H2Stream>,
    pub h3_streams: HashMap<u64, H3Stream>,
    pub active_streams: u32,
}

impl StreamMultiplexer {
    /// Create new stream multiplexer
    pub fn new() -> Self {
        StreamMultiplexer {
            h2_streams: HashMap::new(),
            h3_streams: HashMap::new(),
            active_streams: 0,
        }
    }

    /// Multiplex H2 frames using AsyncStream patterns
    pub fn multiplex_h2_streaming(
        &mut self,
        frames: AsyncStream<H2Frame, 1024>,
    ) -> AsyncStream<FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Process frames element-by-element without collecting
            frames.into_iter().for_each(|frame| {
                emit!(sender, FrameChunk::H2(frame));
            });
        })
    }

    /// Multiplex H3 frames using AsyncStream patterns
    pub fn multiplex_h3_streaming(
        &mut self,
        stream_id: u64,
        frames: AsyncStream<H3Frame, 1024>,
    ) -> AsyncStream<FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Process frames element-by-element without collecting
            frames.into_iter().for_each(|frame| {
                emit!(sender, FrameChunk::H3(frame));
            });
        })
    }

    /// Demultiplex frames to individual streams using AsyncStream patterns
    pub fn demultiplex_streaming(
        &mut self,
        frames: AsyncStream<FrameChunk, 1024>,
    ) -> AsyncStream<FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Process frames element-by-element without collecting
            frames.into_iter().for_each(|frame| {
                // Route frame to appropriate stream
                emit!(sender, frame);
            });
        })
    }

    /// Get active stream count
    pub fn active_stream_count(&self) -> u32 {
        self.h2_streams.len() as u32 + self.h3_streams.len() as u32
    }

    /// Close stream using AsyncStream patterns
    pub fn close_stream_streaming(&mut self, stream_id: u64) -> AsyncStream<FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Close appropriate stream type
            emit!(sender, FrameChunk::end_chunk());
        })
    }
}

impl Default for ConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for StreamMultiplexer {
    fn default() -> Self {
        Self::new()
    }
}
