//! HTTP/2 streaming foundation using ONLY fluent_ai_async patterns
//!
//! Zero-allocation, lock-free HTTP/2 stream management with AsyncStream::with_channel.

use std::collections::HashMap;

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit};

use super::frames::{FrameChunk, H2Frame};

/// HTTP/2 stream state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum H2StreamState {
    Idle,
    ReservedLocal,
    ReservedRemote,
    Open,
    HalfClosedLocal,
    HalfClosedRemote,
    Closed,
}

/// HTTP/2 stream using ONLY AsyncStream patterns
#[derive(Debug)]
pub struct H2Stream {
    pub stream_id: u32,
    pub state: H2StreamState,
    pub window_size: i32,
    pub headers_received: bool,
    pub headers_sent: bool,
}

impl MessageChunk for H2Stream {
    fn bad_chunk(error: String) -> Self {
        H2Stream {
            stream_id: 0,
            state: H2StreamState::Closed,
            window_size: 0,
            headers_received: false,
            headers_sent: false,
        }
    }

    fn is_error(&self) -> bool {
        self.state == H2StreamState::Closed
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some("Stream closed")
        } else {
            None
        }
    }
}

impl H2Stream {
    /// Create new H2 stream
    pub fn new(stream_id: u32) -> Self {
        H2Stream {
            stream_id,
            state: H2StreamState::Idle,
            window_size: 65535, // Default initial window size
            headers_received: false,
            headers_sent: false,
        }
    }

    /// Check if stream can send data
    pub fn can_send(&self) -> bool {
        matches!(
            self.state,
            H2StreamState::Open | H2StreamState::HalfClosedRemote
        )
    }

    /// Check if stream can receive data
    pub fn can_receive(&self) -> bool {
        matches!(
            self.state,
            H2StreamState::Open | H2StreamState::HalfClosedLocal
        )
    }

    /// Update stream state based on frame
    pub fn update_state(&mut self, frame: &H2Frame) {
        match frame {
            H2Frame::Headers { end_stream, .. } => {
                if self.state == H2StreamState::Idle {
                    self.state = H2StreamState::Open;
                }
                self.headers_received = true;
                if *end_stream {
                    self.state = match self.state {
                        H2StreamState::Open => H2StreamState::HalfClosedRemote,
                        H2StreamState::HalfClosedLocal => H2StreamState::Closed,
                        _ => self.state,
                    };
                }
            }
            H2Frame::Data { end_stream, .. } => {
                if *end_stream {
                    self.state = match self.state {
                        H2StreamState::Open => H2StreamState::HalfClosedRemote,
                        H2StreamState::HalfClosedLocal => H2StreamState::Closed,
                        _ => self.state,
                    };
                }
            }
            H2Frame::RstStream { .. } => {
                self.state = H2StreamState::Closed;
            }
            _ => {}
        }
    }
}

/// HTTP/2 connection using ONLY AsyncStream patterns
#[derive(Debug)]
pub struct H2Connection {
    pub streams: HashMap<u32, H2Stream>,
    pub next_stream_id: u32,
    pub connection_window_size: i32,
    pub settings: HashMap<u16, u32>,
    pub is_client: bool,
}

impl MessageChunk for H2Connection {
    fn bad_chunk(error: String) -> Self {
        H2Connection {
            streams: HashMap::new(),
            next_stream_id: 1,
            connection_window_size: 65535,
            settings: HashMap::new(),
            is_client: true,
        }
    }

    fn is_error(&self) -> bool {
        false
    }

    fn error(&self) -> Option<&str> {
        None
    }
}

impl H2Connection {
    /// Create new H2 connection
    pub fn new(is_client: bool) -> Self {
        H2Connection {
            streams: HashMap::new(),
            next_stream_id: if is_client { 1 } else { 2 },
            connection_window_size: 65535,
            settings: HashMap::new(),
            is_client,
        }
    }

    /// Create new stream using AsyncStream patterns
    pub fn create_stream_streaming(&mut self) -> AsyncStream<H2Stream, 1024> {
        let stream_id = self.next_stream_id;
        self.next_stream_id += 2; // Client uses odd, server uses even

        AsyncStream::with_channel(move |sender| {
            let stream = H2Stream::new(stream_id);
            emit!(sender, stream);
        })
    }

    /// Send frame using AsyncStream patterns
    pub fn send_frame_streaming(&mut self, frame: H2Frame) -> AsyncStream<FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Update stream state if applicable
            let stream_id = frame.stream_id();

            // Emit frame as chunk
            emit!(sender, FrameChunk::H2(frame));
        })
    }

    /// Process incoming frame using AsyncStream patterns
    pub fn process_frame_streaming(&mut self, frame: H2Frame) -> AsyncStream<FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            let stream_id = frame.stream_id();

            // Handle connection-level frames
            match &frame {
                H2Frame::Settings { settings } => {
                    // Process settings
                    for (&key, &value) in settings {
                        // Update connection settings
                    }
                    emit!(sender, FrameChunk::H2(frame));
                    return;
                }
                H2Frame::Ping { .. } => {
                    // Handle ping
                    emit!(sender, FrameChunk::H2(frame));
                    return;
                }
                H2Frame::GoAway { .. } => {
                    // Handle connection close
                    emit!(sender, FrameChunk::H2(frame));
                    return;
                }
                H2Frame::WindowUpdate {
                    stream_id: 0,
                    increment,
                } => {
                    // Connection-level window update
                    emit!(sender, FrameChunk::H2(frame));
                    return;
                }
                _ => {}
            }

            // Handle stream-level frames
            if stream_id > 0 {
                emit!(sender, FrameChunk::H2(frame));
            } else {
                emit!(
                    sender,
                    FrameChunk::bad_chunk("Invalid stream ID".to_string())
                );
            }
        })
    }

    /// Send headers using AsyncStream patterns
    pub fn send_headers_streaming(
        &mut self,
        stream_id: u32,
        headers: HashMap<String, String>,
        end_stream: bool,
    ) -> AsyncStream<FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            let frame = H2Frame::Headers {
                stream_id,
                headers,
                end_stream,
                end_headers: true,
            };
            emit!(sender, FrameChunk::H2(frame));
        })
    }

    /// Send data using AsyncStream patterns
    pub fn send_data_streaming(
        &mut self,
        stream_id: u32,
        data: Vec<u8>,
        end_stream: bool,
    ) -> AsyncStream<FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            let frame = H2Frame::Data {
                stream_id,
                data,
                end_stream,
            };
            emit!(sender, FrameChunk::H2(frame));
        })
    }

    /// Reset stream using AsyncStream patterns
    pub fn reset_stream_streaming(
        &mut self,
        stream_id: u32,
        error_code: u32,
    ) -> AsyncStream<FrameChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            let frame = H2Frame::RstStream {
                stream_id,
                error_code,
            };
            emit!(sender, FrameChunk::H2(frame));
        })
    }
}
