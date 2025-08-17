//! Core frame types for HTTP/2 and HTTP/3 streaming
//!
//! Zero-allocation frame handling using fluent_ai_async patterns.

use std::collections::HashMap;

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit};

/// HTTP/2 frame types
#[derive(Debug, Clone)]
pub enum H2Frame {
    Data {
        stream_id: u32,
        data: Vec<u8>,
        end_stream: bool,
    },
    Headers {
        stream_id: u32,
        headers: HashMap<String, String>,
        end_stream: bool,
        end_headers: bool,
    },
    Priority {
        stream_id: u32,
        dependency: u32,
        weight: u8,
        exclusive: bool,
    },
    RstStream {
        stream_id: u32,
        error_code: u32,
    },
    Settings {
        settings: HashMap<u16, u32>,
    },
    PushPromise {
        stream_id: u32,
        promised_stream_id: u32,
        headers: HashMap<String, String>,
    },
    Ping {
        data: [u8; 8],
    },
    GoAway {
        last_stream_id: u32,
        error_code: u32,
        debug_data: Vec<u8>,
    },
    WindowUpdate {
        stream_id: u32,
        increment: u32,
    },
    Continuation {
        stream_id: u32,
        header_block: Vec<u8>,
        end_headers: bool,
    },
}

/// HTTP/3 frame types  
#[derive(Debug, Clone)]
pub enum H3Frame {
    Data { data: Vec<u8> },
    Headers { header_block: Vec<u8> },
    CancelPush { push_id: u64 },
    Settings { settings: HashMap<u64, u64> },
    PushPromise { push_id: u64, header_block: Vec<u8> },
    GoAway { stream_id: u64 },
    MaxPushId { push_id: u64 },
}

/// Unified frame chunk for streaming
#[derive(Debug, Clone)]
pub enum FrameChunk {
    H2(H2Frame),
    H3(H3Frame),
    Error(String),
    End,
}

impl MessageChunk for H2Frame {
    fn bad_chunk(error: String) -> Self {
        H2Frame::GoAway {
            last_stream_id: 0,
            error_code: 2, // INTERNAL_ERROR
            debug_data: error.into_bytes(),
        }
    }

    fn is_error(&self) -> bool {
        matches!(self, H2Frame::GoAway { .. } | H2Frame::RstStream { .. })
    }

    fn error(&self) -> Option<&str> {
        match self {
            H2Frame::GoAway { debug_data, .. } => std::str::from_utf8(debug_data).ok(),
            _ => None,
        }
    }
}

impl MessageChunk for H3Frame {
    fn bad_chunk(error: String) -> Self {
        H3Frame::GoAway { stream_id: 0 }
    }

    fn is_error(&self) -> bool {
        matches!(self, H3Frame::GoAway { .. } | H3Frame::CancelPush { .. })
    }

    fn error(&self) -> Option<&str> {
        match self {
            H3Frame::GoAway { .. } => Some("HTTP/3 connection terminated"),
            H3Frame::CancelPush { .. } => Some("Push stream cancelled"),
            _ => None,
        }
    }
}

impl MessageChunk for FrameChunk {
    fn bad_chunk(error: String) -> Self {
        FrameChunk::Error(error)
    }

    fn is_error(&self) -> bool {
        matches!(self, FrameChunk::Error(_))
    }

    fn error(&self) -> Option<&str> {
        match self {
            FrameChunk::Error(msg) => Some(msg),
            _ => None,
        }
    }
}

impl H2Frame {
    /// Get stream ID for this frame
    pub fn stream_id(&self) -> u32 {
        match self {
            H2Frame::Data { stream_id, .. } => *stream_id,
            H2Frame::Headers { stream_id, .. } => *stream_id,
            H2Frame::Priority { stream_id, .. } => *stream_id,
            H2Frame::RstStream { stream_id, .. } => *stream_id,
            H2Frame::Settings { .. } => 0,
            H2Frame::PushPromise { stream_id, .. } => *stream_id,
            H2Frame::Ping { .. } => 0,
            H2Frame::GoAway { .. } => 0,
            H2Frame::WindowUpdate { stream_id, .. } => *stream_id,
            H2Frame::Continuation { stream_id, .. } => *stream_id,
        }
    }

    /// Check if this frame ends the stream
    pub fn ends_stream(&self) -> bool {
        match self {
            H2Frame::Data { end_stream, .. } => *end_stream,
            H2Frame::Headers { end_stream, .. } => *end_stream,
            H2Frame::RstStream { .. } => true,
            _ => false,
        }
    }
}

impl H3Frame {
    /// Check if this is a data frame
    pub fn is_data(&self) -> bool {
        matches!(self, H3Frame::Data { .. })
    }

    /// Check if this is a headers frame
    pub fn is_headers(&self) -> bool {
        matches!(self, H3Frame::Headers { .. })
    }
}

impl Default for FrameChunk {
    fn default() -> Self {
        FrameChunk::End
    }
}

impl FrameChunk {
    /// Create data chunk
    pub fn data_chunk(data: Vec<u8>) -> Self {
        FrameChunk::H3(H3Frame::Data { data })
    }

    /// Create headers chunk
    pub fn headers_chunk(headers: HashMap<String, String>) -> Self {
        FrameChunk::H3(H3Frame::Headers {
            header_block: serialize_headers(&headers),
        })
    }

    /// Create end chunk
    pub fn end_chunk() -> Self {
        FrameChunk::End
    }

    /// Check if this is an error chunk
    pub fn is_error(&self) -> bool {
        matches!(self, FrameChunk::Error(_))
    }

    /// Check if this is end chunk
    pub fn is_end(&self) -> bool {
        matches!(self, FrameChunk::End)
    }
}

/// Serialize headers to byte block (simplified)
fn serialize_headers(headers: &HashMap<String, String>) -> Vec<u8> {
    let mut block = Vec::new();
    for (key, value) in headers {
        block.extend_from_slice(key.as_bytes());
        block.push(0x00);
        block.extend_from_slice(value.as_bytes());
        block.push(0x00);
    }
    block
}

/// Deserialize headers from byte block using streaming pattern
pub fn deserialize_headers_streaming(block: Vec<u8>) -> AsyncStream<FrameChunk, 1024> {
    AsyncStream::<FrameChunk, 1024>::with_channel(move |sender| {
        let mut headers = HashMap::new();
        let mut offset = 0;

        while offset < block.len() {
            // Simplified header parsing - real HPACK would be more complex
            if offset + 4 > block.len() {
                break;
            }

            let name_len = u16::from_be_bytes([block[offset], block[offset + 1]]) as usize;
            offset += 2;

            if offset + name_len > block.len() {
                emit!(
                    sender,
                    FrameChunk::bad_chunk("Invalid header name length".to_string())
                );
                return;
            }

            let name = String::from_utf8_lossy(&block[offset..offset + name_len]).to_string();
            offset += name_len;

            if offset + 2 > block.len() {
                emit!(
                    sender,
                    FrameChunk::bad_chunk("Invalid header value length".to_string())
                );
                return;
            }

            let value_len = u16::from_be_bytes([block[offset], block[offset + 1]]) as usize;
            offset += 2;

            if offset + value_len > block.len() {
                emit!(
                    sender,
                    FrameChunk::bad_chunk("Invalid header value length".to_string())
                );
                return;
            }

            let value = String::from_utf8_lossy(&block[offset..offset + value_len]).to_string();
            offset += value_len;

            headers.insert(name, value);
        }

        // Emit successful header parsing as H2 headers frame
        let headers_frame = H2Frame::Headers {
            stream_id: 0, // Will be set by caller
            headers,
            end_headers: true,
            end_stream: false,
        };
        emit!(sender, FrameChunk::H2(headers_frame));
    })
}

/// Deserialize headers from byte block (simplified)
pub fn deserialize_headers(block: &[u8]) -> Result<HashMap<String, String>, String> {
    let mut headers = HashMap::new();
    let mut offset = 0;

    while offset < block.len() {
        // Find key
        let key_end = block[offset..]
            .iter()
            .position(|&b| b == 0)
            .ok_or("Invalid header format")?;
        let key = String::from_utf8(block[offset..offset + key_end].to_vec())
            .map_err(|_| "Invalid UTF-8 in header key")?;
        offset += key_end + 1;

        if offset >= block.len() {
            break;
        }

        // Find value
        let value_end = block[offset..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(block.len() - offset);
        let value = String::from_utf8(block[offset..offset + value_end].to_vec())
            .map_err(|_| "Invalid UTF-8 in header value")?;
        offset += value_end + 1;

        headers.insert(key, value);
    }

    Ok(headers)
}
