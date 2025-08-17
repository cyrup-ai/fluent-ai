use std::sync::Arc;

use cyrup_sugars::prelude::MessageChunk;
use fluent_ai_async::prelude::*;

// H2 Connection Chunk - TOKIO-FREE IMPLEMENTATION
#[derive(Debug, Clone)]
pub enum H2ConnectionChunk {
    Ready,
    ConnectionError { message: Arc<str> },
}

impl MessageChunk for H2ConnectionChunk {
    fn bad_chunk(error: String) -> Self {
        H2ConnectionChunk::ConnectionError {
            message: Arc::from(error.as_str()),
        }
    }

    fn error(&self) -> Option<&str> {
        match self {
            H2ConnectionChunk::ConnectionError { message } => Some(message.as_ref()),
            _ => None,
        }
    }

    fn is_error(&self) -> bool {
        matches!(self, H2ConnectionChunk::ConnectionError { .. })
    }
}

impl H2ConnectionChunk {
    #[inline]
    pub fn ready() -> Self {
        H2ConnectionChunk::Ready
    }
}

// H2 Request Chunk
#[derive(Debug, Clone)]
pub enum H2RequestChunk {
    Sent {
        stream_id: u32,
        connection_id: Arc<str>,
    },
    SendError {
        message: Arc<str>,
    },
}

impl MessageChunk for H2RequestChunk {
    fn bad_chunk(error: String) -> Self {
        H2RequestChunk::SendError {
            message: Arc::from(error.as_str()),
        }
    }

    fn error(&self) -> Option<&str> {
        match self {
            H2RequestChunk::SendError { message } => Some(message.as_ref()),
            _ => None,
        }
    }

    fn is_error(&self) -> bool {
        matches!(self, H2RequestChunk::SendError { .. })
    }
}

impl H2RequestChunk {
    #[inline]
    pub fn sent(stream_id: u32, connection_id: Arc<str>) -> Self {
        H2RequestChunk::Sent {
            stream_id,
            connection_id,
        }
    }
}

// H2 Data Chunk
#[derive(Debug, Clone)]
pub enum H2DataChunk {
    Data { bytes: bytes::Bytes },
    StreamComplete,
    DataError { message: Arc<str> },
}

impl MessageChunk for H2DataChunk {
    fn bad_chunk(error: String) -> Self {
        H2DataChunk::DataError {
            message: Arc::from(error.as_str()),
        }
    }

    fn error(&self) -> Option<&str> {
        match self {
            H2DataChunk::DataError { message } => Some(message.as_ref()),
            _ => None,
        }
    }

    fn is_error(&self) -> bool {
        matches!(self, H2DataChunk::DataError { .. })
    }
}

impl H2DataChunk {
    #[inline]
    pub fn from_bytes(bytes: bytes::Bytes) -> Self {
        H2DataChunk::Data { bytes }
    }

    #[inline]
    pub fn stream_complete() -> Self {
        H2DataChunk::StreamComplete
    }
}

// H2 Send Result
#[derive(Debug, Clone)]
pub enum H2SendResult {
    DataSent,
    SendComplete,
    SendError { message: String },
}

impl MessageChunk for H2SendResult {
    fn bad_chunk(error: String) -> Self {
        H2SendResult::SendError { message: error }
    }

    fn error(&self) -> Option<&str> {
        match self {
            H2SendResult::SendError { message } => Some(message.as_str()),
            _ => None,
        }
    }

    fn is_error(&self) -> bool {
        matches!(self, H2SendResult::SendError { .. })
    }
}

impl H2SendResult {
    #[inline]
    pub fn data_sent() -> Self {
        H2SendResult::DataSent
    }

    #[inline]
    pub fn send_complete() -> Self {
        H2SendResult::SendComplete
    }
}
