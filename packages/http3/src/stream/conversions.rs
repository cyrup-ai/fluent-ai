//! Type conversions for HTTP streaming types
//! Provides elegant error handling and type transformations

use super::chunks::{BadChunk, HttpChunk};

// Conversions between BadChunk and HttpChunk for elegant error handling
impl From<BadChunk> for HttpChunk {
    fn from(bad_chunk: BadChunk) -> Self {
        match bad_chunk {
            BadChunk::Error(error) => HttpChunk::Error(error.to_string()),
            BadChunk::ProcessingFailed { error, .. } => HttpChunk::Error(error.to_string()),
        }
    }
}

impl From<crate::HttpError> for BadChunk {
    fn from(error: crate::HttpError) -> Self {
        BadChunk::Error(error)
    }
}

// Enable BadChunk -> serde_json::Value conversion for JSON processing
impl From<BadChunk> for serde_json::Value {
    fn from(bad_chunk: BadChunk) -> Self {
        match bad_chunk {
            BadChunk::Error(error) => {
                serde_json::json!({
                    "error": true,
                    "message": error.to_string(),
                    "type": "http_error"
                })
            }
            BadChunk::ProcessingFailed { error, context } => {
                serde_json::json!({
                    "error": true,
                    "message": error.to_string(),
                    "context": context,
                    "type": "processing_error"
                })
            }
        }
    }
}

// Enable BadChunk -> Vec<u8> conversion for raw byte collection
impl From<BadChunk> for Vec<u8> {
    fn from(bad_chunk: BadChunk) -> Self {
        // Convert error to JSON bytes
        let error_json = serde_json::Value::from(bad_chunk);
        serde_json::to_vec(&error_json).unwrap_or_else(|_| b"[]".to_vec())
    }
}
