//! HTTP response types and utilities
//!
//! Unified response module providing:
//! - Core HttpResponse struct and status handling
//! - Streaming response types and builders
//! - Header management and caching utilities  
//! - Body processing including JSON streaming and SSE parsing

pub mod body;
pub mod chunk;
pub mod core;
pub mod headers;
pub mod streaming;

// Re-export all main types for backward compatibility
pub use core::HttpResponse;

pub use body::{JsonStream, SseEvent};
pub use chunk::{HttpDownloadChunk, HttpRequestChunk};
pub use streaming::{HttpStreamingResponse, HttpStreamingResponseBuilder};

use crate::streaming::chunks::HttpChunk;
pub use crate::types::{HttpVersion, RequestMetadata, TimeoutConfig};
