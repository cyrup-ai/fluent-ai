//! HTTP streaming utilities - Pure streams-first architecture
//! Zero futures, zero Result wrapping - all AsyncStream based
//!
//! This module provides decomposed streaming functionality with logical separation:
//! - `chunks`: Core chunk types (HttpChunk, DownloadChunk, BadChunk, SseEvent)
//! - `streams`: Stream implementations (HttpStream, DownloadStream, type aliases)
//! - `conversions`: Type conversions and From trait implementations

pub mod chunks;
pub mod conversions;
pub mod streams;

// Re-export all public types to maintain API compatibility
pub use chunks::{BadChunk, DownloadChunk, HttpChunk, SseEvent};
pub use streams::{DownloadStream, HttpStream, JsonStream, LinesStream, SseStream};

// MessageChunk implementations moved to wrappers.rs to avoid orphan rule violations
// Use StringWrapper and BytesWrapper instead of implementing directly on external types
