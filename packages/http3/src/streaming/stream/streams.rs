//! Stream implementations for different response types

use std::sync::Arc;

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, AsyncStreamSender};

use crate::streaming::chunks::{BadChunk, DownloadChunk, HttpChunk, SseEvent};

/// HTTP response stream
pub type HttpStream = AsyncStream<HttpChunk>;

/// Download stream for file downloads
pub type DownloadStream = AsyncStream<DownloadChunk>;

/// JSON stream for structured data
pub type JsonStream<T> = AsyncStream<T>;

/// Server-Sent Events stream
pub type SseStream = AsyncStream<SseEvent>;

/// Line-based text stream
pub type LinesStream = AsyncStream<String>;

/// Generic async stream alias
pub type AsyncStreamType<T> = AsyncStream<T>;
