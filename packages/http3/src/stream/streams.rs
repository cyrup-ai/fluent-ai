//! HTTP stream implementations - Pure streams-first architecture
//! Zero futures, zero Result wrapping - all AsyncStream based

use fluent_ai_async::prelude::{ChunkHandler, MessageChunk};
use fluent_ai_async::{AsyncStream, AsyncStreamSender};

use super::chunks::{BadChunk, DownloadChunk, HttpChunk, SseEvent};

/// Pure AsyncStream of HTTP response chunks - NO Result wrapping
pub struct HttpStream {
    inner: AsyncStream<HttpChunk, 1024>,
    chunk_handler:
        Option<Box<dyn Fn(Result<HttpChunk, crate::HttpError>) -> HttpChunk + Send + Sync>>,
}

impl HttpStream {
    /// Create new HttpStream from AsyncStream
    pub fn new(stream: AsyncStream<HttpChunk, 1024>) -> Self {
        Self {
            inner: stream,
            chunk_handler: None,
        }
    }

    /// Create from generator function
    pub fn with_channel<F>(producer: F) -> Self
    where
        F: FnOnce(AsyncStreamSender<HttpChunk>) + Send + 'static,
    {
        Self {
            inner: AsyncStream::with_channel(producer),
            chunk_handler: None,
        }
    }

    /// Get next chunk
    pub fn poll_next(&mut self) -> Option<HttpChunk> {
        self.inner.try_next()
    }

    /// Collect all chunks as bytes
    pub fn collect_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for chunk in self {
            if let HttpChunk::Body(chunk_bytes) = chunk {
                bytes.extend_from_slice(&chunk_bytes);
            }
        }
        bytes
    }

    /// Collect and deserialize JSON
    pub fn collect_json<T>(self) -> Option<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let bytes = self.collect_bytes();
        if bytes.is_empty() {
            return None;
        }
        serde_json::from_slice(&bytes).ok()
    }

    /// Collect single response and deserialize - PUBLIC API COMPATIBILITY
    pub fn collect_one<T>(self) -> T
    where
        T: serde::de::DeserializeOwned + From<BadChunk> + Default,
    {
        match self.collect_json::<T>() {
            Some(result) => result,
            None => T::default(), // Fallback to default if parsing fails
        }
    }

    /// Collect single response with error handler - fluent_ai_async pattern
    pub fn collect_or_else<T, F>(self, mut error_handler: F) -> T
    where
        T: serde::de::DeserializeOwned + MessageChunk + Default + Send + 'static,
        F: FnMut(&T) -> T,
    {
        let bytes = self.collect_bytes();
        if bytes.is_empty() {
            return error_handler(&T::bad_chunk("Empty response".to_string()));
        }
        match serde_json::from_slice::<T>(&bytes) {
            Ok(result) => result,
            Err(_) => error_handler(&T::bad_chunk("Failed to parse response".to_string())),
        }
    }

    /// Collect all responses as vector - PUBLIC API COMPATIBILITY
    pub fn collect_all<T>(self) -> Vec<T>
    where
        T: serde::de::DeserializeOwned + From<BadChunk>,
    {
        // For single response HTTP, this typically returns a single-item vector
        match self.collect_json::<T>() {
            Some(result) => vec![result],
            None => vec![],
        }
    }

    /// Create HTTP stream from raw bytes - pure streams only
    #[must_use]
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self::with_channel(move |sender| {
            std::thread::spawn(move || {
                // Emit the data as a single body chunk
                let chunk = HttpChunk::Body(bytes::Bytes::from(data));
                let _ = sender.send(chunk);
            });
        })
    }
}

/// Iterator implementation for backwards compatibility with existing code
impl Iterator for HttpStream {
    type Item = HttpChunk;

    fn next(&mut self) -> Option<Self::Item> {
        self.poll_next()
    }
}

/// Implement ChunkHandler trait for HttpStream - pure fluent_ai_async pattern
impl ChunkHandler<HttpChunk, crate::HttpError> for HttpStream {
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<HttpChunk, crate::HttpError>) -> HttpChunk + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }
}

/// Pure AsyncStream of download chunks - NO Result wrapping
pub struct DownloadStream {
    inner: AsyncStream<DownloadChunk, 1024>,
    chunk_handler:
        Option<Box<dyn Fn(Result<DownloadChunk, crate::HttpError>) -> DownloadChunk + Send + Sync>>,
}

impl DownloadStream {
    /// Create new download stream
    #[must_use]
    pub fn new(stream: AsyncStream<DownloadChunk, 1024>) -> Self {
        Self {
            inner: stream,
            chunk_handler: None,
        }
    }

    /// Create download stream with progress tracking
    #[must_use]
    pub fn with_progress<F>(producer: F) -> Self
    where
        F: FnOnce(AsyncStreamSender<DownloadChunk>) + Send + 'static,
    {
        Self {
            inner: AsyncStream::with_channel(producer),
            chunk_handler: None,
        }
    }

    /// Get next download chunk
    pub fn poll_next(&mut self) -> Option<DownloadChunk> {
        self.inner.try_next()
    }

    /// Stream extension method equivalent - NO async, pure streams only
    pub fn try_next(&mut self) -> Option<DownloadChunk> {
        self.poll_next()
    }
}

impl ChunkHandler<DownloadChunk, crate::HttpError> for DownloadStream {
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<DownloadChunk, crate::HttpError>) -> DownloadChunk + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }
}

/// Pure AsyncStream of text lines - NO Result wrapping
pub type LinesStream = AsyncStream<String, 1024>;

/// Pure AsyncStream of Server-Sent Events - NO Result wrapping  
pub type SseStream = AsyncStream<SseEvent, 1024>;

/// Pure AsyncStream of JSON objects - NO Result wrapping
pub type JsonStream = AsyncStream<serde_json::Value, 1024>;
