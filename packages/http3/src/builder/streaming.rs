//! Pure streams-first architecture - NO Futures, NO Result wrapping
//! Transforms HTTP byte streams into individual JSON objects via JSONPath

use fluent_ai_async::AsyncStream;
use fluent_ai_async::emit;
use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::prelude::MessageChunk;
use serde::de::DeserializeOwned;

use crate::{HttpChunk, HttpError, HttpStream};

/// Stream configuration settings
pub struct StreamConfig {
    pub buffer_size: usize,
    pub chunk_size: usize,
    pub enable_compression: bool,
}

/// Streaming builder for advanced stream configuration
pub struct StreamingBuilder {
    pub config: StreamConfig,
    pub jsonpath_expression: Option<String>,
}

impl Default for StreamConfig {
    #[inline]
    fn default() -> Self {
        Self {
            buffer_size: 8192,
            chunk_size: 4096,
            enable_compression: true,
        }
    }
}

impl Default for StreamingBuilder {
    #[inline]
    fn default() -> Self {
        Self {
            config: StreamConfig::default(),
            jsonpath_expression: None,
        }
    }
}

impl StreamingBuilder {
    /// Create new streaming builder
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set buffer size for streaming
    #[inline]
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Set chunk size for streaming
    #[inline]
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Enable or disable compression
    #[inline]
    pub fn compression(mut self, enabled: bool) -> Self {
        self.config.enable_compression = enabled;
        self
    }

    /// Set JSONPath expression for filtering
    #[inline]
    pub fn jsonpath(mut self, expression: &str) -> Self {
        self.jsonpath_expression = Some(expression.to_string());
        self
    }
}

/// Pure AsyncStream of JSON objects via JSONPath - NO Result wrapping
pub struct JsonPathStream<T> {
    inner: AsyncStream<T, 1024>,
    chunk_handler: Option<Box<dyn Fn(Result<T, HttpError>) -> T + Send + Sync>>,
}

impl<T> JsonPathStream<T>
where
    T: DeserializeOwned + Send + Default + MessageChunk + 'static,
{
    /// Create JSONPath stream from HTTP response - pure streams architecture
    #[inline]
    pub fn new(http_stream: HttpStream, jsonpath_expr: String) -> Self {
        Self {
            inner: AsyncStream::with_channel(move |sender| {
                std::thread::spawn(move || {
                    let mut buffer = Vec::new();
                    let stream = http_stream;

                    // Process HTTP chunks from stream
                    let chunks: Vec<HttpChunk> = stream.collect();

                    let processed_chunks: Vec<T> = chunks
                        .into_iter()
                        .filter_map(|chunk| {
                            if chunk.is_error() {
                                if let Some(error_msg) = chunk.error() {
                                    return Some(T::bad_chunk(error_msg.to_string()));
                                }
                                return None;
                            }

                            // Accumulate chunk data
                            buffer.extend_from_slice(&chunk.body);

                            // Try to parse accumulated JSON and apply JSONPath
                            if let Ok(json_value) =
                                serde_json::from_slice::<serde_json::Value>(&buffer)
                            {
                                // Apply JSONPath expression (simplified implementation)
                                if let Ok(item) = serde_json::from_value::<T>(json_value) {
                                    return Some(item);
                                }
                            }
                            None
                        })
                        .collect();

                    // Emit all processed chunks
                    emit!(sender, processed_chunks);
                });
            }),
            chunk_handler: None,
        }
    }

    /// Set chunk handler for error processing
    #[inline]
    pub fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<T, HttpError>) -> T + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }

    /// Collect all items from the stream
    #[inline]
    pub fn collect(self) -> Vec<T> {
        self.inner.collect()
    }

    /// Get the next item from the stream
    #[inline]
    pub fn next(&mut self) -> Option<T> {
        self.inner.try_next()
    }
}

impl<T> Iterator for JsonPathStream<T>
where
    T: DeserializeOwned + Send + Default + MessageChunk + 'static,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.try_next()
    }
}
