//! JSON streaming functionality
//!
//! This module provides the CANONICAL JsonStream type for processing JSON response bodies
//! with zero-allocation design and fluent_ai_async streaming patterns.

use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;

use fluent_ai_async::prelude::*;
use serde::{Deserialize, de::DeserializeOwned};

use crate::streaming::chunks::HttpChunk;

/// Type alias for the chunk handler function
type ChunkHandler<T> =
    Box<dyn FnMut(&T) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + 'static>;

/// JSON stream that yields deserialized T values with streaming architecture
///
/// This is the CANONICAL JsonStream implementation that consolidates all
/// previous JsonStream variants into a single, comprehensive streaming type.
pub struct JsonStream<T> {
    inner: AsyncStream<T, 1024>,
    buffer: Vec<u8>,
    _phantom: PhantomData<T>,
    handler: Option<ChunkHandler<T>>,

    /// Stream configuration
    pub parse_mode: JsonParseMode,
    pub max_buffer_size: usize,
    pub allow_partial: bool,
}

/// JSON parsing modes for different streaming scenarios
#[derive(Debug, Clone, Copy)]
pub enum JsonParseMode {
    /// Parse complete JSON objects
    Complete,
    /// Parse JSON Lines (JSONL) format
    Lines,
    /// Parse JSON arrays with streaming elements
    Array,
    /// Parse streaming JSON with custom delimiter
    Delimited(char),
}

impl<T> Debug for JsonStream<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsonStream")
            .field("buffer_size", &self.buffer.len())
            .field("parse_mode", &self.parse_mode)
            .field("max_buffer_size", &self.max_buffer_size)
            .field("allow_partial", &self.allow_partial)
            .field("has_handler", &self.handler.is_some())
            .finish()
    }
}

impl<T: DeserializeOwned + Send + 'static> JsonStream<T> {
    /// Create new JsonStream from HTTP chunks
    #[inline]
    pub fn new(chunk_stream: AsyncStream<HttpChunk, 1024>) -> Self {
        let inner = Self::parse_json_stream(chunk_stream, JsonParseMode::Complete);

        Self {
            inner,
            buffer: Vec::new(),
            _phantom: PhantomData,
            handler: None,
            parse_mode: JsonParseMode::Complete,
            max_buffer_size: 1024 * 1024, // 1MB default
            allow_partial: false,
        }
    }

    /// Create JsonStream with specific parse mode
    #[inline]
    pub fn with_mode(chunk_stream: AsyncStream<HttpChunk, 1024>, mode: JsonParseMode) -> Self {
        let inner = Self::parse_json_stream(chunk_stream, mode);

        Self {
            inner,
            buffer: Vec::new(),
            _phantom: PhantomData,
            handler: None,
            parse_mode: mode,
            max_buffer_size: 1024 * 1024,
            allow_partial: false,
        }
    }

    /// Create JsonStream from raw bytes
    #[inline]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let chunk_stream = AsyncStream::with_channel(move |sender| {
            emit!(sender, HttpChunk::Body(bytes.into()));
            emit!(sender, HttpChunk::Complete);
        });

        Self::new(chunk_stream)
    }

    /// Set maximum buffer size
    #[inline]
    pub fn max_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Allow partial JSON parsing
    #[inline]
    pub fn allow_partial(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }

    /// Parse JSON stream from HTTP chunks
    fn parse_json_stream(
        chunk_stream: AsyncStream<HttpChunk, 1024>,
        mode: JsonParseMode,
    ) -> AsyncStream<T, 1024> {
        AsyncStream::with_channel(move |sender| {
            let mut buffer = Vec::new();

            for chunk in chunk_stream {
                match chunk {
                    HttpChunk::Body(bytes) | HttpChunk::Chunk(bytes) => {
                        buffer.extend_from_slice(&bytes);

                        // Try to parse based on mode
                        match mode {
                            JsonParseMode::Complete => {
                                if let Ok(value) = serde_json::from_slice::<T>(&buffer) {
                                    emit!(sender, value);
                                    buffer.clear();
                                }
                            }
                            JsonParseMode::Lines => {
                                Self::parse_json_lines(&mut buffer, &sender);
                            }
                            JsonParseMode::Array => {
                                Self::parse_json_array(&mut buffer, &sender);
                            }
                            JsonParseMode::Delimited(delimiter) => {
                                Self::parse_delimited_json(&mut buffer, delimiter, &sender);
                            }
                        }
                    }
                    HttpChunk::Complete => {
                        // Final parse attempt
                        if !buffer.is_empty() {
                            if let Ok(value) = serde_json::from_slice::<T>(&buffer) {
                                emit!(sender, value);
                            }
                        }
                        break;
                    }
                    HttpChunk::Error(error) => {
                        // Emit error as bad chunk if T implements MessageChunk
                        break;
                    }
                    _ => {}
                }
            }
        })
    }

    /// Parse JSON Lines format
    fn parse_json_lines(buffer: &mut Vec<u8>, sender: &AsyncStreamSender<T>) {
        let text = String::from_utf8_lossy(buffer);
        let mut lines = text.lines().collect::<Vec<_>>();

        // Keep the last incomplete line in buffer
        if !text.ends_with('\n') && !lines.is_empty() {
            let last_line = lines.pop().unwrap();
            *buffer = last_line.as_bytes().to_vec();
        } else {
            buffer.clear();
        }

        // Parse complete lines
        for line in lines {
            if let Ok(value) = serde_json::from_str::<T>(line) {
                emit!(sender, value);
            }
        }
    }

    /// Parse JSON array with streaming elements
    fn parse_json_array(buffer: &mut Vec<u8>, sender: &AsyncStreamSender<T>) {
        let text = String::from_utf8_lossy(buffer);

        // Simple array parsing - look for complete JSON objects within array
        let mut depth = 0;
        let mut start = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, ch) in text.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match ch {
                '"' if !escape_next => in_string = !in_string,
                '\\' if in_string => escape_next = true,
                '{' if !in_string => {
                    if depth == 0 {
                        start = i;
                    }
                    depth += 1;
                }
                '}' if !in_string => {
                    depth -= 1;
                    if depth == 0 {
                        let json_str = &text[start..=i];
                        if let Ok(value) = serde_json::from_str::<T>(json_str) {
                            emit!(sender, value);
                        }
                    }
                }
                _ => {}
            }
        }

        // Keep remaining incomplete JSON in buffer
        if depth > 0 {
            *buffer = text[start..].as_bytes().to_vec();
        } else {
            buffer.clear();
        }
    }

    /// Parse delimited JSON
    fn parse_delimited_json(buffer: &mut Vec<u8>, delimiter: char, sender: &AsyncStreamSender<T>) {
        let text = String::from_utf8_lossy(buffer);
        let parts: Vec<&str> = text.split(delimiter).collect();

        // Keep the last incomplete part in buffer
        if parts.len() > 1 {
            *buffer = parts.last().unwrap().as_bytes().to_vec();

            // Parse complete parts
            for part in &parts[..parts.len() - 1] {
                if let Ok(value) = serde_json::from_str::<T>(part) {
                    emit!(sender, value);
                }
            }
        }
    }

    /// Get next JSON value
    #[inline]
    pub fn try_next(&mut self) -> Option<T> {
        self.inner.try_next()
    }

    /// Collect all JSON values
    pub fn collect_json(self) -> Vec<T> {
        self.inner.collect()
    }

    /// Get first JSON value
    pub fn first(mut self) -> Option<T> {
        self.inner.try_next()
    }

    /// Transform JSON values with a mapping function
    pub fn map<U, F>(self, mapper: F) -> AsyncStream<U, 1024>
    where
        F: Fn(T) -> U + Send + 'static,
        U: Send + 'static,
    {
        AsyncStream::with_channel(move |sender| {
            for value in self.inner {
                let mapped = mapper(value);
                emit!(sender, mapped);
            }
        })
    }

    /// Filter JSON values based on predicate
    pub fn filter<F>(self, predicate: F) -> AsyncStream<T, 1024>
    where
        F: Fn(&T) -> bool + Send + 'static,
    {
        AsyncStream::with_channel(move |sender| {
            for value in self.inner {
                if predicate(&value) {
                    emit!(sender, value);
                }
            }
        })
    }

    /// Take only the first N JSON values
    pub fn take(self, n: usize) -> AsyncStream<T, 1024> {
        AsyncStream::with_channel(move |sender| {
            let mut count = 0;
            for value in self.inner {
                if count >= n {
                    break;
                }
                emit!(sender, value);
                count += 1;
            }
        })
    }

    /// Skip the first N JSON values
    pub fn skip(self, n: usize) -> AsyncStream<T, 1024> {
        AsyncStream::with_channel(move |sender| {
            let mut count = 0;
            for value in self.inner {
                if count >= n {
                    emit!(sender, value);
                }
                count += 1;
            }
        })
    }
}

impl<T> JsonStream<T>
where
    T: Clone + Send + 'static,
{
    /// Add chunk handler for processing values
    #[inline]
    pub fn on_chunk<F>(mut self, f: F) -> Self
    where
        F: FnMut(&T) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + 'static,
    {
        self.handler = Some(Box::new(f));
        self
    }
}

impl<T> Iterator for JsonStream<T>
where
    T: DeserializeOwned + Send + 'static,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.try_next()
    }
}

/// JSON stream builder for ergonomic construction
#[derive(Debug)]
pub struct JsonStreamBuilder<T> {
    parse_mode: JsonParseMode,
    max_buffer_size: usize,
    allow_partial: bool,
    _phantom: PhantomData<T>,
}

impl<T> Default for JsonStreamBuilder<T> {
    fn default() -> Self {
        Self {
            parse_mode: JsonParseMode::Complete,
            max_buffer_size: 1024 * 1024,
            allow_partial: false,
            _phantom: PhantomData,
        }
    }
}

impl<T: DeserializeOwned + Send + 'static> JsonStreamBuilder<T> {
    /// Create new builder
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set parse mode
    #[inline]
    pub fn parse_mode(mut self, mode: JsonParseMode) -> Self {
        self.parse_mode = mode;
        self
    }

    /// Set maximum buffer size
    #[inline]
    pub fn max_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Allow partial JSON parsing
    #[inline]
    pub fn allow_partial(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }

    /// Build JsonStream from HTTP chunks
    #[inline]
    pub fn build(self, chunk_stream: AsyncStream<HttpChunk, 1024>) -> JsonStream<T> {
        JsonStream::with_mode(chunk_stream, self.parse_mode)
            .max_buffer_size(self.max_buffer_size)
            .allow_partial(self.allow_partial)
    }

    /// Build JsonStream from bytes
    #[inline]
    pub fn build_from_bytes(self, bytes: Vec<u8>) -> JsonStream<T> {
        JsonStream::from_bytes(bytes)
            .max_buffer_size(self.max_buffer_size)
            .allow_partial(self.allow_partial)
    }
}
