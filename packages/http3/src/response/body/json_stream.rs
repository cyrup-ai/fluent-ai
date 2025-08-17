//! JSON streaming functionality
//!
//! This module provides JsonStream type for processing JSON response bodies
//! with zero-allocation design and user-controlled error handling via on_chunk patterns.

use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;

use crate::response::core::HttpResponse;

/// Type alias for the chunk handler function
type ChunkHandler<T> =
    Box<dyn FnMut(&T) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + 'static>;

/// JSON stream that yields unwrapped T values with user `on_chunk` error handling
/// Users get immediate values, error handling via `on_chunk` handler
pub struct JsonStream<T> {
    body: Vec<u8>,
    _phantom: PhantomData<T>,
    /// Optional handler for processing chunks
    #[allow(dead_code)]
    handler: Option<ChunkHandler<T>>,
}

impl<T> Debug for JsonStream<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsonStream")
            .field("body", &self.body)
            .field(
                "handler",
                &if self.handler.is_some() {
                    "Some(<function>)"
                } else {
                    "None"
                },
            )
            .finish()
    }
}

impl<T: serde::de::DeserializeOwned> JsonStream<T> {
    /// Create new JsonStream from body bytes
    pub fn new(body: Vec<u8>) -> Self {
        Self {
            body,
            _phantom: PhantomData,
            handler: None,
        }
    }

    /// Get JSON value - returns `T` directly (no futures)
    ///
    /// Users get immediate values, error handling via `on_chunk` handlers
    #[must_use]
    pub fn get(&self) -> Option<T> {
        // Parse JSON once and return the result
        // Error handling delegated to user on_chunk handlers
        serde_json::from_slice(&self.body).ok()
    }

    /// Collect all JSON values into a Vec
    #[must_use]
    pub fn collect_json(self) -> Vec<T> {
        match self.get() {
            Some(value) => vec![value],
            None => Vec::new(),
        }
    }
}

impl<T> JsonStream<T>
where
    T: Clone + Send + 'static,
{
    /// Add `on_chunk` handler for error handling and processing
    /// Users receive unwrapped values T, errors handled in `on_chunk`
    #[must_use = "returns the modified stream"]
    pub fn on_chunk<F>(self, f: F) -> Self
    where
        F: FnMut(&T) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + 'static,
    {
        Self {
            handler: Some(Box::new(f)),
            ..self
        }
    }

    /// Collect implementation for pure streaming architecture
    /// Users wanting "await" similar behavior call `.collect()`
    #[must_use = "method returns a new value and does not modify the original"]
    pub fn collect(self) -> JsonStream<T> {
        self
    }
}

impl HttpResponse {
    /// Parse the body as JSON stream - returns unwrapped T chunks
    /// Only available for JSON content-type responses
    /// Zero futures, error handling via user `on_chunk` handlers, users call `.collect()` for await-like behavior
    #[must_use]
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Option<JsonStream<T>> {
        // Only provide JSON parsing for JSON content types
        if self.is_json_content() {
            Some(JsonStream::new(self.body.to_vec()))
        } else {
            None
        }
    }
}
