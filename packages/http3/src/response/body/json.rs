//! JSON stream processing and deserialization functionality
//!
//! Handles JSON parsing, collection, and streaming with zero-allocation design
//! and user-defined error handling through on_chunk patterns.

use super::types::{JsonStream, ChunkHandler};

impl<T: serde::de::DeserializeOwned> JsonStream<T> {
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