//! Trait implementations for Http3Builder
//!
//! Contains implementations of external traits including ChunkHandler for
//! fluent_ai_async integration and Debug for development support.

use std::fmt;
use std::sync::Arc;

use fluent_ai_async::prelude::ChunkHandler;

use crate::{HttpChunk, HttpError};

use super::builder_core::Http3Builder;

/// Implement ChunkHandler trait for Http3Builder to support cyrup_sugars on_chunk pattern
impl<S> ChunkHandler<HttpChunk, HttpError> for Http3Builder<S> {
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<HttpChunk, HttpError>) -> HttpChunk + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Arc::new(handler));
        self
    }
}

impl<S> fmt::Debug for Http3Builder<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Http3Builder")
            .field("client", &self.client)
            .field("request", &self.request)
            .field("debug_enabled", &self.debug_enabled)
            .field("jsonpath_config", &self.jsonpath_config)
            .field("chunk_handler", &self.chunk_handler.is_some())
            .finish()
    }
}