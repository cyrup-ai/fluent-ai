//! Contains implementations of external traits including ChunkHandler for
//! fluent_ai_async integration and Debug for development support.

use std::fmt;
use std::sync::Arc;

use fluent_ai_async::prelude::ChunkHandler;

use super::builder_core::Http3Builder;
use crate::{HttpChunk, HttpError};

/// Trait for builder extensions
pub trait BuilderExt {
    /// Add custom chunk handler for stream processing
    fn on_chunk<F>(self, handler: F) -> Self
    where
        F: Fn(Result<HttpChunk, HttpError>) -> HttpChunk + Send + Sync + 'static;
}

/// Request builder extensions
pub trait RequestBuilderExt {
    /// Configure request with custom settings
    fn configure<F>(self, config_fn: F) -> Self
    where
        F: FnOnce(Self) -> Self;

    /// Add middleware to request processing
    fn middleware<F>(self, middleware_fn: F) -> Self
    where
        F: Fn(HttpChunk) -> HttpChunk + Send + Sync + 'static;
}

impl<S> BuilderExt for Http3Builder<S> {
    #[inline]
    fn on_chunk<F>(self, handler: F) -> Self
    where
        F: Fn(Result<HttpChunk, HttpError>) -> HttpChunk + Send + Sync + 'static,
    {
        self.set_chunk_handler(Arc::new(handler))
    }
}

impl<S> RequestBuilderExt for Http3Builder<S> {
    #[inline]
    fn configure<F>(self, config_fn: F) -> Self
    where
        F: FnOnce(Self) -> Self,
    {
        config_fn(self)
    }

    #[inline]
    fn middleware<F>(self, _middleware_fn: F) -> Self
    where
        F: Fn(HttpChunk) -> HttpChunk + Send + Sync + 'static,
    {
        // Implementation would store middleware function for later use
        self
    }
}

/// Implement ChunkHandler trait for Http3Builder to support fluent_ai_async on_chunk pattern
impl<S> ChunkHandler<HttpChunk, HttpError> for Http3Builder<S> {
    #[inline]
    fn on_chunk<F>(self, handler: F) -> Self
    where
        F: Fn(Result<HttpChunk, HttpError>) -> HttpChunk + Send + Sync + 'static,
    {
        self.set_chunk_handler(Arc::new(handler))
    }
}

impl<S> Http3Builder<S> {
    /// Internal method to set chunk handler
    #[inline]
    fn set_chunk_handler(
        mut self,
        handler: Arc<dyn Fn(Result<HttpChunk, HttpError>) -> HttpChunk + Send + Sync>,
    ) -> Self {
        self.chunk_handler = Some(handler);
        self
    }
}

impl<S> fmt::Debug for Http3Builder<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Http3Builder")
            .field("client", &"HttpClient")
            .field("request", &"HttpRequest")
            .field("debug_enabled", &self.debug_enabled)
            .finish()
    }
}
