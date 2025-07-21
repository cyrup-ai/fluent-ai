//! Canonical type definitions for the streams-only architecture.

/// The core asynchronous stream type used throughout the fluent-ai ecosystem.
///
/// This is an unwrapped stream of values, built on top of tokio's unbounded channel receiver.
/// All error handling is expected to be managed by the stream producer and communicated
/// through side channels or a dedicated error handling mechanism within the stream's finalizer (e.g., `on_chunk`).
pub type AsyncStream<T> = tokio_stream::wrappers::UnboundedReceiverStream<T>;

use tokio_stream::Stream;

/// An extension trait for `Stream` that provides the `on_chunk` finalizer method.
///
/// This trait is the cornerstone of the streams-only architecture, providing a uniform
/// way to consume streams of unwrapped values.
pub trait AsyncStreamExt: Stream + Sized {
    /// Consumes the stream, applying a handler to each item in a spawned task.
    ///
    /// This is a "fire-and-forget" operation. It spawns a new Tokio task to drive the
    /// stream to completion, calling the provided handler for each item. This is the
    /// primary method for processing stream data in the fluent-ai ecosystem.
    ///
    /// # Arguments
    ///
    /// * `handler`: A closure that will be called for each item in the stream.
    fn on_chunk<F>(mut self, mut handler: F)
    where
        F: FnMut(Self::Item) + Send + 'static,
        Self::Item: Send + 'static,
        Self: Send + 'static,
    {
        tokio::spawn(async move {
            use tokio_stream::StreamExt;
            while let Some(item) = self.next().await {
                handler(item);
            }
        });
    }
}

/// Blanket implementation of `AsyncStreamExt` for all types that implement `Stream`.
impl<S> AsyncStreamExt for S where S: Stream + Sized {}
