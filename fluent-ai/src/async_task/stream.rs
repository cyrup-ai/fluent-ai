//! Async stream with built-in error handling and collection support

use super::task::NotResult;
use super::error_handlers::{BadAppleChunk, default_stream_error_handler, ChunkHandler, DefaultChunkHandler};
use futures::Stream;
use futures::StreamExt;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

/// Generic async stream wrapper for streaming operations
///
/// IMPORTANT: AsyncStream must never contain Result types - all error handling
/// should be done internally before sending items to the stream.
pub struct AsyncStream<T>
where
    T: NotResult, // T cannot be any Result type
{
    receiver: mpsc::UnboundedReceiver<T>,
}

impl<T> AsyncStream<T>
where
    T: NotResult, // T cannot be any Result type
{
    /// Create a new AsyncStream from an unbounded receiver
    pub fn new(receiver: mpsc::UnboundedReceiver<T>) -> Self {
        Self { receiver }
    }

    /// Create an AsyncStream from a futures Stream
    pub fn from_stream<S>(stream: S) -> crate::AsyncTask<Vec<T>>
    where
        S: Stream<Item = T> + Send + 'static,
        T: Send + 'static,
    {
        crate::AsyncTask::from_future(
            async move { futures::StreamExt::collect::<Vec<_>>(stream).await },
        )
    }

    /// Collect all items from the stream into a Vec
    pub async fn collect(mut self) -> Vec<T> {
        let mut items = Vec::new();
        while let Some(item) = self.receiver.recv().await {
            items.push(item);
        }
        items
    }

    /// Create an AsyncTask that collects this stream
    pub fn collect_async(self) -> crate::AsyncTask<Vec<T>>
    where
        T: Send + 'static,
    {
        crate::AsyncTask::from_future(self.collect())
    }

    /// Create AsyncStream from Result<T,E> stream with custom chunk handler
    pub fn from_result_stream_with_handler<S, E, H>(stream: S, mut handler: H) -> Self
    where
        S: Stream<Item = Result<T, E>> + Send + 'static,
        E: Send + 'static + std::fmt::Display,
        H: ChunkHandler<T> + Send + 'static,
        T: Send + 'static + BadAppleChunk,
    {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut stream = std::pin::pin!(stream);
            while let Some(result) = stream.next().await {
                let item = match result {
                    Ok(value) => handler.handle_chunk(value),
                    Err(error) => default_stream_error_handler(error.to_string()),
                };
                if tx.send(item).is_err() {
                    break;
                }
            }
        });
        Self { receiver: rx }
    }

    /// Create AsyncStream from Result<T,E> stream with default chunk handler
    pub fn from_result_stream_with_default_handler<S, E>(stream: S) -> Self
    where
        S: Stream<Item = Result<T, E>> + Send + 'static,
        E: Send + 'static + std::fmt::Display,
        T: Send + 'static + BadAppleChunk,
    {
        let mut handler = DefaultChunkHandler::<T>::default();
        Self::from_result_stream_with_handler(stream, handler)
    }
}

impl<T> Default for AsyncStream<T>
where
    T: NotResult, // T cannot be any Result type
{
    fn default() -> Self {
        let (_tx, rx) = mpsc::unbounded_channel();
        Self::new(rx)
    }
}

impl<T> Stream for AsyncStream<T>
where
    T: NotResult, // T cannot be any Result type
{
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}
