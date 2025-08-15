//! Bridge between AsyncStream and Future for Service trait compatibility

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use fluent_ai_async::AsyncStream;
use fluent_ai_async::prelude::MessageChunk;

/// A Future that wraps an AsyncStream and polls it to completion
///
/// This enables Service trait compatibility by providing a Future interface
/// over the internal streams-first architecture
pub struct StreamFuture<T: MessageChunk + Default + Send + 'static> {
    stream: AsyncStream<T>,
    completed: bool,
}

impl<T: MessageChunk + Default + Send + 'static> StreamFuture<T> {
    /// Create a new StreamFuture from an AsyncStream
    pub fn new(stream: AsyncStream<T>) -> Self {
        Self {
            stream,
            completed: false,
        }
    }
}

impl<T: MessageChunk + Default + Send + 'static> Future for StreamFuture<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.completed {
            return Poll::Pending;
        }

        match self.stream.try_next() {
            Some(value) => {
                self.completed = true;
                Poll::Ready(value)
            }
            None => Poll::Pending,
        }
    }
}

impl<T: MessageChunk + Default + Send + 'static> From<AsyncStream<T>> for StreamFuture<T> {
    fn from(stream: AsyncStream<T>) -> Self {
        Self::new(stream)
    }
}
