//! AsyncStream builder pattern implementation with ChunkHandler trait

use std::sync::mpsc;

use cyrup_sugars::prelude::*;

#[allow(unused_imports)]
use crate::stream::{AsyncStream, AsyncStreamCompletion, AsyncStreamSender};

/// Builder for AsyncStream with cyrup_sugars ChunkHandler pattern
pub struct AsyncStreamBuilder<T, const CAP: usize = 1024>
where
    T: MessageChunk + Send + 'static,
{
    chunk_handler: Box<dyn Fn(Result<T, String>) -> T + Send + Sync + 'static>,
}

impl<T, const CAP: usize> AsyncStreamBuilder<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    /// Create a new builder with default handler
    pub fn new() -> Self {
        Self {
            chunk_handler: Box::new(|result| match result {
                Ok(chunk) => chunk,
                Err(error) => {
                    log::error!("Stream error: {}", error);
                    T::bad_chunk(error)
                }
            }),
        }
    }

    /// Create stream with channel producer function
    pub fn with_channel<Func>(self, f: Func) -> AsyncStream<T, CAP>
    where
        Func: FnOnce(AsyncStreamSender<T, CAP>) + Send + 'static,
    {
        let (sender, stream) = AsyncStream::channel();
        std::thread::spawn(move || {
            f(sender);
            // Completion is signaled automatically when sender is dropped
        });
        stream
    }

    /// Create empty stream
    pub fn empty(self) -> AsyncStream<T, CAP> {
        self.with_channel(|_sender| {
            // Empty stream - no values to emit
        })
    }

    /// Create stream and sender pair
    pub fn channel(self) -> (AsyncStreamSender<T, CAP>, AsyncStream<T, CAP>) {
        AsyncStream::channel()
    }

    /// Create stream from mpsc receiver
    pub fn from_receiver(self, receiver: mpsc::Receiver<T>) -> AsyncStream<T, CAP> {
        let (sender, stream) = AsyncStream::channel();
        std::thread::spawn(move || {
            while let Ok(item) = receiver.recv() {
                if sender.send(item).is_err() {
                    break; // Stream closed
                }
            }
            // Completion is signaled automatically when sender is dropped
        });
        stream
    }

    /// Create stream with dynamic capacity (runtime configurable)
    pub fn with_dynamic_capacity(self) -> AsyncStream<T, CAP> {
        let (_sender, stream) = AsyncStream::channel_dynamic();
        stream
    }

    /// Create stream with dynamic capacity and channel producer function
    pub fn with_dynamic_channel<Func>(self, f: Func) -> AsyncStream<T, CAP>
    where
        Func: FnOnce(AsyncStreamSender<T, CAP>) + Send + 'static,
    {
        AsyncStream::with_dynamic_channel(f)
    }
}

impl<T, const CAP: usize> ChunkHandler<T, String> for AsyncStreamBuilder<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<T, String>) -> T + Send + Sync + 'static,
    {
        self.chunk_handler = Box::new(handler);
        self
    }
}
