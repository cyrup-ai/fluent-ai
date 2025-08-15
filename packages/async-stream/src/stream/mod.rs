//! AsyncStream module - Zero-allocation streaming primitives
//!
//! This module provides the core streaming functionality with lock-free performance.

mod compat;
mod core;
mod receiver;
mod sender;

pub use core::{AsyncStreamCompletion, Inner};
use std::sync::Arc;

pub use compat::AsyncStreamIterator;
use cyrup_sugars::prelude::*;
pub use receiver::AsyncStream;
pub use sender::AsyncStreamSender;

impl<T, const CAP: usize> AsyncStream<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    /// Create a builder for AsyncStream
    pub fn builder() -> crate::builder::AsyncStreamBuilder<T, CAP> {
        crate::builder::AsyncStreamBuilder::new()
    }
    /// Create a new AsyncStream with a producer closure
    pub fn with_channel<F>(producer: F) -> Self
    where
        F: FnOnce(AsyncStreamSender<T, CAP>) + Send + 'static,
    {
        let chunk_handler = Box::new(|result: Result<T, String>| match result {
            Ok(val) => val,
            Err(err) => T::bad_chunk(err),
        });

        let (inner, parker) = core::Inner::new(chunk_handler);
        let sender = AsyncStreamSender::new(Arc::clone(&inner));
        let completion = AsyncStreamCompletion::new(Arc::clone(&inner));

        std::thread::spawn(move || {
            producer(sender);
            drop(completion); // Signal completion when producer finishes
        });

        AsyncStream::new(inner, parker)
    }

    /// Create a channel pair for AsyncStream
    pub fn channel() -> (AsyncStreamSender<T, CAP>, Self) {
        let chunk_handler = Box::new(|result: Result<T, String>| match result {
            Ok(val) => val,
            Err(err) => T::bad_chunk(err),
        });

        let (inner, parker) = core::Inner::new(chunk_handler);
        let sender = AsyncStreamSender::new(Arc::clone(&inner));
        let stream = AsyncStream::new(inner, parker);

        (sender, stream)
    }

    /// Create a channel with custom chunk handler
    pub fn channel_with_handler<F>(chunk_handler: F) -> (AsyncStreamSender<T, CAP>, Self)
    where
        F: Fn(Result<T, String>) -> T + Send + Sync + 'static,
    {
        let (inner, parker) = core::Inner::new(Box::new(chunk_handler));
        let sender = AsyncStreamSender::new(Arc::clone(&inner));
        let stream = AsyncStream::new(inner, parker);

        (sender, stream)
    }

    /// Create a channel pair with dynamic capacity
    pub fn channel_dynamic() -> (AsyncStreamSender<T, CAP>, Self) {
        let chunk_handler = Box::new(|result: Result<T, String>| match result {
            Ok(val) => val,
            Err(err) => T::bad_chunk(err),
        });

        let (inner, parker) = core::Inner::new_dynamic(chunk_handler);
        let sender = AsyncStreamSender::new(Arc::clone(&inner));
        let stream = AsyncStream::new(inner, parker);

        (sender, stream)
    }

    /// Create an AsyncStream with dynamic capacity and producer closure
    pub fn with_dynamic_channel<F>(producer: F) -> Self
    where
        F: FnOnce(AsyncStreamSender<T, CAP>) + Send + 'static,
    {
        let chunk_handler = Box::new(|result: Result<T, String>| match result {
            Ok(val) => val,
            Err(err) => T::bad_chunk(err),
        });

        let (inner, parker) = core::Inner::new_dynamic(chunk_handler);
        let sender = AsyncStreamSender::new(Arc::clone(&inner));
        let completion = AsyncStreamCompletion::new(Arc::clone(&inner));

        std::thread::spawn(move || {
            producer(sender);
            drop(completion); // Signal completion when producer finishes
        });

        AsyncStream::new(inner, parker)
    }
}
