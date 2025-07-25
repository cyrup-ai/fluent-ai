//! Canonical channel types and helpers for the streams-only architecture.
//!
//! This module provides channel creation utilities that are fully compliant with
//! the NO FUTURES/No Result architecture using crossbeam primitives.

use crate::stream::{AsyncStream, AsyncStreamSender};

/// Creates a new asynchronous stream channel using the canonical AsyncStream architecture.
///
/// This is the primary factory for creating streams in the fluent-ai ecosystem.
/// It returns a sender and a receiver, where the receiver is already wrapped
/// in the canonical `AsyncStream` type.
///
/// # Example
/// ```rust
/// use fluent_ai_async::channel::channel;
/// 
/// let (sender, stream) = channel::<String>();
/// 
/// // Producer thread
/// std::thread::spawn(move || {
///     sender.send("Hello".to_string()).ok();
///     sender.send("World".to_string()).ok();
/// });
/// 
/// // Consumer can use stream patterns
/// let results: Vec<String> = stream.collect();
/// ```
#[inline]
pub fn channel<T>() -> (AsyncStreamSender<T>, AsyncStream<T>)
where
    T: Send + 'static,
{
    AsyncStream::channel_internal()
}

/// Creates a new asynchronous stream channel with custom capacity.
///
/// Uses const-generic capacity for zero-allocation patterns.
#[inline]
pub fn channel_with_capacity<T, const CAP: usize>() -> (AsyncStreamSender<T, CAP>, AsyncStream<T, CAP>)
where
    T: Send + 'static,
{
    AsyncStream::channel_internal()
}

/// Creates an unbounded channel (alias for channel with default capacity).
///
/// Provided for API compatibility - uses the default capacity of 1024.
#[inline]
pub fn unbounded_channel<T>() -> (AsyncStreamSender<T>, AsyncStream<T>)
where
    T: Send + 'static,
{
    channel()
}