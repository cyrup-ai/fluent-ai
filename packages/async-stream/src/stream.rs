//! Blazing-fast, zero-hot-path-allocation streaming primitives (lock-free)
//!
//! Provides AsyncStream for high-performance streaming with lock-free architecture.
//! Minimal allocation (one-time leaked Box) for shared state; zero allocation in hot paths.

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_queue::ArrayQueue;
use cyrup_sugars::prelude::*;

use crate::builder::AsyncStreamBuilder;

pub(crate) struct Inner<T, const CAP: usize>
where
    T: MessageChunk + Send + Default + 'static,
{
    queue: ArrayQueue<T>,
    len: AtomicUsize,
    completion_rx: crossbeam_channel::Receiver<()>,
    chunk_handler: Box<dyn Fn(Result<T, String>) -> T + Send + Sync + 'static>,
}

/// Completion sender for signaling when producer is done (internal only)
pub(crate) struct AsyncStreamCompletion {
    completion_tx: crossbeam_channel::Sender<()>,
}

impl AsyncStreamCompletion {
    /// Signal that the producer is done
    pub(crate) fn signal_completion(self) {
        let _ = self.completion_tx.send(());
    }
}

/// Zero-hot-path-allocation async stream with const-generic capacity.
/// Assumes single receiver (multiple senders via clone); try_next not thread-safe for concurrent calls.
pub struct AsyncStream<T, const CAP: usize = 1024>
where
    T: MessageChunk + Send + Default + 'static,
{
    pub(crate) inner: &'static Inner<T, CAP>,
}

/// Sender half of an async stream channel
pub struct AsyncStreamSender<T, const CAP: usize = 1024>
where
    T: MessageChunk + Send + Default + 'static,
{
    pub(crate) inner: &'static Inner<T, CAP>,
}

impl<T, const CAP: usize> Clone for AsyncStreamSender<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    fn clone(&self) -> Self {
        Self { inner: self.inner }
    }
}

impl<T, const CAP: usize> AsyncStreamSender<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    /// Create new sender with given inner reference
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn new(inner: &'static Inner<T, CAP>) -> Self {
        Self { inner }
    }

    #[inline]
    pub fn try_send(&self, val: T) -> Result<(), T> {
        self.try_send_result(Ok(val))
    }

    #[inline]
    pub fn send(&self, val: T) -> Result<(), T> {
        self.send_result(Ok(val))
    }

    /// Try to send Result<T, String> through handler (cyrup_sugars pattern) - non-blocking
    #[inline]
    pub fn try_send_result(&self, result: Result<T, String>) -> Result<(), T> {
        // Apply handler - always present, zero allocation, blazing-fast
        let processed_val = (self.inner.chunk_handler)(result);

        match self.inner.queue.push(processed_val) {
            Ok(()) => {
                self.inner.len.fetch_add(1, Ordering::Release);
                Ok(())
            }
            Err(v) => Err(v), // Queue full → give caller its item back
        }
    }

    /// Send Result<T, String> through handler (cyrup_sugars pattern)
    #[inline]
    pub fn send_result(&self, result: Result<T, String>) -> Result<(), T> {
        // Apply handler - always present, zero allocation, blazing-fast
        let processed_val = (self.inner.chunk_handler)(result);

        match self.inner.queue.push(processed_val) {
            Ok(()) => {
                self.inner.len.fetch_add(1, Ordering::Release);
                Ok(())
            }
            Err(v) => Err(v),
        }
    }

    /// Send error through cyrup_sugars pattern (creates bad chunk)
    #[inline]
    pub fn send_error(&self, error: String) -> Result<(), T> {
        self.send_result(Err(error))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, const CAP: usize> AsyncStream<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    /// Create stream with ChunkHandler (cyrup_sugars pattern)
    pub(crate) fn channel_with_handler(
        handler: Box<dyn Fn(Result<T, String>) -> T + Send + Sync + 'static>,
    ) -> (Self, AsyncStreamCompletion) {
        let (completion_tx, completion_rx) = crossbeam_channel::bounded(1);
        let inner = Box::leak(Box::new(Inner {
            queue: ArrayQueue::new(CAP),
            len: AtomicUsize::new(0),
            completion_rx,
            chunk_handler: handler,
        }));
        let stream = Self { inner };
        let completion = AsyncStreamCompletion { completion_tx };
        (stream, completion)
    }

    /// Create stream and sender with default handler
    pub fn channel() -> (AsyncStreamSender<T, CAP>, Self) {
        let (stream, _completion) = Self::channel_with_handler(Box::new(|result| match result {
            Ok(chunk) => chunk,
            Err(error) => {
                log::error!("Stream error: {}", error);
                T::bad_chunk(error)
            }
        }));
        let sender = AsyncStreamSender {
            inner: stream.inner,
        };
        (sender, stream)
    }

    /// Create a builder for configuring the stream
    pub fn builder() -> AsyncStreamBuilder<T, CAP> {
        AsyncStreamBuilder::new()
    }

    /// Create empty stream
    pub fn empty() -> Self {
        Self::builder().empty()
    }

    /// Create from mpsc receiver
    pub fn new(receiver: std::sync::mpsc::Receiver<T>) -> Self {
        Self::builder().from_receiver(receiver)
    }

    /// Create stream with channel producer function
    pub fn with_channel<Func>(f: Func) -> Self
    where
        Func: FnOnce(AsyncStreamSender<T, CAP>) + Send + 'static,
    {
        Self::builder().with_channel(f)
    }

    #[inline]
    pub fn try_next(&self) -> Option<T> {
        self.inner.queue.pop().map(|v| {
            self.inner.len.fetch_sub(1, Ordering::Release);
            v
        })
    }

    #[inline]
    pub async fn next(&self) -> Option<T> {
        loop {
            if let Some(val) = self.try_next() {
                return Some(val);
            }
            // Simple spin-wait for now without tokio dependency
            std::thread::yield_now();
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
impl<T, const CAP: usize> futures_util::Stream for AsyncStream<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    type Item = T;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<T>> {
        if let Some(val) = self.try_next() {
            std::task::Poll::Ready(Some(val))
        } else {
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
}

/// Iterator adapter for AsyncStream to enable for-loop syntax
pub struct AsyncStreamIterator<T, const CAP: usize>
where
    T: MessageChunk + Send + Default + 'static,
{
    stream: AsyncStream<T, CAP>,
}

impl<T, const CAP: usize> AsyncStream<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    /// Convert to iterator for for-loop usage
    pub fn into_iter(self) -> AsyncStreamIterator<T, CAP> {
        AsyncStreamIterator { stream: self }
    }

    /// Collect all items from the stream (blocking)
    pub fn collect<C>(self) -> C
    where
        C: FromIterator<T>,
    {
        let mut items = Vec::new();
        let backoff = crossbeam_utils::Backoff::new();

        loop {
            // Drain all available items
            while let Some(item) = self.try_next() {
                items.push(item);
                backoff.reset(); // Reset backoff when we get items
            }

            // Check if producer is done using completion channel
            match self.inner.completion_rx.try_recv() {
                Ok(()) => {
                    // Producer signaled completion - drain any remaining items and finish
                    while let Some(item) = self.try_next() {
                        items.push(item);
                    }
                    break;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    // Producer still working - use elite polling backoff
                    backoff.snooze();
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Producer dropped without signaling - drain remaining items and finish
                    while let Some(item) = self.try_next() {
                        items.push(item);
                    }
                    break;
                }
            }
        }

        items.into_iter().collect()
    }

    /// Collect all items from the stream with error handling (blocking)
    /// If any chunk is a BadChunk type, enters the else condition
    pub fn collect_or_else<C, F>(self, error_handler: F) -> C
    where
        C: FromIterator<T>,
        F: Fn(String) -> C,
        T: std::fmt::Debug,
    {
        let mut items = Vec::new();
        let backoff = crossbeam_utils::Backoff::new();

        loop {
            // Drain all available items and check for error chunks using MessageChunk trait
            while let Some(item) = self.try_next() {
                // Check if this item is an error chunk using the proper MessageChunk trait
                if item.is_error() {
                    // Found an error chunk - enter error condition
                    let error_msg = item
                        .error()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "Unknown error chunk detected".to_string());
                    return error_handler(error_msg);
                }
                items.push(item);
                backoff.reset(); // Reset backoff when we get items
            }

            // Check if producer is done using completion channel
            match self.inner.completion_rx.try_recv() {
                Ok(()) => {
                    // Producer signaled completion - drain any remaining items and check for error chunks
                    while let Some(item) = self.try_next() {
                        if item.is_error() {
                            if let Some(error_msg) = item.error() {
                                return error_handler(error_msg.to_string());
                            } else {
                                return error_handler("Unknown error chunk detected".to_string());
                            }
                        }
                        items.push(item);
                    }
                    break;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    // Producer still working - use elite polling backoff
                    backoff.snooze();
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Producer dropped - drain remaining items and check for error chunks
                    while let Some(item) = self.try_next() {
                        if item.is_error() {
                            if let Some(error_msg) = item.error() {
                                return error_handler(error_msg.to_string());
                            } else {
                                return error_handler("Unknown error chunk detected".to_string());
                            }
                        }
                        items.push(item);
                    }
                    break;
                }
            }
        }

        // No BadChunk found - return normal collection
        items.into_iter().collect()
    }
}

impl<T, const CAP: usize> Iterator for AsyncStreamIterator<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.stream.try_next()
    }
}
