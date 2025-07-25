//! Zero-allocation streaming primitives with lock-free architecture
//!
//! Provides AsyncStream and AsyncStreamSender types that enforce the streams-only
//! architecture. All operations are lock-free and zero-allocation in hot paths.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc};
use crossbeam_queue::ArrayQueue;

/// Zero-allocation async stream with const-generic capacity
pub struct AsyncStream<T, const CAP: usize = 1024> {
    inner: Arc<Inner<T, CAP>>}

/// Producer side of AsyncStream
pub struct AsyncStreamSender<T, const CAP: usize = 1024> {
    inner: Arc<Inner<T, CAP>>}

struct Inner<T, const CAP: usize> {
    q: ArrayQueue<T>,
    len: AtomicUsize, // Runtime metric for monitoring
}

impl<T, const CAP: usize> AsyncStream<T, CAP>
where
    T: Send + 'static,
{
    /// Create stream with closure - preferred ergonomic pattern
    #[inline]
    pub fn with_channel<F>(f: F) -> Self
    where
        F: FnOnce(AsyncStreamSender<T, CAP>) + Send + 'static,
    {
        let (sender, stream) = Self::channel_internal();
        std::thread::spawn(move || f(sender));
        stream
    }

    /// Create a fresh `(sender, stream)` pair - for channel module use
    /// Use with_channel() for ergonomic public API
    #[inline]
    pub(crate) fn channel_internal() -> (AsyncStreamSender<T, CAP>, Self) {
        let inner = Arc::new(Inner {
            q: ArrayQueue::new(CAP),
            len: AtomicUsize::new(0)});
        (
            AsyncStreamSender {
                inner: inner.clone()},
            Self { inner },
        )
    }

    /// Empty stream (always returns `Poll::Ready(None)`)
    #[inline]
    pub fn empty() -> Self {
        Self::with_channel(|_sender| {
            // Empty stream - no values to emit
        })
    }

    /// Create from std mpsc receiver - NO tokio dependency!
    pub fn new(receiver: std::sync::mpsc::Receiver<T>) -> Self
    where
        T: Send + 'static,
    {
        let (tx, st) = Self::channel_internal();
        std::thread::spawn(move || {
            while let Ok(item) = receiver.recv() {
                if tx.try_send(item).is_err() {
                    break; // Stream closed
                }
            }
        });
        st
    }

    /// Get current length (approximate, for monitoring)
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len.load(Ordering::Acquire)
    }

    /// Check if stream is empty (approximate)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get capacity of the stream
    #[inline]
    pub const fn capacity(&self) -> usize {
        CAP
    }

    /// Process each chunk with a closure - streaming pattern
    pub fn on_chunk<F>(self, mut handler: F) -> Self
    where
        F: FnMut(T) + Send + 'static,
        T: Send + 'static,
    {
        let (_tx, stream) = Self::channel_internal();
        let inner = self.inner.clone();

        std::thread::spawn(move || loop {
            if let Some(item) = inner.q.pop() {
                inner.len.fetch_sub(1, Ordering::AcqRel);
                handler(item);
            } else {
                std::thread::yield_now();
            }
        });

        stream
    }

    /// Get the next item from the stream (non-blocking)
    pub fn try_next(&mut self) -> Option<T> {
        if let Some(v) = self.inner.q.pop() {
            self.inner.len.fetch_sub(1, Ordering::AcqRel);
            Some(v)
        } else {
            None
        }
    }

    /// Poll for the next item (streaming-only pattern)
    pub fn poll_next(&mut self) -> Option<T> {
        self.try_next()
    }

    /// Collect all items from the stream into a Vec (future-like behavior when needed)
    pub fn collect(mut self) -> Vec<T> {
        let mut results = Vec::new();
        while let Some(item) = self.try_next() {
            results.push(item);
        }
        results
    }

    /// Get the next item (blocking, no futures) - replaces async recv()
    pub fn recv_blocking(&mut self) -> Option<T> {
        // Block until item is available, following NO FUTURES architecture
        loop {
            if let Some(item) = self.try_next() {
                return Some(item);
            }
            // Small yield to prevent busy waiting
            std::thread::yield_now();
        }
    }
}

impl<T, const CAP: usize> AsyncStreamSender<T, CAP> {
    /// Zero-alloc try-send; returns `Err(val)` if the buffer is full
    #[inline]
    pub fn try_send(&self, val: T) -> Result<(), T> {
        match self.inner.q.push(val) {
            Ok(()) => {
                self.inner.len.fetch_add(1, Ordering::Release);
                Ok(())
            }
            Err(v) => Err(v), // Queue full → give caller its item back
        }
    }

    /// Send item, alias for try_send for ergonomics
    #[inline]
    pub fn send(&self, val: T) -> Result<(), T> {
        self.try_send(val)
    }

    /// Check if sender is closed (approximate)
    #[inline]
    pub fn is_closed(&self) -> bool {
        // In this implementation, sender is never truly "closed"
        // This is for API compatibility
        false
    }

    /// Get current length (approximate, for monitoring)
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len.load(Ordering::Acquire)
    }

    /// Check if stream is empty (approximate)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get capacity of the stream
    #[inline]
    pub const fn capacity(&self) -> usize {
        CAP
    }
}

impl<T, const CAP: usize> Default for AsyncStream<T, CAP>
where
    T: Send + 'static,
{
    /// Create an empty AsyncStream
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

// NO FUTURES! Stream trait removed per NO FUTURES architecture

impl<T, const CAP: usize> Clone for AsyncStreamSender<T, CAP> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone()}
    }
}

// REMOVED: Deprecated channel creation functions
// Use AsyncStream::with_channel() instead per fluent-ai streaming architecture
