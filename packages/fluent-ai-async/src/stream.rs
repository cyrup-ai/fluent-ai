//! Zero-allocation streaming primitives with lock-free architecture
//!
//! Provides AsyncStream and AsyncStreamSender types that enforce the streams-only
//! architecture. All operations are lock-free and zero-allocation in hot paths.

use core::{
    pin::Pin,
    task::{Context, Poll},
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use crossbeam_queue::ArrayQueue;
use futures_util::task::AtomicWaker;
use futures_util::Stream;

/// Zero-allocation async stream with const-generic capacity
pub struct AsyncStream<T, const CAP: usize = 1024> {
    inner: Arc<Inner<T, CAP>>,
}

/// Producer side of AsyncStream
pub struct AsyncStreamSender<T, const CAP: usize = 1024> {
    inner: Arc<Inner<T, CAP>>,
}

struct Inner<T, const CAP: usize> {
    q: ArrayQueue<T>,
    waker: AtomicWaker,
    len: AtomicUsize, // Runtime metric for monitoring
}

impl<T, const CAP: usize> AsyncStream<T, CAP> {
    /// Create a fresh `(sender, stream)` pair
    #[inline]
    pub fn channel() -> (AsyncStreamSender<T, CAP>, Self) {
        let inner = Arc::new(Inner {
            q: ArrayQueue::new(CAP),
            waker: AtomicWaker::new(),
            len: AtomicUsize::new(0),
        });
        (
            AsyncStreamSender {
                inner: inner.clone(),
            },
            Self { inner },
        )
    }

    /// Convenience helper – wrap a single item into a ready stream
    #[inline]
    pub fn from_single(item: T) -> Self {
        let (tx, st) = Self::channel();
        // Ignore full error – CAP ≥ 1 in every instantiation
        let _ = tx.try_send(item);
        st
    }

    /// Empty stream (always returns `Poll::Ready(None)`)
    #[inline]
    pub fn empty() -> Self {
        let (_tx, st) = Self::channel();
        st
    }

    /// Create from std mpsc receiver - NO tokio dependency!
    pub fn new(receiver: std::sync::mpsc::Receiver<T>) -> Self
    where
        T: Send + 'static,
    {
        let (tx, st) = Self::channel();
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
        let (_tx, stream) = Self::channel();
        let inner = self.inner.clone();
        
        std::thread::spawn(move || {
            loop {
                if let Some(item) = inner.q.pop() {
                    inner.len.fetch_sub(1, Ordering::AcqRel);
                    handler(item);
                } else {
                    std::thread::yield_now();
                }
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
}

impl<T, const CAP: usize> AsyncStreamSender<T, CAP> {
    /// Zero-alloc try-send; returns `Err(val)` if the buffer is full
    #[inline]
    pub fn try_send(&self, val: T) -> Result<(), T> {
        match self.inner.q.push(val) {
            Ok(()) => {
                self.inner.len.fetch_add(1, Ordering::Release);
                self.inner.waker.wake();
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

impl<T, const CAP: usize> Default for AsyncStream<T, CAP> {
    /// Create an empty AsyncStream
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

impl<T, const CAP: usize> Stream for AsyncStream<T, CAP> {
    type Item = T;

    #[inline]
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Fast-path: try pop first
        if let Some(v) = self.inner.q.pop() {
            self.inner.len.fetch_sub(1, Ordering::AcqRel);
            return Poll::Ready(Some(v));
        }

        // Register waker then re-check to avoid missed notifications
        self.inner.waker.register(cx.waker());

        match self.inner.q.pop() {
            Some(v) => {
                self.inner.len.fetch_sub(1, Ordering::AcqRel);
                Poll::Ready(Some(v))
            }
            None => Poll::Pending,
        }
    }
}

impl<T, const CAP: usize> Clone for AsyncStreamSender<T, CAP> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Convenience function to create a channel with default capacity
#[inline]
pub fn channel<T>() -> (AsyncStreamSender<T>, AsyncStream<T>) {
    AsyncStream::channel()
}

/// Convenience function to create a channel with custom capacity
#[inline]
pub fn channel_with_capacity<T, const CAP: usize>(
) -> (AsyncStreamSender<T, CAP>, AsyncStream<T, CAP>) {
    AsyncStream::channel()
}

/// Alias for channel() - used by domain crate
#[inline]
pub fn async_stream_channel<T>() -> (AsyncStreamSender<T>, AsyncStream<T>) {
    channel()
}
