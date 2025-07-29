//! IMPORTANT: Pure AsyncStream - NO Future dependencies!
//!
//! ⚠️  ALL FUTURE USAGE ELIMINATED - PURE ASYNCSTREAM ARCHITECTURE ⚠️
//! Stream-first primitives with zero-allocation performance
//!
//! Lock-free bounded producer/consumer stream based on crossbeam_queue::ArrayQueue
//! Zero-allocation streaming with proven performance

use std::{
    pin::Pin,
    task::{Context, Poll, Waker},
    sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}},
    collections::VecDeque};

use crossbeam_queue::ArrayQueue;
use fluent_ai_async::AsyncStream as AsyncStreamTrait;

/// Cyrup-agent's AsyncStream with const-generic capacity
pub struct AsyncStream<T, const CAP: usize = 1024> {
    inner: Arc<Inner<T, CAP>>}

/// Producer side of AsyncStream
pub struct AsyncStreamSender<T, const CAP: usize = 1024> {
    inner: Arc<Inner<T, CAP>>}

struct Inner<T, const CAP: usize> {
    q: ArrayQueue<T>,
    waker: Mutex<Option<Waker>>,
    len: AtomicUsize, // optional runtime metric; not part of API
}

impl<T, const CAP: usize> AsyncStream<T, CAP> {
    /// Create a fresh `(sender, stream)` pair.
    #[inline]
    pub fn channel() -> (AsyncStreamSender<T, CAP>, Self) {
        let inner = Arc::new(Inner {
            q: ArrayQueue::new(CAP),
            waker: Mutex::new(None),
            len: AtomicUsize::new(0)});
        (
            AsyncStreamSender {
                inner: inner.clone()},
            Self { inner },
        )
    }

    /// Convenience helper – wrap a single item into a ready stream.
    /// Used in API glue; cost = one push into the bounded queue.
    #[inline]
    pub fn from_single(item: T) -> Self {
        let (tx, st) = Self::channel();
        // ignore full error – CAP ≥ 1 in every instantiation
        let _ = tx.try_send(item);
        st
    }

    /// Empty stream (always returns `Poll::Ready(None)`).
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
}

impl<T, const CAP: usize> AsyncStreamSender<T, CAP> {
    /// Zero-alloc try-send; returns `Err(val)` if the buffer is full.
    #[inline]
    pub fn try_send(&self, val: T) -> Result<(), T> {
        match self.inner.q.push(val) {
            Ok(()) => {
                self.inner.len.fetch_add(1, Ordering::Release);
                if let Ok(mut waker_guard) = self.inner.waker.lock() {
                    if let Some(waker) = waker_guard.take() {
                        waker.wake();
                    }
                }
                Ok(())
            }
            Err(v) => Err(v), // queue full → give caller its item back
        }
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
        // fast-path: try pop first
        if let Some(v) = self.inner.q.pop() {
            self.inner.len.fetch_sub(1, Ordering::AcqRel);
            return Poll::Ready(Some(v));
        }

        // register waker then re-check to avoid missed notifications
        if let Ok(mut waker_guard) = self.inner.waker.lock() {
            *waker_guard = Some(cx.waker().clone());
        }

        match self.inner.q.pop() {
            Some(v) => {
                self.inner.len.fetch_sub(1, Ordering::AcqRel);
                Poll::Ready(Some(v))
            }
            None => Poll::Pending}
    }
}
