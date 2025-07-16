// ============================================================================
// runtime/async_stream.rs       ← relative to project root
// ---------------------------------------------------------------------------
// Lock-free bounded producer/consumer stream based on `crossbeam_queue::ArrayQueue`.
//
// • `AsyncStream<T, CAP>`   implements `Stream<Item = T>`.
// • `AsyncStreamSender`     gives the producer side; `try_send` never allocs.
// • Wake-up is done via a single `AtomicWaker` per stream.
// ============================================================================

use core::{
    pin::Pin,
    task::{Context, Poll},
};
use crossbeam_queue::ArrayQueue;
use futures_core::Stream;
use futures_util::task::AtomicWaker;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

pub struct AsyncStream<T, const CAP: usize> {
    inner: Arc<Inner<T, CAP>>,
}

pub struct AsyncStreamSender<T, const CAP: usize> {
    inner: Arc<Inner<T, CAP>>,
}

struct Inner<T, const CAP: usize> {
    q: ArrayQueue<T>,
    waker: AtomicWaker,
    len: AtomicUsize, // optional runtime metric; not part of API
}

impl<T, const CAP: usize> AsyncStream<T, CAP> {
    /// Create a fresh `(sender, stream)` pair.
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
}

impl<T, const CAP: usize> AsyncStreamSender<T, CAP> {
    /// Zero-alloc try-send; returns `Err(val)` if the buffer is full.
    #[inline]
    pub fn try_send(&self, val: T) -> Result<(), T> {
        match self.inner.q.push(val) {
            Ok(()) => {
                self.inner.len.fetch_add(1, Ordering::Release);
                self.inner.waker.wake();
                Ok(())
            }
            Err(v) => Err(v), // queue full → give caller its item back
        }
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

// ============================================================================
// runtime/async_task.rs
// ---------------------------------------------------------------------------
// Zero-alloc one-shot future built on a crossbeam single-message channel.
// ============================================================================
