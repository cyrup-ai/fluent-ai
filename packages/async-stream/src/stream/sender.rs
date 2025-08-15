//! AsyncStreamSender - Producer side of the stream
//!
//! Zero-allocation sending with crossbeam primitives and parker/unparker signaling.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use cyrup_sugars::prelude::*;

use super::core::Inner;

/// Producer side of AsyncStream - cloneable for multiple producers
pub struct AsyncStreamSender<T, const CAP: usize = 1024> {
    pub(crate) inner: Arc<Inner<T, CAP>>,
}

impl<T, const CAP: usize> AsyncStreamSender<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    pub fn new(inner: Arc<Inner<T, CAP>>) -> Self {
        Self { inner }
    }

    /// Send a value to the stream (non-blocking)
    pub fn send(&self, value: T) -> Result<(), T> {
        match self.inner.queue.push(value) {
            Ok(()) => {
                self.inner.len.fetch_add(1, Ordering::Release);

                // Batched notification to reduce unparker overhead
                let pending = self
                    .inner
                    .pending_notifications
                    .fetch_add(1, Ordering::AcqRel)
                    + 1;
                let threshold = self.inner.notification_threshold.load(Ordering::Acquire);
                if pending >= threshold {
                    self.inner.pending_notifications.store(0, Ordering::Release);
                    self.inner.unparker.unpark();
                }

                Ok(())
            }
            Err(v) => Err(v),
        }
    }

    /// Send a Result to the stream (processes via chunk handler)
    pub fn send_result(&self, result: Result<T, String>) -> Result<(), T> {
        let processed_val = (self.inner.chunk_handler)(result);
        match self.inner.queue.push(processed_val) {
            Ok(()) => {
                self.inner.len.fetch_add(1, Ordering::Release);

                // Batched notification to reduce unparker overhead
                let pending = self
                    .inner
                    .pending_notifications
                    .fetch_add(1, Ordering::AcqRel)
                    + 1;
                let threshold = self.inner.notification_threshold.load(Ordering::Acquire);
                if pending >= threshold {
                    self.inner.pending_notifications.store(0, Ordering::Release);
                    self.inner.unparker.unpark();
                }

                Ok(())
            }
            Err(v) => Err(v),
        }
    }

    /// Try to send a value (non-blocking, returns immediately)
    pub fn try_send(&self, value: T) -> Result<(), T> {
        self.send(value)
    }

    /// Get current queue length
    pub fn len(&self) -> usize {
        self.inner.len.load(Ordering::Acquire)
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get queue capacity
    pub fn capacity(&self) -> usize {
        CAP
    }
}

impl<T, const CAP: usize> Clone for AsyncStreamSender<T, CAP> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}
