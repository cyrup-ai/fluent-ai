//! AsyncStream - Consumer side of the stream
//!
//! Zero-allocation streaming with efficient parker-based waiting.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crossbeam_utils::{Backoff, sync::Parker};
use cyrup_sugars::prelude::*;

use super::core::Inner;

/// Consumer side of AsyncStream with zero-allocation streaming
pub struct AsyncStream<T, const CAP: usize = 1024> {
    pub(crate) inner: Arc<Inner<T, CAP>>,
    pub(crate) parker: Parker,
}

impl<T, const CAP: usize> AsyncStream<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    pub fn new(inner: Arc<Inner<T, CAP>>, parker: Parker) -> Self {
        Self { inner, parker }
    }

    /// Get next value from stream (blocking with efficient waiting)
    pub async fn next(&self) -> Option<T> {
        let backoff = Backoff::new();
        loop {
            if let Some(val) = self.try_next() {
                return Some(val);
            }
            match self.inner.completion_rx.try_recv() {
                Ok(()) => return self.try_next(),
                Err(crossbeam_channel::TryRecvError::Disconnected) => return self.try_next(),
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    if backoff.is_completed() {
                        self.parker.park();
                    } else {
                        backoff.snooze();
                    }
                }
            }
        }
    }

    /// Try to get next value (non-blocking)
    pub fn try_next(&self) -> Option<T> {
        match self.inner.queue.pop() {
            Some(val) => {
                self.inner.len.fetch_sub(1, Ordering::Release);
                Some(val)
            }
            None => None,
        }
    }

    /// Collect all items from the stream (blocking)
    pub fn collect(self) -> Vec<T> {
        let mut items = Vec::with_capacity(self.len());
        let backoff = Backoff::new();

        // Collect all currently available items
        while let Some(item) = self.try_next() {
            items.push(item);
        }

        // Wait for completion or more items
        loop {
            match self.inner.completion_rx.try_recv() {
                Ok(()) => {
                    // Stream completed, collect any remaining items
                    while let Some(item) = self.try_next() {
                        items.push(item);
                    }
                    break;
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Sender dropped, collect remaining items
                    while let Some(item) = self.try_next() {
                        items.push(item);
                    }
                    break;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    // Check for new items before backing off
                    if let Some(item) = self.try_next() {
                        items.push(item);
                        continue;
                    }
                    if backoff.is_completed() {
                        self.parker.park();
                    } else {
                        backoff.snooze();
                    }
                }
            }
        }

        items
    }

    /// Collect the first item from the stream (blocking)
    pub fn collect_one(self) -> T {
        let backoff = Backoff::new();
        loop {
            if let Some(item) = self.try_next() {
                return item;
            }
            match self.inner.completion_rx.try_recv() {
                Ok(()) => {
                    // Stream completed, try one more time
                    if let Some(item) = self.try_next() {
                        return item;
                    }
                    // If no item available, return default
                    return T::default();
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Sender dropped, try one more time
                    if let Some(item) = self.try_next() {
                        return item;
                    }
                    return T::default();
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    if backoff.is_completed() {
                        self.parker.park();
                    } else {
                        backoff.snooze();
                    }
                }
            }
        }
    }

    /// Collect the first item from the stream with error handling (blocking)
    pub fn collect_one_or_else<F>(self, error_handler: F) -> T
    where
        F: FnOnce(&T) -> T,
    {
        let backoff = Backoff::new();
        loop {
            if let Some(item) = self.try_next() {
                if item.is_error() {
                    return error_handler(&item);
                }
                return item;
            }
            match self.inner.completion_rx.try_recv() {
                Ok(()) => {
                    // Stream completed, try one more time
                    if let Some(item) = self.try_next() {
                        if item.is_error() {
                            return error_handler(&item);
                        }
                        return item;
                    }
                    // If no item available, return default
                    return T::default();
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Sender dropped, try one more time
                    if let Some(item) = self.try_next() {
                        if item.is_error() {
                            return error_handler(&item);
                        }
                        return item;
                    }
                    return T::default();
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    if backoff.is_completed() {
                        self.parker.park();
                    } else {
                        backoff.snooze();
                    }
                }
            }
        }
    }

    /// Collect all items from the stream with error handling (blocking)
    pub fn collect_or_else<F>(self, mut error_handler: F) -> Vec<T>
    where
        F: FnMut(&T) -> T,
    {
        let mut items = Vec::with_capacity(self.len());
        let backoff = Backoff::new();

        // Collect all currently available items
        while let Some(item) = self.try_next() {
            if item.is_error() {
                return vec![error_handler(&item)];
            }
            items.push(item);
        }

        // Wait for completion or more items
        loop {
            match self.inner.completion_rx.try_recv() {
                Ok(()) => {
                    // Stream completed, collect any remaining items
                    while let Some(item) = self.try_next() {
                        if item.is_error() {
                            return vec![error_handler(&item)];
                        }
                        items.push(item);
                    }
                    break;
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Sender dropped, collect remaining items
                    while let Some(item) = self.try_next() {
                        if item.is_error() {
                            return vec![error_handler(&item)];
                        }
                        items.push(item);
                    }
                    break;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    // Check for new items before backing off
                    if let Some(item) = self.try_next() {
                        if item.is_error() {
                            return vec![error_handler(&item)];
                        }
                        items.push(item);
                        continue;
                    }
                    if backoff.is_completed() {
                        self.parker.park();
                    } else {
                        backoff.snooze();
                    }
                }
            }
        }

        // No BadChunk found - return normal collection
        items
    }

    /// Get current queue length
    pub fn len(&self) -> usize {
        self.inner.len.load(Ordering::Acquire)
    }

    /// Check if stream is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get queue capacity
    pub fn capacity(&self) -> usize {
        CAP
    }
}
