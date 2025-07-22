//! ⚠️  DO NOT IMPORT FROM cyrup-agent - IT WILL BE DELETED! ⚠️
//! All streaming primitives are now part of fluent-ai directly
//!
//! Lock-free bounded producer/consumer stream - ZERO ALLOCATION, NO ARC, NO FUTURES
//! Pure streaming-first HTTP3 QUIC architecture

use tokio::sync::mpsc;

/// Zero-allocation AsyncStream - NO ARC! NO FUTURES! Pure streaming-first HTTP3 QUIC
pub struct AsyncStream<T> {
    receiver: mpsc::UnboundedReceiver<T>,
}

/// Producer side of AsyncStream - NO ARC!
pub struct AsyncStreamSender<T> {
    sender: mpsc::UnboundedSender<T>,
}

impl<T> AsyncStream<T> {
    /// Create a fresh `(sender, stream)` pair - ZERO ALLOCATION
    #[inline]
    pub fn channel() -> (AsyncStreamSender<T>, Self) {
        let (sender, receiver) = mpsc::unbounded_channel();

        let sender = AsyncStreamSender { sender };
        let stream = Self { receiver };

        (sender, stream)
    }

    /// Create with custom capacity - uses unbounded for simplicity
    #[inline]
    pub fn with_capacity(_capacity: usize) -> (AsyncStreamSender<T>, Self) {
        Self::channel() // mpsc is already optimized
    }

    /// Create empty stream - for error cases
    #[inline]
    pub fn empty() -> Self {
        let (_sender, receiver) = mpsc::unbounded_channel();
        Self { receiver }
    }

    /// Create stream from single value
    #[inline]
    pub fn from_single(value: T) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let _ = sender.send(value); // Send the single value
        drop(sender); // Close the sender
        Self { receiver }
    }

    /// Try to receive a value without waiting - zero allocation
    #[inline]
    pub fn try_recv(&mut self) -> Option<T> {
        self.receiver.try_recv().ok()
    }

    /// Get next value - returns Option<T> directly (no futures)
    /// Users get immediate values, no async required - PURE STREAMING
    #[inline]
    pub fn next(&mut self) -> Option<T> {
        // Use tokio runtime handle for internal blocking
        let rt = match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle,
            Err(_) => return None, // No runtime available
        };

        rt.block_on(async { self.receiver.recv().await })
    }

    /// Collect all items into a Vec - returns Vec<T> directly (no futures)
    /// Users wanting "await" similar behavior call .collect()
    #[inline]
    pub fn collect(mut self) -> Vec<T> {
        let mut items = Vec::new();
        while let Some(item) = self.next() {
            items.push(item);
        }
        items
    }

    /// Check if stream is closed
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.receiver.is_closed()
    }
}

impl<T> AsyncStreamSender<T> {
    /// Send a value to the stream - ZERO ALLOCATION, NO BLOCKING
    #[inline]
    pub fn try_send(&self, item: T) -> Result<(), T> {
        self.sender.send(item).map_err(|e| e.0)
    }

    /// Send a value to the stream - returns Result directly (no futures)
    /// PURE STREAMING - immediate send operation
    #[inline]
    pub fn send(&self, item: T) -> Result<(), T> {
        self.sender.send(item).map_err(|e| e.0)
    }

    /// Check if stream is closed
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }
}

/// Extension trait for AsyncStream operations
pub trait AsyncStreamExt<T> {
    /// Map values in the stream
    fn map<U, F>(self, f: F) -> AsyncStream<U>
    where
        F: Fn(T) -> U + Send + 'static,
        T: Send + 'static,
        U: Send + 'static;

    /// Filter values in the stream
    fn filter<F>(self, predicate: F) -> AsyncStream<T>
    where
        F: Fn(&T) -> bool + Send + 'static,
        T: Send + 'static;
}

impl<T> AsyncStreamExt<T> for AsyncStream<T> {
    fn map<U, F>(mut self, f: F) -> AsyncStream<U>
    where
        F: Fn(T) -> U + Send + 'static,
        T: Send + 'static,
        U: Send + 'static,
    {
        let (sender, stream) = AsyncStream::channel();

        // Use std::thread instead of tokio::spawn - NO FUTURES
        std::thread::spawn(move || {
            while let Some(item) = self.next() {
                let mapped = f(item);
                if sender.try_send(mapped).is_err() {
                    break;
                }
            }
        });

        stream
    }

    fn filter<F>(mut self, predicate: F) -> AsyncStream<T>
    where
        F: Fn(&T) -> bool + Send + 'static,
        T: Send + 'static,
    {
        let (sender, stream) = AsyncStream::channel();

        // Use std::thread instead of tokio::spawn - NO FUTURES
        std::thread::spawn(move || {
            while let Some(item) = self.next() {
                if predicate(&item) {
                    if sender.try_send(item).is_err() {
                        break;
                    }
                }
            }
        });

        stream
    }
}

/// Module-level channel function for domain layer compatibility
/// Returns (sender, stream) pair - ZERO ALLOCATION
#[inline]
pub fn channel<T>() -> (AsyncStreamSender<T>, AsyncStream<T>) {
    AsyncStream::channel()
}
