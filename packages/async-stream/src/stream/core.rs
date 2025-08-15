//! Core AsyncStream structures and shared state
//!
//! Zero-allocation, lock-free streaming primitives with crossbeam backing.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crossbeam_channel::{Receiver, Sender, bounded};
use crossbeam_queue::{ArrayQueue, SegQueue};
use crossbeam_utils::sync::{Parker, Unparker};

/// Queue variant for compile-time vs runtime capacity configuration
pub enum QueueVariant<T> {
    /// Fixed capacity known at compile-time (zero-allocation hot path)
    Fixed(ArrayQueue<T>),
    /// Dynamic capacity configurable at runtime
    Dynamic(SegQueue<T>),
}

impl<T> QueueVariant<T> {
    /// Push an item to the queue
    pub fn push(&self, item: T) -> Result<(), T> {
        match self {
            QueueVariant::Fixed(queue) => queue.push(item),
            QueueVariant::Dynamic(queue) => {
                queue.push(item);
                Ok(())
            }
        }
    }

    /// Pop an item from the queue
    pub fn pop(&self) -> Option<T> {
        match self {
            QueueVariant::Fixed(queue) => queue.pop(),
            QueueVariant::Dynamic(queue) => queue.pop(),
        }
    }

    /// Create a new fixed capacity queue
    pub fn new_fixed(capacity: usize) -> Self {
        QueueVariant::Fixed(ArrayQueue::new(capacity))
    }

    /// Create a new dynamic capacity queue
    pub fn new_dynamic() -> Self {
        QueueVariant::Dynamic(SegQueue::new())
    }
}

/// Shared state between all stream components
/// Uses Box::leak for zero-allocation hot paths
pub struct Inner<T, const CAP: usize> {
    pub queue: QueueVariant<T>,
    pub len: AtomicUsize,
    pub completion_tx: Sender<()>,
    pub completion_rx: Receiver<()>,
    pub chunk_handler: Box<dyn Fn(Result<T, String>) -> T + Send + Sync>,
    pub unparker: Unparker,
    pub notification_threshold: AtomicUsize,
    pub pending_notifications: AtomicUsize,
}

impl<T, const CAP: usize> Inner<T, CAP> {
    /// Create new Inner with fixed capacity (const-generic)
    pub fn new(
        chunk_handler: Box<dyn Fn(Result<T, String>) -> T + Send + Sync>,
    ) -> (Arc<Self>, Parker) {
        let (completion_tx, completion_rx) = bounded(1);
        let parker = Parker::new();
        let unparker = parker.unparker().clone();

        let inner = Arc::new(Inner {
            queue: QueueVariant::new_fixed(CAP),
            len: AtomicUsize::new(0),
            completion_tx,
            completion_rx,
            chunk_handler,
            unparker,
            notification_threshold: AtomicUsize::new(16), // Default batch size
            pending_notifications: AtomicUsize::new(0),
        });

        (inner, parker)
    }

    /// Create new Inner with dynamic capacity
    pub fn new_dynamic(
        chunk_handler: Box<dyn Fn(Result<T, String>) -> T + Send + Sync>,
    ) -> (Arc<Self>, Parker) {
        let (completion_tx, completion_rx) = bounded(1);
        let parker = Parker::new();
        let unparker = parker.unparker().clone();

        let inner = Arc::new(Inner {
            queue: QueueVariant::new_dynamic(),
            len: AtomicUsize::new(0),
            completion_tx,
            completion_rx,
            chunk_handler,
            unparker,
            notification_threshold: AtomicUsize::new(16), // Default batch size
            pending_notifications: AtomicUsize::new(0),
        });

        (inner, parker)
    }
}

/// Completion signaling for AsyncStream
pub struct AsyncStreamCompletion<T, const CAP: usize> {
    inner: Arc<Inner<T, CAP>>,
}

impl<T, const CAP: usize> AsyncStreamCompletion<T, CAP> {
    pub fn new(inner: Arc<Inner<T, CAP>>) -> Self {
        Self { inner }
    }

    /// Signal that the stream is complete (no more values will be sent)
    pub fn signal_completion(&self) {
        let _ = self.inner.completion_tx.try_send(());
        // Always unpark on completion to wake any waiting threads
        self.inner.unparker.unpark();
    }
}

impl<T, const CAP: usize> Drop for AsyncStreamCompletion<T, CAP> {
    fn drop(&mut self) {
        self.signal_completion();
    }
}
