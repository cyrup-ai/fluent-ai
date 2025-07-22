//! Pure streaming task primitives - NO Future/async/await!
//!
//! Zero-allocation, crossbeam-based streaming primitives with proven performance.
//! Stream-first design with .collect() for await-like behavior.

use crossbeam_channel::{bounded, Receiver, Sender};

use crate::stream::{AsyncStream, AsyncStreamSender};

/// Pure streaming task - NO Future implementation!
/// Zero-allocation one-shot streaming built on crossbeam
pub struct AsyncTask<T> {
    rx: Receiver<T>,
}

impl<T> AsyncTask<T>
where
    T: Send + 'static,
{
    /// Create AsyncTask from crossbeam receiver for first-class streaming
    #[inline]
    pub fn new(rx: Receiver<T>) -> Self {
        AsyncTask { rx }
    }

    /// Create an AsyncTask that immediately resolves to the given value
    #[inline]
    pub fn from_value(value: T) -> Self {
        let (tx, rx): (Sender<T>, Receiver<T>) = bounded(1);
        let _ = tx.send(value); // Send immediately
        AsyncTask { rx }
    }

    /// Collect the result (blocking) - replaces .await behavior
    /// This is the primary method for await-like usage
    #[inline]
    pub fn collect(self) -> T {
        self.rx
            .recv()
            .expect("AsyncTask sender dropped without sending")
    }

    /// Non-blocking try to receive result
    #[inline]
    pub fn try_collect(&self) -> Result<T, crossbeam_channel::TryRecvError> {
        self.rx.try_recv()
    }

    /// Get the underlying receiver for advanced streaming patterns
    #[inline]
    pub fn into_receiver(self) -> Receiver<T> {
        self.rx
    }

    /// Check if task is ready (non-blocking)
    #[inline]
    pub fn is_ready(&self) -> bool {
        !self.rx.is_empty()
    }
}

/// Spawn a closure onto a thread and return an `AsyncTask` that resolves to its output.
/// Replaces spawn_async - NO FUTURE USAGE!
#[inline]
pub fn spawn_task<F, T>(f: F) -> AsyncTask<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx): (Sender<T>, Receiver<T>) = bounded(1);

    std::thread::spawn(move || {
        let result = f();
        let _ = tx.send(result); // Ignore error – receiver may be gone
    });

    AsyncTask { rx }
}

/// Spawn a closure that produces multiple values as a stream
#[inline]
pub fn spawn_stream<F, T>(f: F) -> AsyncStream<T>
where
    F: FnOnce(AsyncStreamSender<T>) + Send + 'static,
    T: Send + 'static,
{
    let (sender, stream) = AsyncStream::channel();

    std::thread::spawn(move || {
        f(sender);
    });

    stream
}

/// Spawn a closure with custom stream capacity
#[inline]
pub fn spawn_stream_with_capacity<F, T, const CAP: usize>(f: F) -> AsyncStream<T, CAP>
where
    F: FnOnce(AsyncStreamSender<T, CAP>) + Send + 'static,
    T: Send + 'static,
{
    let (sender, stream) = AsyncStream::channel();

    std::thread::spawn(move || {
        f(sender);
    });

    stream
}
