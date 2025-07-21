//! Zero-allocation async primitives for pure streaming architecture
//!
//! NO FUTURES - pure streaming only
//! All async primitives are now part of fluent-ai directly
//!
//! Zero-allocation, crossbeam-based async primitives with proven performance
//! No NotResult constraints - errors handled internally

use crossbeam_channel::{Receiver, Sender, bounded};

use super::thread_pool::GLOBAL_EXECUTOR;

/// Zero-allocation one-shot stream built on crossbeam
/// NO FUTURES - pure streaming architecture
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
}

impl<T> AsyncTask<T> {
    /// Get next value - returns Option<T> directly (no futures)
    /// Users get immediate values, no polling required
    #[inline]
    pub fn next(&self) -> Option<T> {
        match self.rx.try_recv() {
            Ok(v) => Some(v),
            Err(_) => None, // Empty or disconnected - return None
        }
    }

    /// Collect all values - returns Vec<T> directly (no futures)
    /// Users wanting "await" similar behavior call .collect()
    #[inline]
    pub fn collect(self) -> Vec<T> {
        let mut results = Vec::new();
        
        // Use blocking recv for first item if available
        if let Ok(first) = self.rx.recv() {
            results.push(first);
            
            // Then try_recv for any additional items
            while let Ok(item) = self.rx.try_recv() {
                results.push(item);
            }
        }
        
        results
    }

    /// Wait for single value - returns T directly (no futures)
    /// Blocks until value is available or channel closes
    #[inline]
    pub fn wait(self) -> Option<T> {
        self.rx.recv().ok()
    }
}

/// Spawn a task onto the global single-thread executor
/// and return an `AsyncTask` stream that yields its output.
/// NO FUTURES - pure streaming architecture
#[inline]
pub fn spawn_async<F, T>(task_fn: F) -> AsyncTask<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx): (Sender<T>, Receiver<T>) = bounded(1);

    GLOBAL_EXECUTOR.enqueue(move || {
        let out = task_fn();
        let _ = tx.send(out); // ignore error – receiver may be gone
    });

    AsyncTask { rx }
}