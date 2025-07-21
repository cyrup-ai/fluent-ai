//! IMPORTANT: Cyrup-agent's battle-tested async primitives COPIED INTO fluent-ai
//!
//! ⚠️  DO NOT IMPORT FROM cyrup-agent - IT WILL BE DELETED! ⚠️
//! All async primitives are now part of fluent-ai directly
//!
//! Zero-allocation, crossbeam-based async primitives with proven performance
//! No NotResult constraints - errors handled internally

use core::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use crossbeam_channel::{Receiver, Sender, bounded};

use super::thread_pool::GLOBAL_EXECUTOR;

/// Cyrup-agent's battle-tested AsyncTask
/// Zero-allocation one-shot future built on crossbeam
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

impl<T> Future for AsyncTask<T> {
    type Output = T;

    #[inline]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // fast path – non-blocking recv
        if let Ok(v) = self.rx.try_recv() {
            return Poll::Ready(v);
        }

        // register waker & double-check
        GLOBAL_EXECUTOR.register_waker(self.rx.clone(), cx.waker().clone());

        match self.rx.try_recv() {
            Ok(v) => Poll::Ready(v),
            Err(crossbeam_channel::TryRecvError::Empty) => Poll::Pending,
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                panic!("AsyncTask sender dropped without sending")
            }
        }
    }
}

/// Spawn a future onto the global single-thread executor
/// and return an `AsyncTask` that resolves to its output.
#[inline]
pub fn spawn_async<Fut, T>(fut: Fut) -> AsyncTask<T>
where
    Fut: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx): (Sender<T>, Receiver<T>) = bounded(1);

    GLOBAL_EXECUTOR.enqueue(async move {
        let out = fut.await;
        let _ = tx.send(out); // ignore error – receiver may be gone
    });

    AsyncTask { rx }
}
