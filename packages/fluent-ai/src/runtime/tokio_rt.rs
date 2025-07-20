// ============================================================================
// File: src/tokio_rt.rs
// ----------------------------------------------------------------------------
// Thin façade over Tokio that exposes **opaque Task handles** instead of
// leaking `tokio::task::JoinHandle`.  Synchronous callers can stay non-async
// by polling the returned future directly.
// ============================================================================

use std::{future::Future, pin::Pin};

use tokio::task::{JoinError, JoinHandle};

// ---------------------------------------------------------------------------
// Public, opaque handle
// ---------------------------------------------------------------------------
#[repr(transparent)]
pub struct Task<T>(JoinHandle<T>);

impl<T: Send + 'static> Future for Task<T> {
    type Output = Result<T, JoinError>;

    #[inline]
    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: transparent wrapper
        unsafe { self.map_unchecked_mut(|s| &mut s.0) }.poll(cx)
    }
}

impl<T> Task<T> {
    /// Cancel the underlying Tokio task.
    #[inline(always)]
    pub fn cancel(&self) {
        self.0.abort();
    }
}

/// Spawn onto the global Tokio runtime, returning an opaque [`Task`].
#[inline(always)]
pub fn spawn<F, T>(fut: F) -> Task<T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    Task(tokio::spawn(fut))
}

// ============================================================================
// End of file
// ============================================================================
