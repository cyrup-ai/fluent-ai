//! Shared async utilities for the Desktop Commander project
//!
//! This module provides reusable async primitives that follow the project's
//! conventions of returning concrete types instead of boxed futures or async fn.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::oneshot;

/// Marker trait to prevent Result types in AsyncTask/AsyncStream
///
/// This trait is automatically implemented for all types except Result types.
/// It uses negative impls to explicitly exclude Result<T, E> from being used
/// in AsyncTask<T> or AsyncStream<T>.
pub auto trait NotResult {}

// Negative implementations - Result types do NOT implement NotResult
impl<T, E> !NotResult for Result<T, E> {}

/// Generic async task wrapper for single operations
///
/// This wraps a oneshot::Receiver and implements Future to provide
/// a concrete return type instead of boxed futures or async fn.
///
/// IMPORTANT: AsyncTask must never return Result types - all error handling
/// should be done internally before creating the AsyncTask.
pub struct AsyncTask<T>
where
    T: NotResult, // T cannot be any Result type
{
    receiver: oneshot::Receiver<T>,
}

impl<T> AsyncTask<T>
where
    T: NotResult, // T cannot be any Result type
{
    /// Create a new AsyncTask from a oneshot receiver
    pub fn new(receiver: oneshot::Receiver<T>) -> Self {
        Self { receiver }
    }

    /// Create an AsyncTask from a future
    pub fn from_future<F>(future: F) -> Self
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let result = future.await;
            let _ = tx.send(result);
        });
        Self { receiver: rx }
    }

    /// Create an AsyncTask from a value
    pub fn from_value(value: T) -> Self
    where
        T: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(value);
        Self { receiver: rx }
    }

    /// Create an AsyncTask by spawning a closure
    pub fn spawn<F>(closure: F) -> Self
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let result = closure();
            let _ = tx.send(result);
        });
        Self { receiver: rx }
    }
}

impl<T> Future for AsyncTask<T>
where
    T: NotResult, // T cannot be any Result type
{
    type Output = Result<T, oneshot::error::RecvError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.receiver).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(Ok(result)),
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }
}
