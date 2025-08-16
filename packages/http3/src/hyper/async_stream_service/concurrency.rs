//! Concurrency limiting layer implementation for AsyncStream services
//!
//! Provides concurrency limiting functionality equivalent to tower::limit::ConcurrencyLimitLayer
//! but using atomic counters and pure AsyncStream patterns.

use std::sync::Arc;

use fluent_ai_async::{AsyncStream, emit, spawn_task};
use http::Uri;

use super::core::{AsyncStreamLayer, AsyncStreamService, ConnResult};
use crate::hyper::connect::Conn;
use crate::hyper::error::BoxError;

/// Concurrency limiting layer - AsyncStream equivalent of tower::limit::ConcurrencyLimitLayer  
#[derive(Clone, Debug)]
pub struct AsyncStreamConcurrencyLayer {
    max_concurrent: usize,
    // Using atomic counter for concurrency limiting instead of semaphore
    active_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl AsyncStreamConcurrencyLayer {
    /// Create a new concurrency limiting layer with the specified maximum concurrent requests
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            active_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

impl<S> AsyncStreamLayer<S> for AsyncStreamConcurrencyLayer
where
    S: AsyncStreamService<Uri, Response = Conn, Error = BoxError> + Clone + Send + 'static,
{
    type Service = AsyncStreamConcurrencyService<S>;

    fn layer(&self, service: S) -> Self::Service {
        AsyncStreamConcurrencyService {
            inner: service,
            max_concurrent: self.max_concurrent,
            active_count: self.active_count.clone(),
        }
    }
}

/// Concurrency limiting service implementation
#[derive(Debug)]
pub struct AsyncStreamConcurrencyService<S> {
    inner: S,
    max_concurrent: usize,
    active_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl<S> AsyncStreamService<Uri> for AsyncStreamConcurrencyService<S>
where
    S: AsyncStreamService<Uri, Response = Conn, Error = BoxError> + Clone + Send + 'static,
{
    type Response = Conn;
    type Error = BoxError;

    fn is_ready(&mut self) -> bool {
        let current = self.active_count.load(std::sync::atomic::Ordering::Relaxed);
        current < self.max_concurrent && self.inner.is_ready()
    }

    fn call(&mut self, request: Uri) -> AsyncStream<ConnResult<Self::Response>> {
        let active_count = self.active_count.clone();
        let max_concurrent = self.max_concurrent;
        // Create a clone instead of using unsafe zeroed - this preserves the service state
        let mut inner_service = self.inner.clone();

        AsyncStream::with_channel(move |sender| {
            let _task = spawn_task(move || {
                // Try to acquire a slot
                loop {
                    let current = active_count.load(std::sync::atomic::Ordering::Acquire);
                    if current >= max_concurrent {
                        // At limit, emit error and return
                        emit!(sender, ConnResult::error("concurrency limit exceeded"));
                        return;
                    }

                    // Try to increment the counter
                    if active_count
                        .compare_exchange(
                            current,
                            current + 1,
                            std::sync::atomic::Ordering::Release,
                            std::sync::atomic::Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        break;
                    }
                }

                // Execute with slot acquired
                let inner_stream = inner_service.call(request);
                let stream_iter = inner_stream;

                let result = if let Some(conn_result) = stream_iter.try_next() {
                    conn_result
                } else {
                    ConnResult::error("no connection result from inner service")
                };

                // Release the slot
                active_count.fetch_sub(1, std::sync::atomic::Ordering::Release);

                emit!(sender, result);
            });

            // Task execution is handled within the spawn_task closure
            // No additional result handling needed here
        })
    }
}
