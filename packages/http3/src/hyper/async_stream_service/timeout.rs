//! Timeout layer implementation for AsyncStream services
//!
//! Provides timeout functionality equivalent to tower::timeout::TimeoutLayer
//! but using pure AsyncStream patterns with zero allocation.

use std::time::Duration;

use fluent_ai_async::{AsyncStream, emit, spawn_task};
use http::Uri;

use super::core::{AsyncStreamLayer, AsyncStreamService, ConnResult};
use crate::hyper::connect::Conn;
use crate::hyper::error::BoxError;

/// Timeout layer - AsyncStream equivalent of tower::timeout::TimeoutLayer
#[derive(Clone, Debug)]
pub struct AsyncStreamTimeoutLayer {
    timeout: Duration,
}

impl AsyncStreamTimeoutLayer {
    /// Create a new timeout layer with the specified duration
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }
}

impl<S> AsyncStreamLayer<S> for AsyncStreamTimeoutLayer
where
    S: AsyncStreamService<Uri, Response = Conn, Error = BoxError> + Clone + Send + 'static,
{
    type Service = AsyncStreamTimeoutService<S>;

    fn layer(&self, service: S) -> Self::Service {
        AsyncStreamTimeoutService {
            inner: service,
            timeout: self.timeout,
        }
    }
}

/// Timeout service implementation
#[derive(Debug)]
pub struct AsyncStreamTimeoutService<S> {
    inner: S,
    timeout: Duration,
}

impl<S> AsyncStreamService<Uri> for AsyncStreamTimeoutService<S>
where
    S: AsyncStreamService<Uri, Response = Conn, Error = BoxError> + Clone + Send + 'static,
{
    type Response = Conn;
    type Error = BoxError;

    fn is_ready(&mut self) -> bool {
        self.inner.is_ready()
    }

    fn call(&mut self, request: Uri) -> AsyncStream<ConnResult<Self::Response>> {
        let timeout = self.timeout;
        // Create a clone instead of using unsafe zeroed - this preserves the service state
        let mut inner_service = self.inner.clone();

        AsyncStream::with_channel(move |sender| {
            let _task = spawn_task(move || {
                let start = std::time::Instant::now();
                let inner_stream = inner_service.call(request);
                let stream_iter = inner_stream;

                loop {
                    if start.elapsed() > timeout {
                        emit!(sender, ConnResult::timeout());
                        return;
                    }

                    if let Some(conn_result) = stream_iter.try_next() {
                        emit!(sender, conn_result);
                        return;
                    }

                    // Small sleep to prevent busy waiting
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            });

            // Task execution is handled within the spawn_task closure
            // No additional result handling needed here
        })
    }
}
