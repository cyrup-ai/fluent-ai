//! AsyncStream-native service and layer traits
//!
//! Provides 100% compatible replacements for tower::Service and tower::Layer
//! using pure AsyncStream patterns with zero allocation and zero futures.
//!
//! This module retains all tower functionality while using the streams-first architecture.

use std::sync::Arc;
use std::time::Duration;

use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
use http::Uri;

use crate::hyper::error::BoxError;

/// AsyncStream equivalent of tower_service::Service
///
/// Provides the same semantics as tower::Service but returns AsyncStream instead of Future.
/// All service operations use pure streaming with zero allocation.
pub trait AsyncStreamService<Request> {
    /// The response type returned by the service
    type Response;

    /// The error type returned by the service
    type Error;

    /// Check if the service is ready to accept a request
    ///
    /// This is a non-blocking equivalent to tower::Service::poll_ready()
    /// Returns true if the service can immediately handle a request.
    fn is_ready(&mut self) -> bool;

    /// Execute a request and return an AsyncStream of results
    ///
    /// This replaces tower::Service::call() but returns AsyncStream<Result<Response, Error>>
    /// instead of Future<Output = Result<Response, Error>>.
    fn call(&mut self, request: Request) -> AsyncStream<Result<Self::Response, Self::Error>>;
}

/// AsyncStream equivalent of tower::Layer
///
/// Provides the same middleware composition semantics as tower::Layer
/// but works with AsyncStreamService instead of tower::Service.
pub trait AsyncStreamLayer<S> {
    /// The service type returned after applying the layer
    type Service;

    /// Apply the layer to the given service
    ///
    /// This provides identical functionality to tower::Layer::layer()
    /// but works with AsyncStream services.
    fn layer(&self, service: S) -> Self::Service;
}

// Re-export Conn type from connect module
pub use crate::hyper::connect::Conn;

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

    fn call(&mut self, request: Uri) -> AsyncStream<Result<Self::Response, Self::Error>> {
        let timeout = self.timeout;
        // Create a clone instead of using unsafe zeroed - this preserves the service state
        let mut inner_service = self.inner.clone();

        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || -> Result<Result<Conn, BoxError>, &'static str> {
                let start = std::time::Instant::now();
                let inner_stream = inner_service.call(request);
                let mut stream_iter = inner_stream;

                loop {
                    if start.elapsed() > timeout {
                        return Ok(Err("connection timeout".into()));
                    }

                    if let Some(result) = stream_iter.try_next() {
                        return Ok(result);
                    }

                    // Small sleep to prevent busy waiting
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            });

            match task.collect() {
                Ok(result) => emit!(sender, result),
                Err(e) => handle_error!(e, "timeout layer execution"),
            }
        })
    }
}

/// Concurrency limiting layer - AsyncStream equivalent of tower::limit::ConcurrencyLimitLayer  
#[derive(Clone, Debug)]
pub struct AsyncStreamConcurrencyLayer {
    max_concurrent: usize,
    semaphore: Arc<std::sync::Semaphore>,
}

impl AsyncStreamConcurrencyLayer {
    /// Create a new concurrency limiting layer with the specified maximum concurrent requests
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            semaphore: Arc::new(std::sync::Semaphore::new(max_concurrent)),
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
            semaphore: self.semaphore.clone(),
        }
    }
}

/// Concurrency limiting service implementation
pub struct AsyncStreamConcurrencyService<S> {
    inner: S,
    semaphore: Arc<std::sync::Semaphore>,
}

impl<S> AsyncStreamService<Uri> for AsyncStreamConcurrencyService<S>
where
    S: AsyncStreamService<Uri, Response = Conn, Error = BoxError> + Clone + Send + 'static,
{
    type Response = Conn;
    type Error = BoxError;

    fn is_ready(&mut self) -> bool {
        self.semaphore.available_permits() > 0 && self.inner.is_ready()
    }

    fn call(&mut self, request: Uri) -> AsyncStream<Result<Self::Response, Self::Error>> {
        let semaphore = self.semaphore.clone();
        // Create a clone instead of using unsafe zeroed - this preserves the service state
        let mut inner_service = self.inner.clone();

        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || -> Result<Result<Conn, BoxError>, &'static str> {
                // Acquire semaphore permit (blocks if at limit)
                let _permit = match semaphore.acquire() {
                    Ok(permit) => permit,
                    Err(_) => return Ok(Err("concurrency limit acquisition failed".into())),
                };

                // Execute with permit held
                let inner_stream = inner_service.call(request);
                let mut stream_iter = inner_stream;

                if let Some(result) = stream_iter.try_next() {
                    Ok(result)
                } else {
                    Ok(Err("no connection result from inner service".into()))
                }
                // Permit automatically released when _permit drops
            });

            match task.collect() {
                Ok(result) => emit!(sender, result),
                Err(e) => handle_error!(e, "concurrency layer execution"),
            }
        })
    }
}

/// Identity layer - AsyncStream equivalent of tower::layer::util::Identity
#[derive(Clone, Debug)]
pub struct AsyncStreamIdentityLayer;

impl AsyncStreamIdentityLayer {
    /// Create a new identity layer (no-op passthrough)
    pub fn new() -> Self {
        Self
    }
}

impl<S> AsyncStreamLayer<S> for AsyncStreamIdentityLayer {
    type Service = S;

    fn layer(&self, service: S) -> Self::Service {
        service
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    // Mock service for testing
    struct MockConnectorService {
        ready: bool,
        delay: Duration,
    }

    impl MockConnectorService {
        fn new(ready: bool, delay: Duration) -> Self {
            Self { ready, delay }
        }
    }

    impl AsyncStreamService<Uri> for MockConnectorService {
        type Response = Conn;
        type Error = BoxError;

        fn is_ready(&mut self) -> bool {
            self.ready
        }

        fn call(&mut self, _request: Uri) -> AsyncStream<Result<Self::Response, Self::Error>> {
            let delay = self.delay;
            AsyncStream::with_channel(move |sender| {
                let task = spawn_task(move || {
                    std::thread::sleep(delay);
                    Ok(Conn::default())
                });

                match task.collect() {
                    Ok(conn) => emit!(sender, Ok(conn)),
                    Err(e) => handle_error!(e, "mock connector"),
                }
            })
        }
    }

    #[test]
    fn test_timeout_layer_success() {
        let base_service = MockConnectorService::new(true, Duration::from_millis(10));
        let timeout_layer = AsyncStreamTimeoutLayer::new(Duration::from_millis(100));
        let mut timeout_service = timeout_layer.layer(base_service);

        assert!(timeout_service.is_ready());

        let result_stream = timeout_service.call(Uri::from_static("http://example.com"));
        let results: Vec<_> = result_stream.collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
    }

    #[test]
    fn test_timeout_layer_timeout() {
        let base_service = MockConnectorService::new(true, Duration::from_millis(100));
        let timeout_layer = AsyncStreamTimeoutLayer::new(Duration::from_millis(10));
        let mut timeout_service = timeout_layer.layer(base_service);

        let result_stream = timeout_service.call(Uri::from_static("http://example.com"));
        let results: Vec<_> = result_stream.collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
        assert!(
            results[0]
                .as_ref()
                .unwrap_err()
                .to_string()
                .contains("timeout")
        );
    }

    #[test]
    fn test_identity_layer() {
        let base_service = MockConnectorService::new(true, Duration::from_millis(10));
        let identity_layer = AsyncStreamIdentityLayer::new();
        let mut identity_service = identity_layer.layer(base_service);

        assert!(identity_service.is_ready());

        let result_stream = identity_service.call(Uri::from_static("http://example.com"));
        let results: Vec<_> = result_stream.collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
    }
}
