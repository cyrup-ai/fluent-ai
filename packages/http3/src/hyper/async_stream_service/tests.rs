//! Tests for AsyncStream service implementations
//!
//! Comprehensive test suite for timeout, concurrency, and identity layers
//! using mock services to verify functionality.

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
    use http::Uri;

    use super::super::concurrency::{AsyncStreamConcurrencyLayer, AsyncStreamConcurrencyService};
    use super::super::core::AsyncStreamLayer;
    use super::super::core::{AsyncStreamService, ConnResult};
    use super::super::identity::AsyncStreamIdentityLayer;
    use super::super::timeout::{AsyncStreamTimeoutLayer, AsyncStreamTimeoutService};
    use crate::hyper::connect::Conn;
    use crate::hyper::error::BoxError;

    // Mock service for testing
    #[derive(Clone)]
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

        fn call(&mut self, _request: Uri) -> AsyncStream<ConnResult<Self::Response>> {
            let delay = self.delay;
            AsyncStream::with_channel(move |sender| {
                let _task = spawn_task(move || {
                    std::thread::sleep(delay);
                    emit!(sender, ConnResult::success(Conn::default()));
                });
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
        assert!(results[0].is_success());
    }

    #[test]
    fn test_timeout_layer_timeout() {
        let base_service = MockConnectorService::new(true, Duration::from_millis(100));
        let timeout_layer = AsyncStreamTimeoutLayer::new(Duration::from_millis(10));
        let mut timeout_service = timeout_layer.layer(base_service);

        let result_stream = timeout_service.call(Uri::from_static("http://example.com"));
        let results: Vec<_> = result_stream.collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_error());
    }

    #[test]
    fn test_concurrency_layer_within_limit() {
        let base_service = MockConnectorService::new(true, Duration::from_millis(10));
        let concurrency_layer = AsyncStreamConcurrencyLayer::new(2);
        let mut concurrency_service = concurrency_layer.layer(base_service);

        assert!(concurrency_service.is_ready());

        let result_stream = concurrency_service.call(Uri::from_static("http://example.com"));
        let results: Vec<_> = result_stream.collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_success());
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
        assert!(results[0].is_success());
    }

    #[test]
    fn test_conn_result_methods() {
        let success_result = ConnResult::success("test");
        assert!(success_result.is_success());
        assert!(!success_result.is_error());
        assert_eq!(success_result.into_success(), Some("test"));

        let error_result = ConnResult::<String>::error("test error");
        assert!(!error_result.is_success());
        assert!(error_result.is_error());
        assert_eq!(error_result.error(), Some("test error"));

        let timeout_result = ConnResult::<String>::timeout();
        assert!(!timeout_result.is_success());
        assert!(timeout_result.is_error());
    }
}
