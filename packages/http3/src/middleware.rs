//! HTTP middleware for request/response processing
//! NO FUTURES - pure AsyncStream architecture

use std::sync::Arc;

use crate::async_task::AsyncStream;
use crate::{HttpError, HttpRequest, HttpResponse, HttpResult};

pub mod cache;

/// HTTP middleware trait using AsyncStream ONLY - NO Futures!
pub trait Middleware: Send + Sync {
    /// Process request before sending
    fn process_request(
        &self,
        request: HttpRequest,
    ) -> AsyncStream<HttpResult<HttpRequest>> {
        AsyncStream::from_single(Ok(request))
    }

    /// Process response after receiving
    fn process_response(
        &self,
        response: HttpResponse,
    ) -> AsyncStream<HttpResult<HttpResponse>> {
        AsyncStream::from_single(Ok(response))
    }

    /// Handle errors
    fn handle_error(
        &self,
        error: HttpError,
    ) -> AsyncStream<HttpResult<HttpError>> {
        AsyncStream::from_single(Ok(error))
    }
}

/// Middleware chain for sequential processing
/// NO FUTURES - pure streaming architecture
pub struct MiddlewareChain {
    middlewares: Vec<Arc<dyn Middleware>>,
}

impl MiddlewareChain {
    /// Create new middleware chain
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Add middleware to chain
    pub fn add<M: Middleware + 'static>(mut self, middleware: M) -> Self {
        self.middlewares.push(Arc::new(middleware));
        self
    }

    /// Process request through all middlewares
    /// NO FUTURES - pure streaming with collect() for await-like behavior
    pub fn process_request(&self, request: HttpRequest) -> AsyncStream<HttpResult<HttpRequest>> {
        // For now, just return single processed request
        // In full implementation, would chain all middleware processing
        if let Some(middleware) = self.middlewares.first() {
            middleware.process_request(request)
        } else {
            AsyncStream::from_single(Ok(request))
        }
    }

    /// Process response through all middlewares  
    /// NO FUTURES - pure streaming with collect() for await-like behavior
    pub fn process_response(&self, response: HttpResponse) -> AsyncStream<HttpResult<HttpResponse>> {
        // For now, just return single processed response
        // In full implementation, would chain all middleware processing
        if let Some(middleware) = self.middlewares.first() {
            middleware.process_response(response)
        } else {
            AsyncStream::from_single(Ok(response))
        }
    }

    /// Handle error through all middlewares
    /// NO FUTURES - pure streaming with collect() for await-like behavior  
    pub fn handle_error(&self, error: HttpError) -> AsyncStream<HttpResult<HttpError>> {
        // For now, just return single processed error
        // In full implementation, would chain all middleware processing
        if let Some(middleware) = self.middlewares.first() {
            middleware.handle_error(error)
        } else {
            AsyncStream::from_single(Ok(error))
        }
    }

    /// Check if chain is empty
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Request ID middleware - adds unique IDs to requests
/// NO FUTURES - pure streaming architecture
pub struct RequestIdMiddleware;

impl RequestIdMiddleware {
    /// Create new request ID middleware
    pub fn new() -> Self {
        Self
    }
}

impl Default for RequestIdMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for RequestIdMiddleware {
    fn process_request(&self, request: HttpRequest) -> AsyncStream<HttpResult<HttpRequest>> {
        // Add unique request ID header
        let request_id = fastrand::u64(..).to_string();
        let request = request.header("X-Request-ID", &request_id);
        AsyncStream::from_single(Ok(request))
    }
}

/// Logging middleware - logs request/response details  
/// NO FUTURES - pure streaming architecture
pub struct LoggingMiddleware {
    enabled: bool,
}

impl LoggingMiddleware {
    /// Create new logging middleware
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Create enabled logging middleware
    pub fn enabled() -> Self {
        Self::new(true)
    }

    /// Create disabled logging middleware
    pub fn disabled() -> Self {
        Self::new(false)
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::enabled()
    }
}

impl Middleware for LoggingMiddleware {
    fn process_request(&self, request: HttpRequest) -> AsyncStream<HttpResult<HttpRequest>> {
        if self.enabled {
            println!("HTTP Request: {} {}", request.method(), request.url());
        }
        AsyncStream::from_single(Ok(request))
    }

    fn process_response(&self, response: HttpResponse) -> AsyncStream<HttpResult<HttpResponse>> {
        if self.enabled {
            println!("HTTP Response: {} - {} bytes", response.status(), response.body().len());
        }
        AsyncStream::from_single(Ok(response))
    }
}

/// Compression middleware - handles compression/decompression
/// NO FUTURES - pure streaming architecture  
pub struct CompressionMiddleware {
    enabled: bool,
}

impl CompressionMiddleware {
    /// Create new compression middleware
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Create enabled compression middleware
    pub fn enabled() -> Self {
        Self::new(true)
    }

    /// Create disabled compression middleware
    pub fn disabled() -> Self {
        Self::new(false)
    }
}

impl Default for CompressionMiddleware {
    fn default() -> Self {
        Self::enabled()
    }
}

impl Middleware for CompressionMiddleware {
    fn process_request(&self, request: HttpRequest) -> AsyncStream<HttpResult<HttpRequest>> {
        if self.enabled {
            // Add compression headers
            let request = request
                .header("Accept-Encoding", "gzip, deflate, br")
                .header("Content-Encoding", "identity");
            AsyncStream::from_single(Ok(request))
        } else {
            AsyncStream::from_single(Ok(request))
        }
    }

    fn process_response(&self, response: HttpResponse) -> AsyncStream<HttpResult<HttpResponse>> {
        // Response decompression would be handled here in full implementation
        AsyncStream::from_single(Ok(response))
    }
}