//! HTTP middleware for request/response processing
//! Aligned with Reqwest patterns - simplified direct result processing

use std::sync::Arc;

use crate::{HttpError, HttpRequest, HttpResponse, HttpResult};

pub mod cache;

/// HTTP middleware trait aligned with Reqwest patterns
pub trait Middleware: Send + Sync {
    /// Process request before sending - returns result directly
    fn process_request(&self, request: HttpRequest) -> HttpResult<HttpRequest> {
        Ok(request)
    }

    /// Process response after receiving - returns result directly  
    fn process_response(&self, response: HttpResponse) -> HttpResult<HttpResponse> {
        Ok(response)
    }

    /// Handle errors - returns result directly
    fn handle_error(&self, error: HttpError) -> HttpResult<HttpError> {
        Ok(error)
    }
}

/// Middleware chain for sequential processing
/// Simplified direct result processing aligned with Reqwest patterns
#[derive(Default)]
pub struct MiddlewareChain {
    middlewares: Vec<Arc<dyn Middleware>>}

impl MiddlewareChain {
    /// Create a new middleware chain
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add middleware to the chain
    ///
    /// # Arguments
    /// * `middleware` - The middleware to add to the chain
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn add<M: Middleware + 'static>(mut self, middleware: M) -> Self {
        self.middlewares.push(Arc::new(middleware));
        self
    }

    /// Process request through all middlewares sequentially
    pub fn process_request(&self, mut request: HttpRequest) -> HttpResult<HttpRequest> {
        for middleware in &self.middlewares {
            request = middleware.process_request(request)?;
        }
        Ok(request)
    }

    /// Process response through all middlewares in reverse order
    pub fn process_response(&self, mut response: HttpResponse) -> HttpResult<HttpResponse> {
        for middleware in self.middlewares.iter().rev() {
            response = middleware.process_response(response)?;
        }
        Ok(response)
    }

    /// Handle error through all middlewares in reverse order
    pub fn handle_error(&self, mut error: HttpError) -> HttpResult<HttpError> {
        for middleware in self.middlewares.iter().rev() {
            error = middleware.handle_error(error)?;
        }
        Ok(error)
    }
}

/// Pre-built middleware implementations
/// Adds unique request IDs to requests
#[derive(Debug, Default)]
pub struct RequestIdMiddleware;

impl RequestIdMiddleware {
    /// Create a new `RequestIdMiddleware` instance
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Middleware for RequestIdMiddleware {
    fn process_request(&self, request: HttpRequest) -> HttpResult<HttpRequest> {
        let request_id = fastrand::u64(..).to_string();
        let request = request.header(
            http::HeaderName::from_static("x-request-id"),
            http::HeaderValue::from_str(&request_id).map_err(|e| HttpError::StreamError {
                message: format!("Invalid request ID: {e}")
            })?,
        );
        Ok(request)
    }
}

/// Request/response logging middleware
#[derive(Debug)]
pub struct LoggingMiddleware {
    enabled: bool}

impl LoggingMiddleware {
    /// Create a new LoggingMiddleware instance with logging enabled
    pub fn new() -> Self {
        Self { enabled: true }
    }

    /// Create a new LoggingMiddleware instance with configurable logging state
    pub fn enabled(enabled: bool) -> Self {
        Self { enabled }
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for LoggingMiddleware {
    fn process_request(&self, request: HttpRequest) -> HttpResult<HttpRequest> {
        if self.enabled {
            println!("HTTP Request: {} {}", request.method(), request.url());
        }
        Ok(request)
    }

    fn process_response(&self, response: HttpResponse) -> HttpResult<HttpResponse> {
        if self.enabled {
            println!(
                "HTTP Response: {} - {} bytes",
                response.status(),
                response.body().len()
            );
        }
        Ok(response)
    }
}

/// Compression middleware for request/response
#[derive(Debug)]
pub struct CompressionMiddleware {
    enabled: bool}

impl CompressionMiddleware {
    /// Create a new CompressionMiddleware instance with compression enabled
    pub fn new() -> Self {
        Self { enabled: true }
    }

    /// Create a new CompressionMiddleware instance with configurable compression state
    pub fn enabled(enabled: bool) -> Self {
        Self { enabled }
    }
}

impl Default for CompressionMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for CompressionMiddleware {
    fn process_request(&self, request: HttpRequest) -> HttpResult<HttpRequest> {
        if self.enabled {
            let request = request
                .header(
                    http::header::ACCEPT_ENCODING,
                    http::HeaderValue::from_static("gzip, deflate, br"),
                )
                .header(
                    http::header::CONTENT_ENCODING,
                    http::HeaderValue::from_static("identity"),
                );
            Ok(request)
        } else {
            Ok(request)
        }
    }

    fn process_response(&self, response: HttpResponse) -> HttpResult<HttpResponse> {
        // Response decompression would be handled here in full implementation
        Ok(response)
    }
}
