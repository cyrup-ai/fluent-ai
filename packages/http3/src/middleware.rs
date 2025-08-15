//! HTTP middleware for request/response processing
//! Simplified, streaming-first processing aligned with `fluent_ai_http3`'s zero-allocation design

use std::sync::Arc;

use crate::{HttpError, HttpRequest, HttpResponse, HttpResult};

pub mod cache;

/// HTTP middleware trait for fluent_ai_http3
pub trait Middleware: Send + Sync {
    /// Process request before sending - returns HttpResult directly
    fn process_request(&self, request: HttpRequest) -> HttpResult<HttpRequest> {
        HttpResult::Ok(request)
    }

    /// Process response after receiving - returns HttpResult directly  
    fn process_response(&self, response: HttpResponse) -> HttpResult<HttpResponse> {
        HttpResult::Ok(response)
    }

    /// Handle errors - returns HttpResult directly
    fn handle_error(&self, error: HttpError) -> HttpResult<HttpError> {
        HttpResult::Ok(error)
    }
}

/// Middleware chain for sequential processing
/// Simplified direct result processing aligned with fluent_ai_http3 patterns
#[derive(Default)]
pub struct MiddlewareChain {
    middlewares: Vec<Arc<dyn Middleware>>,
}

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
    pub fn process_request(&self, mut request: HttpRequest) -> Result<HttpRequest, HttpError> {
        for middleware in &self.middlewares {
            match middleware.process_request(request) {
                HttpResult::Ok(req) => request = req,
                HttpResult::Err(err) => return Err(err),
            }
        }
        Ok(request)
    }

    /// Process response through all middlewares in reverse order
    pub fn process_response(&self, mut response: HttpResponse) -> Result<HttpResponse, HttpError> {
        for middleware in self.middlewares.iter().rev() {
            match middleware.process_response(response) {
                HttpResult::Ok(resp) => response = resp,
                HttpResult::Err(err) => return Err(err),
            }
        }
        Ok(response)
    }

    /// Handle error through all middlewares in reverse order
    pub fn handle_error(&self, mut error: HttpError) -> Result<HttpError, HttpError> {
        for middleware in self.middlewares.iter().rev() {
            match middleware.handle_error(error) {
                HttpResult::Ok(err) => error = err,
                HttpResult::Err(err) => return Err(err),
            }
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
        match http::HeaderValue::from_str(&request_id) {
            Ok(header_value) => {
                let request =
                    request.header(http::HeaderName::from_static("x-request-id"), header_value);
                HttpResult::Ok(request)
            }
            Err(e) => HttpResult::Err(HttpError::StreamError {
                message: format!("Invalid request ID: {e}"),
            }),
        }
    }
}

/// Request/response logging middleware
#[derive(Debug)]
pub struct LoggingMiddleware {
    enabled: bool,
}

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
        HttpResult::Ok(request)
    }

    fn process_response(&self, response: HttpResponse) -> HttpResult<HttpResponse> {
        if self.enabled {
            println!(
                "HTTP Response: {} - {} bytes",
                response.status(),
                response.body().len()
            );
        }
        HttpResult::Ok(response)
    }
}

/// Compression middleware for request/response
#[derive(Debug)]
pub struct CompressionMiddleware {
    enabled: bool,
}

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
            HttpResult::Ok(request)
        } else {
            HttpResult::Ok(request)
        }
    }

    fn process_response(&self, response: HttpResponse) -> HttpResult<HttpResponse> {
        // Response decompression would be handled here in full implementation
        HttpResult::Ok(response)
    }
}
