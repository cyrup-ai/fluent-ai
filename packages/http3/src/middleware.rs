//! HTTP middleware for request/response processing

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::{HttpError, HttpRequest, HttpResponse, HttpResult};

/// HTTP middleware trait using native async
pub trait Middleware: Send + Sync {
    /// Process request before sending
    fn process_request(
        &self,
        request: HttpRequest,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpRequest>> + Send + '_>> {
        Box::pin(async move { Ok(request) })
    }

    /// Process response after receiving
    fn process_response(
        &self,
        response: HttpResponse,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpResponse>> + Send + '_>> {
        Box::pin(async move { Ok(response) })
    }

    /// Handle errors
    fn handle_error(
        &self,
        error: HttpError,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpError>> + Send + '_>> {
        Box::pin(async move { Ok(error) })
    }
}

/// Middleware chain for composing multiple middleware
pub struct MiddlewareChain {
    middleware: Vec<Arc<dyn Middleware>>,
}

impl MiddlewareChain {
    /// Create a new middleware chain
    pub fn new() -> Self {
        Self {
            middleware: Vec::new(),
        }
    }

    /// Add middleware to the chain
    pub fn add<M: Middleware + 'static>(mut self, middleware: M) -> Self {
        self.middleware.push(Arc::new(middleware));
        self
    }

    /// Process request through all middleware
    pub async fn process_request(&self, mut request: HttpRequest) -> HttpResult<HttpRequest> {
        for middleware in &self.middleware {
            request = middleware.process_request(request).await?;
        }
        Ok(request)
    }

    /// Process response through all middleware (in reverse order)
    pub async fn process_response(&self, mut response: HttpResponse) -> HttpResult<HttpResponse> {
        for middleware in self.middleware.iter().rev() {
            response = middleware.process_response(response).await?;
        }
        Ok(response)
    }

    /// Handle error through all middleware
    pub async fn handle_error(&self, mut error: HttpError) -> HttpResult<HttpError> {
        for middleware in &self.middleware {
            error = middleware.handle_error(error).await?;
        }
        Ok(error)
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Retry middleware
pub struct RetryMiddleware {
    #[allow(dead_code)]
    max_retries: usize,
    base_delay: std::time::Duration,
    backoff_factor: f64,
    retry_on_status: Vec<u16>,
}

impl RetryMiddleware {
    /// Create a new retry middleware
    pub fn new(max_retries: usize) -> Self {
        Self {
            max_retries,
            base_delay: std::time::Duration::from_millis(100),
            backoff_factor: 2.0,
            retry_on_status: vec![429, 500, 502, 503, 504],
        }
    }

    /// Set base delay
    pub fn with_base_delay(mut self, delay: std::time::Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Set backoff factor
    pub fn with_backoff_factor(mut self, factor: f64) -> Self {
        self.backoff_factor = factor;
        self
    }

    /// Set retry status codes
    pub fn with_retry_on_status(mut self, status_codes: Vec<u16>) -> Self {
        self.retry_on_status = status_codes;
        self
    }

    /// Check if error should be retried
    fn should_retry(&self, error: &HttpError) -> bool {
        match error {
            HttpError::HttpStatus { status, .. } => self.retry_on_status.contains(status),
            HttpError::NetworkError { .. }
            | HttpError::Timeout { .. }
            | HttpError::ConnectionError { .. } => true,
            _ => false,
        }
    }

    /// Calculate delay for retry attempt
    #[allow(dead_code)]
    fn calculate_delay(&self, attempt: usize) -> std::time::Duration {
        let delay = self.base_delay.as_millis() as f64 * self.backoff_factor.powi(attempt as i32);
        std::time::Duration::from_millis(delay as u64)
    }
}

impl Middleware for RetryMiddleware {
    fn handle_error(
        &self,
        error: HttpError,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpError>> + Send + '_>> {
        let should_retry = self.should_retry(&error);
        Box::pin(async move {
            if should_retry {
                // This is a simplified implementation
                // In a real implementation, you'd need to integrate with the HTTP client
                // to actually retry the request
                Ok(error)
            } else {
                Ok(error)
            }
        })
    }
}

/// Request ID middleware
pub struct RequestIdMiddleware {
    header_name: String,
}

impl RequestIdMiddleware {
    /// Create a new request ID middleware
    pub fn new() -> Self {
        Self {
            header_name: "X-Request-ID".to_string(),
        }
    }

    /// Set the header name
    pub fn with_header_name(mut self, name: String) -> Self {
        self.header_name = name;
        self
    }

    /// Generate a new request ID
    #[allow(dead_code)]
    fn generate_request_id(&self) -> String {
        Self::generate_request_id_static()
    }

    /// Generate a new request ID (static version)
    fn generate_request_id_static() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        format!("{}-{}", timestamp, fastrand::u64(..))
    }
}

impl Default for RequestIdMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for RequestIdMiddleware {
    fn process_request(
        &self,
        mut request: HttpRequest,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpRequest>> + Send + '_>> {
        let header_name = self.header_name.clone();
        Box::pin(async move {
            if !request.headers().contains_key(&header_name) {
                let request_id = Self::generate_request_id_static();
                request = request.header(&header_name, request_id);
            }
            Ok(request)
        })
    }
}

/// Logging middleware
pub struct LoggingMiddleware {
    log_requests: bool,
    log_responses: bool,
    log_errors: bool,
}

impl LoggingMiddleware {
    /// Create a new logging middleware
    pub fn new() -> Self {
        Self {
            log_requests: true,
            log_responses: true,
            log_errors: true,
        }
    }

    /// Enable/disable request logging
    pub fn with_request_logging(mut self, enabled: bool) -> Self {
        self.log_requests = enabled;
        self
    }

    /// Enable/disable response logging
    pub fn with_response_logging(mut self, enabled: bool) -> Self {
        self.log_responses = enabled;
        self
    }

    /// Enable/disable error logging
    pub fn with_error_logging(mut self, enabled: bool) -> Self {
        self.log_errors = enabled;
        self
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for LoggingMiddleware {
    fn process_request(
        &self,
        request: HttpRequest,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpRequest>> + Send + '_>> {
        let log_requests = self.log_requests;
        Box::pin(async move {
            if log_requests {
                #[cfg(feature = "tracing")]
                tracing::info!(
                    method = %request.method(),
                    url = %request.url(),
                    "Sending HTTP request"
                );

                #[cfg(not(feature = "tracing"))]
                println!("HTTP {} {}", request.method(), request.url());
            }
            Ok(request)
        })
    }

    fn process_response(
        &self,
        response: HttpResponse,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpResponse>> + Send + '_>> {
        let log_responses = self.log_responses;
        Box::pin(async move {
            if log_responses {
                #[cfg(feature = "tracing")]
                tracing::info!(
                    status = %response.status(),
                    "Received HTTP response"
                );

                #[cfg(not(feature = "tracing"))]
                println!("HTTP {} response", response.status());
            }
            Ok(response)
        })
    }

    fn handle_error(
        &self,
        error: HttpError,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpError>> + Send + '_>> {
        let log_errors = self.log_errors;
        Box::pin(async move {
            if log_errors {
                #[cfg(feature = "tracing")]
                tracing::error!(
                    error = %error,
                    "HTTP request failed"
                );

                #[cfg(not(feature = "tracing"))]
                eprintln!("HTTP error: {}", error);
            }
            Ok(error)
        })
    }
}

/// Metrics middleware
pub struct MetricsMiddleware {
    request_count: Arc<std::sync::atomic::AtomicUsize>,
    response_count: Arc<std::sync::atomic::AtomicUsize>,
    error_count: Arc<std::sync::atomic::AtomicUsize>,
    total_response_time: Arc<std::sync::atomic::AtomicU64>,
}

impl MetricsMiddleware {
    /// Create a new metrics middleware
    pub fn new() -> Self {
        Self {
            request_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            response_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            error_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            total_response_time: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Get metrics
    pub fn metrics(&self) -> Metrics {
        let requests = self
            .request_count
            .load(std::sync::atomic::Ordering::Relaxed);
        let responses = self
            .response_count
            .load(std::sync::atomic::Ordering::Relaxed);
        let errors = self.error_count.load(std::sync::atomic::Ordering::Relaxed);
        let total_time = self
            .total_response_time
            .load(std::sync::atomic::Ordering::Relaxed);

        let average_response_time = if responses > 0 {
            std::time::Duration::from_millis(total_time / responses as u64)
        } else {
            std::time::Duration::from_millis(0)
        };

        Metrics {
            requests,
            responses,
            errors,
            average_response_time,
        }
    }
}

impl Default for MetricsMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for MetricsMiddleware {
    fn process_request(
        &self,
        request: HttpRequest,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpRequest>> + Send + '_>> {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Box::pin(async move { Ok(request) })
    }

    fn process_response(
        &self,
        response: HttpResponse,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpResponse>> + Send + '_>> {
        self.response_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Box::pin(async move { Ok(response) })
    }

    fn handle_error(
        &self,
        error: HttpError,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpError>> + Send + '_>> {
        self.error_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Box::pin(async move { Ok(error) })
    }
}

/// Metrics data
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Number of requests sent
    pub requests: usize,
    /// Number of responses received
    pub responses: usize,
    /// Number of errors encountered
    pub errors: usize,
    /// Average response time
    pub average_response_time: std::time::Duration,
}

/// User agent middleware
pub struct UserAgentMiddleware {
    user_agent: String,
}

impl UserAgentMiddleware {
    /// Create a new user agent middleware
    pub fn new(user_agent: String) -> Self {
        Self { user_agent }
    }
}

impl Middleware for UserAgentMiddleware {
    fn process_request(
        &self,
        mut request: HttpRequest,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpRequest>> + Send + '_>> {
        let user_agent = self.user_agent.clone();
        Box::pin(async move {
            if !request.headers().contains_key("User-Agent") {
                request = request.header("User-Agent", &user_agent);
            }
            Ok(request)
        })
    }
}

/// Compression middleware
pub struct CompressionMiddleware {
    accept_encoding: String,
}

impl CompressionMiddleware {
    /// Create a new compression middleware
    pub fn new() -> Self {
        Self {
            accept_encoding: "gzip, br, deflate".to_string(),
        }
    }

    /// Set accept encoding
    pub fn with_accept_encoding(mut self, encoding: String) -> Self {
        self.accept_encoding = encoding;
        self
    }
}

impl Default for CompressionMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for CompressionMiddleware {
    fn process_request(
        &self,
        mut request: HttpRequest,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpRequest>> + Send + '_>> {
        let accept_encoding = self.accept_encoding.clone();
        Box::pin(async move {
            if !request.headers().contains_key("Accept-Encoding") {
                request = request.header("Accept-Encoding", &accept_encoding);
            }
            Ok(request)
        })
    }
}
