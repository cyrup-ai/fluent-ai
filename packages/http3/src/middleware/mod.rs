//! HTTP middleware for request/response processing
//! Simplified, streaming-first processing aligned with `fluent_ai_http3`'s zero-allocation design

use std::sync::Arc;

use crate::{HttpError, HttpRequest, HttpResponse, HttpResult};

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
#[derive(Default)]
pub struct MiddlewareChain {
    middlewares: Vec<Arc<dyn Middleware>>,
}

impl MiddlewareChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add<M: Middleware + 'static>(mut self, middleware: M) -> Self {
        self.middlewares.push(Arc::new(middleware));
        self
    }
}

/// Cache middleware module
pub mod cache;
pub use cache::CacheMiddleware;
