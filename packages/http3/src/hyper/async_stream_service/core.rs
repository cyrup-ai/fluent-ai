//! Core types and traits for AsyncStream services
//!
//! Provides AsyncStream-native service and layer traits as 100% compatible
//! replacements for tower::Service and tower::Layer using pure AsyncStream patterns.

use fluent_ai_async::AsyncStream;
use fluent_ai_async::prelude::MessageChunk;

/// Connection result type for AsyncStream compatibility
///
/// Represents connection outcomes in error-as-data pattern following fluent_ai_async architecture.
/// Implements MessageChunk for zero-allocation streaming.
#[derive(Debug, Clone)]
pub enum ConnResult<T> {
    /// Successful connection
    Success(T),
    /// Connection error with message
    Error(String),
    /// Connection timeout
    Timeout,
}

impl<T> Default for ConnResult<T> {
    fn default() -> Self {
        ConnResult::Error("default connection result".to_string())
    }
}

impl<T> MessageChunk for ConnResult<T> {
    fn is_error(&self) -> bool {
        matches!(self, ConnResult::Error(_) | ConnResult::Timeout)
    }

    fn bad_chunk(error_message: String) -> Self {
        ConnResult::Error(error_message)
    }

    fn error(&self) -> Option<&str> {
        match self {
            ConnResult::Error(msg) => Some(msg),
            _ => None,
        }
    }
}

impl<T> ConnResult<T> {
    /// Create a successful connection result
    pub fn success(value: T) -> Self {
        ConnResult::Success(value)
    }

    /// Create an error connection result
    pub fn error(message: impl Into<String>) -> Self {
        ConnResult::Error(message.into())
    }

    /// Create a timeout connection result
    pub fn timeout() -> Self {
        ConnResult::Timeout
    }

    /// Check if the result is successful
    pub fn is_success(&self) -> bool {
        matches!(self, ConnResult::Success(_))
    }

    /// Extract the success value if present
    pub fn into_success(self) -> Option<T> {
        match self {
            ConnResult::Success(value) => Some(value),
            _ => None,
        }
    }
}

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
    /// This replaces tower::Service::call() but returns AsyncStream<ConnResult<Response>>
    /// following the fluent_ai_async error-as-data pattern.
    fn call(&mut self, request: Request) -> AsyncStream<ConnResult<Self::Response>>;

    /// Get error information from the service
    ///
    /// Returns None if the service is not in an error state.
    fn error(&self) -> Option<&str> {
        None
    }
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
