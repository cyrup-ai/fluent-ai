//! Core HTTP protocol traits
//!
//! Provides trait abstractions for HTTP/2, HTTP/3, and QUIC protocols
//! using fluent_ai_async patterns with builder-based interfaces.

use bytes::Bytes;
use fluent_ai_async::prelude::*;
use std::time::Duration;

use crate::protocols::types::{HttpMethod, HttpVersion, TimeoutConfig};

/// Core HTTP protocol abstraction
pub trait HttpProtocol {
    type RequestBuilder: HttpRequestBuilder;
    type ConnectionState: ConnectionState;
    
    /// Create a new request builder
    fn request_builder(&self) -> Self::RequestBuilder;
    
    /// Get current connection state
    fn connection_state(&self) -> Self::ConnectionState;
    
    /// Protocol version
    fn version(&self) -> HttpVersion;
    
    /// Configuration
    fn config(&self) -> &TimeoutConfig;
}

/// Request builder trait (avoids async fn in traits)
pub trait HttpRequestBuilder {
    type ResponseChunk: MessageChunk;
    
    /// Set HTTP method
    fn method(self, method: HttpMethod) -> Self;
    
    /// Set request URI
    fn uri(self, uri: &str) -> Self;
    
    /// Add header
    fn header(self, name: &str, value: &str) -> Self;
    
    /// Set request body
    fn body(self, body: Bytes) -> Self;
    
    /// Execute request and return streaming response
    fn execute(self) -> AsyncStream<Self::ResponseChunk, 1024>;
}

/// Connection state abstraction
pub trait ConnectionState {
    /// Check if connection is ready for requests
    fn is_ready(&self) -> bool;
    
    /// Check if connection is closed
    fn is_closed(&self) -> bool;
    
    /// Get error message if connection failed
    fn error_message(&self) -> Option<&str>;
    
    /// Get connection uptime
    fn uptime(&self) -> Option<Duration>;
}

/// Protocol-specific configuration trait
pub trait ProtocolConfig: Clone + Send + Sync {
    /// Validate configuration parameters
    fn validate(&self) -> Result<(), String>;
    
    /// Get timeout settings
    fn timeout_config(&self) -> TimeoutConfig;
}
