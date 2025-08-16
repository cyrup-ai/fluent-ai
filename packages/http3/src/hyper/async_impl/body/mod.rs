//! HTTP Body implementation with streams-first architecture
//!
//! This module provides a complete HTTP body implementation using AsyncStream patterns
//! for zero-allocation streaming. The module is decomposed into logical components
//! for better maintainability and separation of concerns.
//!
//! # Architecture
//!
//! - `bridge`: AsyncStream to HttpBody bridging functionality
//! - `constructors`: Body creation and factory methods  
//! - `conversions`: Type conversions and From trait implementations
//! - `timeout`: Timeout handling for body operations
//! - `streaming`: Stream processing and frame handling
//!
//! # Usage
//!
//! ```rust
//! use crate::hyper::body::Body;
//!
//! // Create body from bytes
//! let body = Body::from("Hello, world!");
//!
//! // Create body from stream
//! let stream_body = Body::wrap_stream(stream);
//! ```

mod bridge;
mod constructors;
mod conversions;
mod streaming;
mod timeout;

// Re-export the main Body type and its key functionality
// Re-export internal types needed by other modules
pub(super) use bridge::AsyncStreamHttpBody;
pub use constructors::Body;
pub(super) use constructors::Inner;
pub(super) use conversions::IntoBytesBody;
#[cfg(feature = "multipart")]
pub use streaming::DataStream;
pub use timeout::{ResponseBody, boxed, response, total_timeout, with_read_timeout};

#[cfg(test)]
mod integration_tests {
    use http_body::Body as HttpBodyTrait;

    use super::*;

    #[test]
    fn test_body_creation() {
        let body = Body::from("test data");
        assert!(!body.is_end_stream());
        assert_eq!(body.size_hint().exact(), Some(9));
    }

    #[test]
    fn test_empty_body() {
        let body = Body::empty();
        assert!(body.is_end_stream());
        assert_eq!(body.size_hint().exact(), Some(0));
    }

    #[test]
    fn test_body_clone() {
        let body = Body::from("clone test");
        let cloned = body.try_clone();
        assert!(cloned.is_some());
    }

    #[test]
    fn test_body_reuse() {
        let body = Body::from("reuse test");
        let (reuse_bytes, _body) = body.try_reuse();
        assert!(reuse_bytes.is_some());
        assert_eq!(reuse_bytes.unwrap(), "reuse test");
    }
}
