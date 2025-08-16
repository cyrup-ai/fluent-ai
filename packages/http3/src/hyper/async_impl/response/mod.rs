//! HTTP Response implementation with streams-first architecture
//!
//! This module provides a complete HTTP response implementation using AsyncStream patterns
//! for processing response bodies with text, JSON, and binary data support.
//!
//! # Architecture
//!
//! The response implementation is decomposed into logical components:
//!
//! - `types`: Core data structures and type definitions
//! - `core`: Constructor and basic accessor methods  
//! - `body`: Body processing methods (text, JSON, bytes streaming)
//! - `utils`: Utility methods and error handling
//! - `conversions`: Type conversions and trait implementations
//!
//! # Usage
//!
//! ```rust
//! use crate::hyper::Response;
//!
//! // Access response metadata
//! let status = response.status();
//! let headers = response.headers();
//! let url = response.url();
//!
//! // Process response body
//! let text_stream = response.text();
//! let json_stream = response.json::<MyType>();
//! let bytes_stream = response.bytes_stream();
//! ```

mod body;
mod conversions;
mod core;
mod types;
mod utils;

// Re-export the main Response type and helper types
pub use types::{Response, StringChunk};

#[cfg(test)]
mod tests {
    use http::response::Builder;
    use url::Url;

    use super::*;
    use crate::hyper::ResponseBuilderExt;

    #[test]
    fn test_from_http_response() {
        let url = Url::parse("https://localhost").expect("test URL should parse");
        let response = Builder::new()
            .status(200)
            .url(url.clone())
            .body("foo")
            .expect("test response build should succeed");
        let response = Response::from(response);

        assert_eq!(response.status(), 200);
        assert_eq!(*response.url(), url);
    }

    #[test]
    fn test_response_status() {
        let url = Url::parse("https://example.com").expect("test URL should parse");
        let response = Builder::new()
            .status(404)
            .url(url)
            .body("")
            .expect("test response build should succeed");
        let response = Response::from(response);

        assert_eq!(response.status().as_u16(), 404);
        assert!(response.status().is_client_error());
    }

    #[test]
    fn test_response_headers() {
        let url = Url::parse("https://example.com").expect("test URL should parse");
        let response = Builder::new()
            .status(200)
            .header("content-type", "application/json")
            .url(url)
            .body("")
            .expect("test response build should succeed");
        let response = Response::from(response);

        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "application/json"
        );
    }

    #[test]
    fn test_error_for_status() {
        let url = Url::parse("https://example.com").expect("test URL should parse");
        let response = Builder::new()
            .status(400)
            .url(url)
            .body("")
            .expect("test response build should succeed");
        let response = Response::from(response);

        assert!(response.error_for_status().is_err());
    }

    #[test]
    fn test_success_for_status() {
        let url = Url::parse("https://example.com").expect("test URL should parse");
        let response = Builder::new()
            .status(200)
            .url(url)
            .body("")
            .expect("test response build should succeed");
        let response = Response::from(response);

        assert!(response.error_for_status().is_ok());
    }
}
