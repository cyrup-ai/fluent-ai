//! Middleware cache module tests
//!
//! Tests for middleware cache functionality, mirroring src/middleware/cache.rs

use std::time::{Duration, SystemTime};

use fluent_ai_http3::common::cache::httpdate;
use fluent_ai_http3::middleware::cache::CacheMiddleware;
use fluent_ai_http3::{HttpResponse, Middleware};
use http::StatusCode;
use http::header::HeaderMap;

#[cfg(test)]
mod middleware_cache_tests {
    use super::*;

    #[test]
    fn test_cache_middleware_creation() {
        let middleware = CacheMiddleware::new().with_default_expires_hours(12);

        // Verify middleware configuration using public getters
        assert_eq!(middleware.default_expires_hours(), 12);
        assert!(middleware.generates_etags());
    }

    #[test]
    fn test_cache_middleware_default() {
        let middleware = CacheMiddleware::default();

        // Verify default configuration using public getters
        assert_eq!(middleware.default_expires_hours(), 24); // 24 hours default
        assert!(middleware.generates_etags());
    }

    #[test]
    fn test_etag_generation() {
        let middleware = CacheMiddleware::new();

        // Create mock response for testing
        let headers = HeaderMap::new();
        let response = HttpResponse::new(StatusCode::OK, &headers, b"test content".to_vec());

        // Process through middleware to trigger ETag generation
        let processed = middleware
            .process_response(response)
            .expect("Processing should succeed");

        // Test ETag was added
        assert!(processed.etag().is_some());
        let etag = processed.etag().expect("ETag should be present");
        assert!(etag.starts_with("W/\""));
        assert!(etag.ends_with("\""));
        assert!(etag.len() > 4); // Should have content beyond W/""

        // Same content should generate same ETag (process again)
        let headers2 = HeaderMap::new();
        let response2 = HttpResponse::new(StatusCode::OK, &headers2, b"test content".to_vec());
        let processed2 = middleware
            .process_response(response2)
            .expect("Processing should succeed");
        assert_eq!(processed.etag(), processed2.etag());
    }

    #[test]
    fn test_expires_computation() {
        let middleware = CacheMiddleware::new().with_default_expires_hours(1);

        let headers = HeaderMap::new();
        let response = HttpResponse::new(StatusCode::OK, &headers, b"test".to_vec());

        // Process through middleware to compute expires
        let processed = middleware
            .process_response(response)
            .expect("Processing should succeed");

        // Check computed expires
        let computed_expires = processed
            .computed_expires()
            .expect("Computed expires should be present");
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();

        // Should be approximately 1 hour in the future
        assert!(computed_expires > now);
        assert!(computed_expires <= now + 3600 + 10); // Allow 10 second tolerance
    }

    #[test]
    fn test_http_date_parsing() {
        // Test RFC 7231 IMF-fixdate format
        let date1 = "Sun, 06 Nov 1994 08:49:37 GMT";
        let result1 = httpdate::parse_http_date(date1);
        assert!(result1.is_ok());

        // Test RFC 850 format
        let date2 = "Sunday, 06-Nov-94 08:49:37 GMT";
        let result2 = httpdate::parse_http_date(date2);
        assert!(result2.is_ok());

        // Test ANSI C asctime format
        let date3 = "Sun Nov  6 08:49:37 1994";
        let result3 = httpdate::parse_http_date(date3);
        assert!(result3.is_ok());

        // Test invalid format
        let invalid_date = "not a date";
        let result4 = httpdate::parse_http_date(invalid_date);
        assert!(result4.is_err());
    }

    #[test]
    fn test_http_date_formatting() {
        let time = SystemTime::UNIX_EPOCH + Duration::from_secs(784111777);
        let formatted = httpdate::fmt_http_date(time);

        // Should be in RFC 7231 format
        assert!(formatted.contains("GMT"));
        assert!(formatted.contains(","));

        // Should be parseable back
        let parsed = httpdate::parse_http_date(&formatted);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_cache_response_processing() {
        let middleware = CacheMiddleware::new();

        // Create response without ETag or Expires
        let headers = HeaderMap::new();
        let response = HttpResponse::new(StatusCode::OK, &headers, b"test content".to_vec());
        assert!(response.etag().is_none());
        assert!(response.expires().is_none());

        // Process through middleware
        let processed = middleware
            .process_response(response)
            .expect("Response processing should succeed");

        // Should now have ETag
        assert!(processed.etag().is_some());

        // Should have computed expires header
        assert!(processed.headers().contains_key("x-computed-expires"));
        assert!(processed.expires().is_some());
    }
}
