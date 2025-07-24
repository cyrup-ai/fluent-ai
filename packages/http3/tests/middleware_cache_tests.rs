//! Integration tests for HTTP3 cache middleware
//!
//! These tests verify the cache middleware functionality including:
//! - ETag generation and processing
//! - Expires header computation
//! - Cache directive parsing
//! - HTTP date parsing utilities

use http3::middleware::cache::CacheMiddleware;
use http3::{HttpRequest, HttpResponse, Middleware};

#[test]
fn test_cache_middleware_creation() {
    let middleware = CacheMiddleware::new().with_default_expires_hours(12);

    // Verify middleware configuration
    assert_eq!(middleware.default_expires_hours(), 12);
    assert!(middleware.generates_etags());
}

#[test]
fn test_cache_middleware_default() {
    let middleware = CacheMiddleware::default();

    // Verify default configuration
    assert_eq!(middleware.default_expires_hours(), 24); // 24 hours default
    assert!(middleware.generates_etags());
}

#[test]
fn test_etag_generation() {
    let middleware = CacheMiddleware::new();

    // Create mock response for testing
    let response = HttpResponse::new(200, b"test content".to_vec());

    // Test ETag generation
    let etag = middleware.generate_etag(&response);
    assert!(etag.starts_with("W/\""));
    assert!(etag.ends_with("\""));
    assert!(etag.len() > 4); // Should have content beyond W/""

    // Same content should generate same ETag
    let etag2 = middleware.generate_etag(&response);
    assert_eq!(etag, etag2);
}

#[test]
fn test_expires_computation() {
    let middleware = CacheMiddleware::new().with_default_expires_hours(1);

    let response = HttpResponse::new(200, b"test".to_vec());

    // Test with no user-specified expires
    let computed_expires = middleware.compute_expires(&response, None);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();

    // Should be approximately 1 hour in the future
    assert!(computed_expires > now);
    assert!(computed_expires <= now + 3600 + 10); // Allow 10 second tolerance
}

#[test]
fn test_cache_directive_parsing() {
    let middleware = CacheMiddleware::new();

    // Test max-age parsing
    assert_eq!(
        middleware.parse_max_age_directive("max-age=3600"),
        Some(3600)
    );
    assert_eq!(
        middleware.parse_max_age_directive("public, max-age=7200, must-revalidate"),
        Some(7200)
    );
    assert_eq!(
        middleware.parse_max_age_directive("no-cache, no-store"),
        None
    );
    assert_eq!(middleware.parse_max_age_directive("max-age=invalid"), None);
}

#[test]
fn test_request_cache_directives_extraction() {
    let middleware = CacheMiddleware::new();

    // Test Cache-Control header extraction
    let mut request = HttpRequest::get("http://example.com/test");
    request = request.header("cache-control", "max-age=1800"); // 30 minutes

    let extracted_hours = middleware.extract_request_cache_directives(&request);
    assert_eq!(extracted_hours, Some(0)); // 1800 seconds = 0.5 hours, rounded down

    // Test custom cache expires header
    let mut request2 = HttpRequest::get("http://example.com/test");
    request2 = request2.header("x-cache-expires-hours", "6");

    let extracted_hours2 = middleware.extract_request_cache_directives(&request2);
    assert_eq!(extracted_hours2, Some(6));
}

#[test]
fn test_http_date_parsing() {
    use http3::common::cache::httpdate;

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
    use http3::common::cache::httpdate;

    let time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(784111777);
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

    // Create response without ETag
    let response = HttpResponse::new(200, b"test content".to_vec());
    assert!(response.etag().is_none());

    // Process through middleware
    let processed = middleware.process_response(response);
    assert!(processed.is_ok());

    let processed_response = processed.expect("Response processing should succeed");

    // Should now have ETag
    assert!(processed_response.etag().is_some());

    // Should have computed expires
    assert!(
        processed_response
            .headers()
            .contains_key("x-computed-expires")
    );
}
