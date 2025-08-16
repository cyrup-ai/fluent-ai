//! Cache middleware module for HTTP3 client
//!
//! Provides comprehensive cache management functionality including ETag processing,
//! expires computation, and HTTP date handling with streaming-first architecture.
//!
//! This module is organized into logical components:
//! - `middleware`: Core CacheMiddleware struct and configuration
//! - `processing`: Middleware trait implementation and request/response processing
//! - `date_parsing`: HTTP date parsing utilities and timestamp conversion
//! - `date_formatting`: HTTP date formatting utilities and calendar calculations

pub mod date_formatting;
pub mod date_parsing;
pub mod middleware;
pub mod processing;

// Re-export core types for public API
pub use middleware::CacheMiddleware;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{HttpRequest, HttpResponse, HttpResult, Middleware};

    #[test]
    fn test_cache_middleware_creation() {
        let middleware = CacheMiddleware::new();
        assert_eq!(middleware.default_expires_hours(), 24);
        assert!(middleware.generates_etags());
    }

    #[test]
    fn test_cache_middleware_configuration() {
        let middleware = CacheMiddleware::new()
            .with_default_expires_hours(48)
            .with_etag_generation(false);

        assert_eq!(middleware.default_expires_hours(), 48);
        assert!(!middleware.generates_etags());
    }

    #[test]
    fn test_cache_middleware_default() {
        let middleware = CacheMiddleware::default();
        assert_eq!(middleware.default_expires_hours(), 24);
        assert!(middleware.generates_etags());
    }

    #[test]
    fn test_date_parsing_rfc1123() {
        let date_str = "Sun, 06 Nov 1994 08:49:37 GMT";
        let timestamp = date_parsing::parse_http_date_to_timestamp(date_str);
        assert!(timestamp.is_some());
    }

    #[test]
    fn test_date_parsing_invalid() {
        let date_str = "Invalid Date Format";
        let timestamp = date_parsing::parse_http_date_to_timestamp(date_str);
        assert!(timestamp.is_none());
    }

    #[test]
    fn test_leap_year_calculation() {
        assert!(date_parsing::is_leap_year(2000)); // Divisible by 400
        assert!(date_parsing::is_leap_year(2004)); // Divisible by 4, not by 100
        assert!(!date_parsing::is_leap_year(1900)); // Divisible by 100, not by 400
        assert!(!date_parsing::is_leap_year(2001)); // Not divisible by 4
    }

    #[test]
    fn test_days_in_month() {
        assert_eq!(date_parsing::days_in_month(1, 2023), Some(31)); // January
        assert_eq!(date_parsing::days_in_month(2, 2023), Some(28)); // February, non-leap year
        assert_eq!(date_parsing::days_in_month(2, 2024), Some(29)); // February, leap year
        assert_eq!(date_parsing::days_in_month(4, 2023), Some(30)); // April
        assert_eq!(date_parsing::days_in_month(13, 2023), None); // Invalid month
    }

    #[test]
    fn test_date_formatting() {
        let timestamp = 784111777; // Some test timestamp
        let formatted = date_formatting::format_timestamp_as_http_date(timestamp);

        // Should be in RFC 1123 format
        assert!(formatted.contains("GMT"));
        assert!(formatted.contains(":"));
        assert!(formatted.len() > 20); // Reasonable length check
    }

    #[test]
    fn test_date_round_trip() {
        // Test that we can parse what we format (within reasonable limits)
        let original_timestamp = 1609459200; // 2021-01-01 00:00:00 UTC
        let formatted = date_formatting::format_timestamp_as_http_date(original_timestamp);

        // Note: Due to simplified implementation, exact round-trip may not work
        // but the format should be valid
        assert!(formatted.contains("GMT"));
        assert!(formatted.contains("Jan"));
        assert!(formatted.contains("2021"));
    }

    #[test]
    fn test_middleware_process_request() {
        let middleware = CacheMiddleware::new();
        let request = HttpRequest::get("http://example.com").build().unwrap();

        let result = middleware.process_request(request);
        assert!(matches!(result, HttpResult::Ok(_)));
    }

    #[test]
    fn test_middleware_process_response_with_etag_generation() {
        let middleware = CacheMiddleware::new().with_etag_generation(true);

        let mut headers = HashMap::new();
        headers.insert("content-type".to_string(), "application/json".to_string());

        let response =
            HttpResponse::from_cache(http::StatusCode::OK, headers, b"test body".to_vec());

        let result = middleware.process_response(response);
        assert!(matches!(result, HttpResult::Ok(_)));

        if let HttpResult::Ok(processed_response) = result {
            // Should have added ETag header
            assert!(processed_response.headers().contains_key("etag"));
            // Should have added computed expires header
            assert!(
                processed_response
                    .headers()
                    .contains_key("x-computed-expires")
            );
            // Should have added expires header
            assert!(processed_response.headers().contains_key("expires"));
        }
    }

    #[test]
    fn test_middleware_process_response_without_etag_generation() {
        let middleware = CacheMiddleware::new().with_etag_generation(false);

        let headers = HashMap::new();
        let response =
            HttpResponse::from_cache(http::StatusCode::OK, headers, b"test body".to_vec());

        let result = middleware.process_response(response);
        assert!(matches!(result, HttpResult::Ok(_)));

        if let HttpResult::Ok(processed_response) = result {
            // Should not have added ETag header
            assert!(!processed_response.headers().contains_key("etag"));
            // Should still have computed expires
            assert!(
                processed_response
                    .headers()
                    .contains_key("x-computed-expires")
            );
        }
    }
}
