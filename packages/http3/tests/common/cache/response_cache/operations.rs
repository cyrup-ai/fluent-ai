//! Response cache operations tests

use fluent_ai_http3::common::cache::response_cache::{ResponseCache, HttpResponse};
use http::{Response, StatusCode};

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_response() -> HttpResponse {
        Response::builder()
            .status(StatusCode::OK)
            .header("etag", "\"test-etag\"")
            .body(())
            .unwrap()
    }

    #[test]
    fn test_should_cache_with_etag() {
        let cache = ResponseCache::default();
        let response = create_test_response();

        assert!(cache.should_cache(&response));
    }

    #[test]
    fn test_should_not_cache_error() {
        let cache = ResponseCache::default();
        let response = Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(())
            .unwrap();

        assert!(!cache.should_cache(&response));
    }

    #[test]
    fn test_cache_key_generation() {
        let cache = ResponseCache::default();
        let url = "https://example.com/api/data";
        let method = "GET";

        let key1 = cache.generate_cache_key(url, method);
        let key2 = cache.generate_cache_key(url, method);
        
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_uniqueness() {
        let cache = ResponseCache::default();
        
        let key1 = cache.generate_cache_key("https://example.com/api/data", "GET");
        let key2 = cache.generate_cache_key("https://example.com/api/other", "GET");
        let key3 = cache.generate_cache_key("https://example.com/api/data", "POST");
        
        assert_ne!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key2, key3);
    }

    #[test]
    fn test_cache_expiry_check() {
        let cache = ResponseCache::default();
        let response = Response::builder()
            .status(StatusCode::OK)
            .header("cache-control", "max-age=3600")
            .body(())
            .unwrap();

        assert!(cache.is_fresh(&response));
    }

    #[test]
    fn test_cache_with_no_cache_header() {
        let cache = ResponseCache::default();
        let response = Response::builder()
            .status(StatusCode::OK)
            .header("cache-control", "no-cache")
            .body(())
            .unwrap();

        assert!(!cache.should_cache(&response));
    }
}