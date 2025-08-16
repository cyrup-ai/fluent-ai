//! Matcher functionality and URL handling tests

use fluent_ai_http3::hyper::proxy::{Matcher, ProxyUrl};
use fluent_ai_http3::Url;
use fluent_ai_async::prelude::MessageChunk;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matcher_system() {
        let matcher = Matcher::system();

        // System matcher should be created successfully
        assert!(matcher.maybe_has_http_auth);
        assert!(matcher.maybe_has_http_custom_headers);
    }

    #[test]
    fn test_proxy_url_message_chunk() {
        // Test successful URL creation
        let url = Url::parse("http://example.com").unwrap();
        let proxy_url = ProxyUrl::new(url.clone());
        assert_eq!(proxy_url.into_url(), url);

        // Test bad chunk creation
        let bad_chunk = ProxyUrl::bad_chunk("Test error".to_string());
        assert!(bad_chunk.error().is_some());
        assert_eq!(bad_chunk.error().unwrap(), "Test error");
    }

    #[test]
    fn test_matcher_patterns() {
        let patterns = vec!["*.example.com".to_string(), "localhost".to_string()];
        let matcher = Matcher::new(patterns);

        // Test wildcard matching
        let uri: http::Uri = "http://sub.example.com".parse().unwrap();
        assert!(matcher.matches(&uri));

        // Test exact matching
        let uri: http::Uri = "http://localhost".parse().unwrap();
        assert!(matcher.matches(&uri));

        // Test non-matching
        let uri: http::Uri = "http://other.com".parse().unwrap();
        assert!(!matcher.matches(&uri));
    }

    #[test]
    fn test_matcher_intercept() {
        let target = "http://proxy.example.com";
        let matcher = fluent_ai_http3::hyper::proxy::Proxy::all(target)
            .expect("proxy should create")
            .into_matcher();

        let url = Url::parse("http://test.com").unwrap();
        let intercepted = matcher.intercept(&url);
        assert!(intercepted.is_some());

        let intercepted = intercepted.unwrap();
        assert_eq!(intercepted.uri().to_string(), target);
    }
}