//! Matcher functionality and URL handling tests

#[cfg(test)]
mod tests {
    use super::super::utilities::*;

    #[test]
    fn test_matcher_system() {
        let matcher = super::super::super::Matcher::system();

        // System matcher should be created successfully
        assert!(matcher.maybe_has_http_auth);
        assert!(matcher.maybe_has_http_custom_headers);
    }

    #[test]
    fn test_proxy_url_message_chunk() {
        use fluent_ai_async::prelude::MessageChunk;

        use super::super::super::url_handling::ProxyUrl;

        // Test successful URL creation
        let url = crate::Url::parse("http://example.com").unwrap();
        let proxy_url = ProxyUrl::new(url.clone());
        assert_eq!(proxy_url.into_url(), url);

        // Test bad chunk creation
        let bad_chunk = ProxyUrl::bad_chunk("Test error".to_string());
        assert!(bad_chunk.error().is_some());
        assert_eq!(bad_chunk.error().unwrap(), "Test error");
    }

    #[test]
    fn test_matcher_patterns() {
        use super::super::super::matcher::matcher::Matcher;

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
    fn test_matcher_builder() {
        use super::super::super::matcher::matcher::MatcherBuilder;

        let builder = MatcherBuilder::new()
            .all("http://proxy1.com".to_string())
            .no("localhost")
            .http("http://proxy2.com".to_string())
            .https("https://proxy3.com".to_string());

        let matcher = builder.build();

        // Builder should create matcher successfully
        assert_eq!(matcher.patterns.len(), 1); // Only "localhost" in no_patterns
        assert_eq!(matcher.patterns[0], "localhost");
    }
}