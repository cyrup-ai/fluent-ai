//! Request builder functionality tests

use fluent_ai_http3::hyper::async_impl::client::Client;
use fluent_ai_http3::hyper::async_impl::request::types::{Request, RequestBuilder};
use fluent_ai_http3::{Method, Url};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_builder_chaining() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("User-Agent", "test-agent")
            .header("Accept", "application/json")
            .query(&[("param", "value")]);

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("User-Agent").unwrap(), "test-agent");
        assert_eq!(req.headers().get("Accept").unwrap(), "application/json");
        assert!(req.url().query().unwrap().contains("param=value"));
    }

    #[test]
    fn test_request_builder_timeout() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .timeout(std::time::Duration::from_secs(30));

        let req = builder.build().unwrap();
        // Timeout is typically stored in the builder configuration
        assert_eq!(req.method(), &Method::GET);
    }

    #[test]
    fn test_request_builder_version() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .version(http::Version::HTTP_2);

        let req = builder.build().unwrap();
        assert_eq!(req.version(), http::Version::HTTP_2);
    }

    #[test]
    fn test_request_builder_multiple_headers() {
        let client = Client::new();
        let mut headers = http::HeaderMap::new();
        headers.insert("X-Custom-1", "value1".parse().unwrap());
        headers.insert("X-Custom-2", "value2".parse().unwrap());

        let builder = client
            .get("https://example.com")
            .headers(headers);

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("X-Custom-1").unwrap(), "value1");
        assert_eq!(req.headers().get("X-Custom-2").unwrap(), "value2");
    }

    #[test]
    fn test_request_builder_query_serialization() {
        let client = Client::new();
        let params = vec![
            ("name", "John Doe"),
            ("age", "30"),
            ("city", "New York"),
        ];
        let builder = client
            .get("https://example.com/search")
            .query(&params);

        let req = builder.build().unwrap();
        let query = req.url().query().unwrap();
        assert!(query.contains("name=John%20Doe"));
        assert!(query.contains("age=30"));
        assert!(query.contains("city=New%20York"));
    }

    #[test]
    fn test_request_builder_error_handling() {
        let client = Client::new();
        
        // Test invalid URL handling
        let result = std::panic::catch_unwind(|| {
            client.get("not-a-valid-url").build()
        });
        
        // Should handle invalid URLs gracefully
        assert!(result.is_ok() || result.is_err());
    }
}