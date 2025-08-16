use std::collections::HashMap;

use super::super::super::client::Client;
use super::super::types::{Request, RequestBuilder};
use crate::{Method, Url};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_builder_header() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("X-Custom-Header", "test-value");

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("X-Custom-Header").unwrap(), "test-value");
    }

    #[test]
    fn test_request_builder_headers() {
        let client = Client::new();
        let mut headers = crate::header::HeaderMap::new();
        headers.insert("X-Custom-1", "value1".parse().unwrap());
        headers.insert("X-Custom-2", "value2".parse().unwrap());

        let builder = client.get("https://example.com").headers(headers);

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("X-Custom-1").unwrap(), "value1");
        assert_eq!(req.headers().get("X-Custom-2").unwrap(), "value2");
    }

    #[test]
    fn test_request_builder_user_agent() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .user_agent("test-agent/1.0");

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("user-agent").unwrap(), "test-agent/1.0");
    }

    #[test]
    fn test_request_builder_content_type() {
        let client = Client::new();
        let builder = client
            .post("https://example.com")
            .content_type("application/xml");

        let req = builder.build().unwrap();
        assert_eq!(
            req.headers().get("content-type").unwrap(),
            "application/xml"
        );
    }

    #[test]
    fn test_request_builder_remove_header() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("X-Test", "value")
            .remove_header("X-Test");

        let req = builder.build().unwrap();
        assert!(!req.headers().contains_key("X-Test"));
    }

    #[test]
    fn test_request_builder_clear_headers() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("X-Test-1", "value1")
            .header("X-Test-2", "value2")
            .clear_headers();

        let req = builder.build().unwrap();
        assert!(!req.headers().contains_key("X-Test-1"));
        assert!(!req.headers().contains_key("X-Test-2"));
    }

    #[test]
    fn test_request_builder_has_header() {
        let client = Client::new();
        let builder = client.get("https://example.com").header("X-Test", "value");

        assert!(builder.has_header("X-Test"));
        assert!(!builder.has_header("X-Missing"));
    }

    #[test]
    fn test_request_builder_get_header() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("X-Test", "test-value");

        assert_eq!(builder.get_header("X-Test").unwrap(), "test-value");
        assert!(builder.get_header("X-Missing").is_none());
    }
}
