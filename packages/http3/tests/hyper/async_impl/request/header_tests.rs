//! Request header handling tests

use fluent_ai_http3::hyper::async_impl::client::Client;
use fluent_ai_http3::hyper::async_impl::request::types::{Request, RequestBuilder};
use fluent_ai_http3::{Method, Url};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_builder_single_header() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("Content-Type", "application/json");

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("Content-Type").unwrap(), "application/json");
    }

    #[test]
    fn test_request_builder_header_override() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("Accept", "text/html")
            .header("Accept", "application/json");

        let req = builder.build().unwrap();
        // Last header should win
        assert_eq!(req.headers().get("Accept").unwrap(), "application/json");
    }

    #[test]
    fn test_request_builder_case_insensitive_headers() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("content-type", "text/plain")
            .header("ACCEPT", "text/html");

        let req = builder.build().unwrap();
        assert!(req.headers().contains_key("content-type"));
        assert!(req.headers().contains_key("accept"));
    }

    #[test]
    fn test_request_builder_user_agent() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("User-Agent", "MyApp/1.0");

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("User-Agent").unwrap(), "MyApp/1.0");
    }

    #[test]
    fn test_request_builder_custom_headers() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("X-API-Key", "secret-key")
            .header("X-Request-ID", "12345");

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("X-API-Key").unwrap(), "secret-key");
        assert_eq!(req.headers().get("X-Request-ID").unwrap(), "12345");
    }

    #[test]
    fn test_request_builder_header_map() {
        let client = Client::new();
        let mut headers = http::HeaderMap::new();
        headers.insert("Authorization", "Bearer token".parse().unwrap());
        headers.insert("Content-Length", "100".parse().unwrap());

        let builder = client
            .post("https://example.com")
            .headers(headers);

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("Authorization").unwrap(), "Bearer token");
        assert_eq!(req.headers().get("Content-Length").unwrap(), "100");
    }
}