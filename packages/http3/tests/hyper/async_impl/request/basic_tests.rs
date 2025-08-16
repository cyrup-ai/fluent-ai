//! Basic request functionality tests

use fluent_ai_http3::hyper::async_impl::client::Client;
use fluent_ai_http3::hyper::async_impl::request::types::{Request, RequestBuilder};
use fluent_ai_http3::{Method, Url};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_new() {
        let url = Url::parse("https://example.com").unwrap();
        let req = Request::new(Method::GET, url.clone());

        assert_eq!(req.method(), &Method::GET);
        assert_eq!(req.url(), &url);
        assert!(req.body().is_none());
    }

    #[test]
    fn test_request_try_clone() {
        let url = Url::parse("https://example.com").unwrap();
        let req = Request::new(Method::POST, url);

        let cloned = req.try_clone().unwrap();
        assert_eq!(req.method(), cloned.method());
        assert_eq!(req.url(), cloned.url());
    }

    #[test]
    fn test_request_builder_url() {
        let client = Client::new();
        let builder = client.get("https://example.com/path");

        let req = builder.build().unwrap();
        assert_eq!(req.url().as_str(), "https://example.com/path");
        assert_eq!(req.method(), &Method::GET);
    }

    #[test]
    fn test_request_builder_method() {
        let client = Client::new();
        let builder = client.request(Method::PUT, "https://example.com");

        let req = builder.build().unwrap();
        assert_eq!(req.method(), &Method::PUT);
    }

    #[test]
    fn test_request_builder_query_params() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .query(&[("key1", "value1"), ("key2", "value2")]);

        let req = builder.build().unwrap();
        let url_str = req.url().as_str();
        assert!(url_str.contains("key1=value1"));
        assert!(url_str.contains("key2=value2"));
    }

    #[test]
    fn test_request_builder_headers() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("X-Custom-Header", "custom-value")
            .header("Content-Type", "application/json");

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("X-Custom-Header").unwrap(), "custom-value");
        assert_eq!(req.headers().get("Content-Type").unwrap(), "application/json");
    }

    #[test]
    fn test_request_version() {
        let url = Url::parse("https://example.com").unwrap();
        let req = Request::new(Method::GET, url);

        // Default HTTP version should be HTTP/1.1
        assert_eq!(req.version(), http::Version::HTTP_11);
    }
}