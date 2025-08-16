use std::collections::HashMap;

use super::super::super::client::Client;
use super::super::types::{Request, RequestBuilder};
use crate::{Method, Url};

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
    fn test_request_builder_try_clone() {
        let client = Client::new();
        let builder = client
            .post("https://example.com")
            .header("X-Test", "value")
            .body("test body");

        let cloned = builder.try_clone().unwrap();
        let req = cloned.build().unwrap();

        assert_eq!(req.method(), &Method::POST);
        assert_eq!(req.headers().get("X-Test").unwrap(), "value");
    }

    #[test]
    fn test_request_extensions() {
        let url = Url::parse("https://example.com").unwrap();
        let mut req = Request::new(Method::GET, url);

        req.extensions_mut().insert("test-data");
        assert_eq!(req.extensions().get::<&str>(), Some(&"test-data"));
    }

    #[test]
    fn test_request_version() {
        let url = Url::parse("https://example.com").unwrap();
        let mut req = Request::new(Method::GET, url);

        *req.version_mut() = http::Version::HTTP_2;
        assert_eq!(req.version(), http::Version::HTTP_2);
    }
}
