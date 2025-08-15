use super::super::client::Client;
use super::types::{Request, RequestBuilder};
use crate::{Method, Url};
use std::collections::HashMap;

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
    fn test_request_builder_header() {
        let client = Client::new();
        let builder = client.get("https://example.com")
            .header("X-Custom-Header", "test-value");
        
        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("X-Custom-Header").unwrap(), "test-value");
    }

    #[test]
    fn test_request_builder_basic_auth() {
        let client = Client::new();
        let builder = client.get("https://example.com")
            .basic_auth("username", Some("password"));
        
        let req = builder.build().unwrap();
        assert!(req.headers().contains_key("authorization"));
        assert!(req.headers().get("authorization").unwrap().is_sensitive());
    }

    #[test]
    fn test_request_builder_bearer_auth() {
        let client = Client::new();
        let builder = client.get("https://example.com")
            .bearer_auth("token123");
        
        let req = builder.build().unwrap();
        let auth_header = req.headers().get("authorization").unwrap();
        assert_eq!(auth_header, "Bearer token123");
        assert!(auth_header.is_sensitive());
    }

    #[test]
    fn test_request_builder_form() {
        let client = Client::new();
        let mut form_data = HashMap::new();
        form_data.insert("key1", "value1");
        form_data.insert("key2", "value2");
        
        let builder = client.post("https://example.com")
            .form(&form_data);
        
        let req = builder.build().unwrap();
        assert_eq!(
            req.headers().get("content-type").unwrap(),
            "application/x-www-form-urlencoded"
        );
        assert!(req.body().is_some());
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_request_builder_json() {
        let client = Client::new();
        let json_data = serde_json::json!({
            "name": "test",
            "value": 42
        });
        
        let builder = client.post("https://example.com")
            .json(&json_data);
        
        let req = builder.build().unwrap();
        assert_eq!(
            req.headers().get("content-type").unwrap(),
            "application/json"
        );
        assert!(req.body().is_some());
    }

    #[test]
    fn test_request_builder_timeout() {
        use std::time::Duration;
        
        let client = Client::new();
        let builder = client.get("https://example.com")
            .timeout(Duration::from_secs(30));
        
        let req = builder.build().unwrap();
        assert_eq!(req.timeout(), Some(&Duration::from_secs(30)));
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
        let builder = client.post("https://example.com")
            .header("X-Test", "value")
            .body("test body");
        
        let cloned = builder.try_clone().unwrap();
        let req = cloned.build().unwrap();
        
        assert_eq!(req.method(), &Method::POST);
        assert_eq!(req.headers().get("X-Test").unwrap(), "value");
    }

    #[test]
    fn test_request_builder_headers() {
        let client = Client::new();
        let mut headers = crate::header::HeaderMap::new();
        headers.insert("X-Custom-1", "value1".parse().unwrap());
        headers.insert("X-Custom-2", "value2".parse().unwrap());
        
        let builder = client.get("https://example.com")
            .headers(headers);
        
        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("X-Custom-1").unwrap(), "value1");
        assert_eq!(req.headers().get("X-Custom-2").unwrap(), "value2");
    }

    #[test]
    fn test_request_builder_user_agent() {
        let client = Client::new();
        let builder = client.get("https://example.com")
            .user_agent("test-agent/1.0");
        
        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("user-agent").unwrap(), "test-agent/1.0");
    }

    #[test]
    fn test_request_builder_content_type() {
        let client = Client::new();
        let builder = client.post("https://example.com")
            .content_type("application/xml");
        
        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("content-type").unwrap(), "application/xml");
    }

    #[test]
    fn test_request_builder_api_key() {
        let client = Client::new();
        let builder = client.get("https://example.com")
            .api_key("secret-key-123");
        
        let req = builder.build().unwrap();
        let api_key_header = req.headers().get("X-API-Key").unwrap();
        assert_eq!(api_key_header, "secret-key-123");
        assert!(api_key_header.is_sensitive());
    }

    #[test]
    fn test_request_builder_custom_auth() {
        let client = Client::new();
        let builder = client.get("https://example.com")
            .custom_auth("Custom", "credentials123");
        
        let req = builder.build().unwrap();
        let auth_header = req.headers().get("authorization").unwrap();
        assert_eq!(auth_header, "Custom credentials123");
        assert!(auth_header.is_sensitive());
    }

    #[test]
    fn test_request_builder_text_body() {
        let client = Client::new();
        let builder = client.post("https://example.com")
            .text("Hello, world!");
        
        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("content-type").unwrap(), "text/plain");
        assert!(req.body().is_some());
    }

    #[test]
    fn test_request_builder_bytes_body() {
        let client = Client::new();
        let data = vec![1, 2, 3, 4, 5];
        let builder = client.post("https://example.com")
            .bytes(data);
        
        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("content-type").unwrap(), "application/octet-stream");
        assert!(req.body().is_some());
    }

    #[test]
    fn test_request_builder_empty_body() {
        let client = Client::new();
        let builder = client.post("https://example.com")
            .body("initial body")
            .empty_body();
        
        let req = builder.build().unwrap();
        assert!(req.body().is_none());
    }

    #[test]
    fn test_request_builder_remove_header() {
        let client = Client::new();
        let builder = client.get("https://example.com")
            .header("X-Test", "value")
            .remove_header("X-Test");
        
        let req = builder.build().unwrap();
        assert!(!req.headers().contains_key("X-Test"));
    }

    #[test]
    fn test_request_builder_clear_headers() {
        let client = Client::new();
        let builder = client.get("https://example.com")
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
        let builder = client.get("https://example.com")
            .header("X-Test", "value");
        
        assert!(builder.has_header("X-Test"));
        assert!(!builder.has_header("X-Missing"));
    }

    #[test]
    fn test_request_builder_get_header() {
        let client = Client::new();
        let builder = client.get("https://example.com")
            .header("X-Test", "test-value");
        
        assert_eq!(builder.get_header("X-Test").unwrap(), "test-value");
        assert!(builder.get_header("X-Missing").is_none());
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

    #[test]
    fn test_request_debug() {
        let url = Url::parse("https://example.com").unwrap();
        let req = Request::new(Method::GET, url);
        
        let debug_str = format!("{:?}", req);
        assert!(debug_str.contains("Request"));
        assert!(debug_str.contains("GET"));
        assert!(debug_str.contains("example.com"));
    }

    #[test]
    fn test_request_builder_debug() {
        let client = Client::new();
        let builder = client.get("https://example.com");
        
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("RequestBuilder"));
    }

    #[test]
    fn test_request_builder_error_state() {
        let client = Client::new();
        let builder = client.get("invalid-url")
            .header("Invalid\nHeader", "value");
        
        let result = builder.build();
        assert!(result.is_err());
    }
}