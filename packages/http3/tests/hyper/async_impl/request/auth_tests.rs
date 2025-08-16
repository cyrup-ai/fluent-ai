//! Authentication tests for HTTP request functionality

use fluent_ai_http3::hyper::async_impl::client::Client;
use fluent_ai_http3::hyper::async_impl::request::types::{Request, RequestBuilder};
use fluent_ai_http3::{Method, Url};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_builder_basic_auth() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .basic_auth("username", Some("password"));

        let req = builder.build().unwrap();
        assert!(req.headers().contains_key("authorization"));
        assert!(req.headers().get("authorization").unwrap().is_sensitive());
    }

    #[test]
    fn test_request_builder_bearer_auth() {
        let client = Client::new();
        let builder = client.get("https://example.com").bearer_auth("token123");

        let req = builder.build().unwrap();
        let auth_header = req.headers().get("authorization").unwrap();
        assert_eq!(auth_header, "Bearer token123");
    }

    #[test]
    fn test_request_builder_custom_auth() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("authorization", "Custom auth-token");

        let req = builder.build().unwrap();
        let auth_header = req.headers().get("authorization").unwrap();
        assert_eq!(auth_header, "Custom auth-token");
    }

    #[test]
    fn test_request_auth_headers_case_insensitive() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .header("Authorization", "Bearer test-token");

        let req = builder.build().unwrap();
        assert!(req.headers().contains_key("authorization"));
    }

    #[test]
    fn test_multiple_auth_methods() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .basic_auth("user", Some("pass"))
            .bearer_auth("token");

        let req = builder.build().unwrap();
        // Last auth method should win
        let auth_header = req.headers().get("authorization").unwrap();
        assert_eq!(auth_header, "Bearer token");
    }
}