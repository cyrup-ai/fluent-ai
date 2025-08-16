use std::collections::HashMap;

use super::super::super::client::Client;
use super::super::types::{Request, RequestBuilder};
use crate::{Method, Url};

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
        assert!(auth_header.is_sensitive());
    }

    #[test]
    fn test_request_builder_api_key() {
        let client = Client::new();
        let builder = client.get("https://example.com").api_key("secret-key-123");

        let req = builder.build().unwrap();
        let api_key_header = req.headers().get("X-API-Key").unwrap();
        assert_eq!(api_key_header, "secret-key-123");
        assert!(api_key_header.is_sensitive());
    }

    #[test]
    fn test_request_builder_custom_auth() {
        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .custom_auth("Custom", "credentials123");

        let req = builder.build().unwrap();
        let auth_header = req.headers().get("authorization").unwrap();
        assert_eq!(auth_header, "Custom credentials123");
        assert!(auth_header.is_sensitive());
    }
}
