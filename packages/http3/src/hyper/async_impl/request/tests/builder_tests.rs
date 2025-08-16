use std::collections::HashMap;

use super::super::super::client::Client;
use super::super::types::{Request, RequestBuilder};
use crate::{Method, Url};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_builder_timeout() {
        use std::time::Duration;

        let client = Client::new();
        let builder = client
            .get("https://example.com")
            .timeout(Duration::from_secs(30));

        let req = builder.build().unwrap();
        assert_eq!(req.timeout(), Some(&Duration::from_secs(30)));
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
        let builder = client.get("invalid-url").header("Invalid\nHeader", "value");

        let result = builder.build();
        assert!(result.is_err());
    }
}
