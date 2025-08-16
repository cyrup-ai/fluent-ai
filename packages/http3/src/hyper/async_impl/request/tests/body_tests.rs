use std::collections::HashMap;

use super::super::super::client::Client;
use super::super::types::{Request, RequestBuilder};
use crate::{Method, Url};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_builder_form() {
        let client = Client::new();
        let mut form_data = HashMap::new();
        form_data.insert("key1", "value1");
        form_data.insert("key2", "value2");

        let builder = client.post("https://example.com").form(&form_data);

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

        let builder = client.post("https://example.com").json(&json_data);

        let req = builder.build().unwrap();
        assert_eq!(
            req.headers().get("content-type").unwrap(),
            "application/json"
        );
        assert!(req.body().is_some());
    }

    #[test]
    fn test_request_builder_text_body() {
        let client = Client::new();
        let builder = client.post("https://example.com").text("Hello, world!");

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("content-type").unwrap(), "text/plain");
        assert!(req.body().is_some());
    }

    #[test]
    fn test_request_builder_bytes_body() {
        let client = Client::new();
        let data = vec![1, 2, 3, 4, 5];
        let builder = client.post("https://example.com").bytes(data);

        let req = builder.build().unwrap();
        assert_eq!(
            req.headers().get("content-type").unwrap(),
            "application/octet-stream"
        );
        assert!(req.body().is_some());
    }

    #[test]
    fn test_request_builder_empty_body() {
        let client = Client::new();
        let builder = client
            .post("https://example.com")
            .body("initial body")
            .empty_body();

        let req = builder.build().unwrap();
        assert!(req.body().is_none());
    }
}
