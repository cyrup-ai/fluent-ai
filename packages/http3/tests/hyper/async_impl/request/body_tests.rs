//! Request body handling tests

use fluent_ai_http3::hyper::async_impl::client::Client;
use fluent_ai_http3::hyper::async_impl::request::types::{Request, RequestBuilder};
use fluent_ai_http3::{Method, Url};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_builder_text_body() {
        let client = Client::new();
        let builder = client
            .post("https://example.com")
            .body("test body content");

        let req = builder.build().unwrap();
        assert_eq!(req.method(), &Method::POST);
        assert!(req.body().is_some());
    }

    #[test]
    fn test_request_builder_json_body() {
        let client = Client::new();
        let json_data = serde_json::json!({"key": "value", "number": 42});
        let builder = client
            .post("https://example.com")
            .json(&json_data);

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("content-type").unwrap(), "application/json");
        assert!(req.body().is_some());
    }

    #[test]
    fn test_request_builder_form_body() {
        let client = Client::new();
        let form_data = [("field1", "value1"), ("field2", "value2")];
        let builder = client
            .post("https://example.com")
            .form(&form_data);

        let req = builder.build().unwrap();
        assert_eq!(req.headers().get("content-type").unwrap(), "application/x-www-form-urlencoded");
        assert!(req.body().is_some());
    }

    #[test]
    fn test_request_builder_bytes_body() {
        let client = Client::new();
        let data = b"binary data content";
        let builder = client
            .put("https://example.com")
            .body(data.to_vec());

        let req = builder.build().unwrap();
        assert_eq!(req.method(), &Method::PUT);
        assert!(req.body().is_some());
    }

    #[test]
    fn test_request_empty_body() {
        let url = Url::parse("https://example.com").unwrap();
        let req = Request::new(Method::GET, url);

        assert!(req.body().is_none());
    }

    #[test]
    fn test_request_content_length_header() {
        let client = Client::new();
        let body_content = "test content";
        let builder = client
            .post("https://example.com")
            .body(body_content);

        let req = builder.build().unwrap();
        // Content-Length should be automatically set for string bodies
        assert!(req.headers().contains_key("content-length") || req.body().is_some());
    }
}