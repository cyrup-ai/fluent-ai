//! Builder module tests
//!
//! Tests for HTTP3 builder functionality, mirroring src/builder.rs

use fluent_ai_http3::{Http3, HttpStreamExt};

#[cfg(test)]
mod builder_tests {
    use super::*;

    #[tokio::test]
    async fn test_fluent_builder_get_request() {
        // This test uses httpbin.org, a public testing service.
        let url = "https://httpbin.org/get";

        let stream = Http3::json()
            .url(url)
            .headers([("x-custom-header", "Cascade-Test")])
            .api_key("test-api-key")
            .get(url);

        // The new API uses collect on the stream
        let responses: Vec<serde_json::Value> = stream.collect();
        let body_str = serde_json::to_string(&responses[0]).expect("Failed to serialize JSON");
        let body: serde_json::Value =
            serde_json::from_str(&body_str).expect("Failed to parse JSON");

        // Basic validation on the collected body
        assert!(body.is_object());
        assert!(body.get("headers").is_some());
        if let Some(headers) = body.get("headers") {
            if let Some(header) = headers.get("X-Custom-Header") {
                assert_eq!(header, "Cascade-Test");
            } else {
                panic!("Missing X-Custom-Header");
            }
        }
    }

    #[tokio::test]
    async fn basic_builder_flow() {
        // This test uses httpbin.org, a public testing service.
        let url = "https://httpbin.org/get";

        let stream = Http3::json()
            .url(url)
            .headers([("x-custom-header", "Cascade-Test")])
            .api_key("test-api-key")
            .get(url);

        // The new API uses collect on the stream
        let responses: Vec<serde_json::Value> = stream.collect();
        let body_str = serde_json::to_string(&responses[0]).expect("Failed to serialize JSON");
        let body: serde_json::Value =
            serde_json::from_str(&body_str).expect("Failed to parse JSON");

        // Basic validation on the collected body.
        assert!(body.is_object(), "Response body should be a JSON object");
        let headers = body.get("headers").expect("Response should have headers");
        let custom_header = headers
            .get("X-Custom-Header")
            .expect("Missing X-Custom-Header");
        assert_eq!(custom_header, "Cascade-Test");
    }
}
