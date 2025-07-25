//! Integration tests for the fluent HTTP/3 builder.

use fluent_ai_http3::{Http3, HttpStreamExt};
use http::HeaderName;

#[tokio::test]
async fn test_fluent_builder_get_request() {
    // This test uses httpbin.org, a public testing service.
    let url = "https://httpbin.org/get";

    let stream = Http3::json()
        .url(url)
        .headers(|| {
            let mut map = std::collections::HashMap::new();
            map.insert(HeaderName::from_static("x-custom-header"), "Cascade-Test");
            map
        })
        .api_key("test-api-key")
        .get(url);

    // The new API uses async collect
    let body: serde_json::Value = HttpStreamExt::collect(stream);

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
        .headers(|| {
            let mut map = std::collections::HashMap::new();
            map.insert(HeaderName::from_static("x-custom-header"), "Cascade-Test");
            map
        })
        .api_key("test-api-key")
        .get(url);

    // The new API uses async collect, which consumes the stream.
    let body: serde_json::Value = HttpStreamExt::collect(stream);

    // Basic validation on the collected body.
    assert!(body.is_object(), "Response body should be a JSON object");
    let headers = body.get("headers").expect("Response should have headers");
    let custom_header = headers
        .get("X-Custom-Header")
        .expect("Missing X-Custom-Header");
    assert_eq!(custom_header, "Cascade-Test");
}
