//! Integration tests for the fluent HTTP/3 builder.

use fluent_ai_http3::{Header, Http3, HttpChunk, HttpError, HttpStreamExt};
use futures_util::StreamExt;
use http::{HeaderMap, HeaderValue};

#[tokio::test]
async fn test_fluent_builder_get_request() {
    // This test uses httpbin.org, a public testing service.
    let url = "https://httpbin.org/get";

    let stream = Http3::json()
        .url(url)
        .headers(|| {
            let mut map = std::collections::HashMap::new();
            map.insert(Header::X_CUSTOM_HEADER, "Cascade-Test");
            map
        })
        .api_key("test-api-key")
        .get(url);

    // The new API uses async collect
    let body: serde_json::Value = stream.collect().await;

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

    let mut head_received = false;
    while let Some(result) = stream.next().await {
        match result {
            Ok(HttpChunk::Head(status, headers)) => {
                assert_eq!(status, 200);
                assert!(headers.contains_key("content-type"));
                assert_eq!(headers.get("content-type").unwrap(), "application/json");
                head_received = true;
            }
            Ok(HttpChunk::Body(chunk)) => {
                assert!(!chunk.is_empty());
            }
            Err(e) => {
                panic!("Request failed with error: {:?}", e);
            }
        }
    }

    assert!(
        head_received,
        "Did not receive the head chunk of the response."
    );
}

#[tokio::test]
async fn basic_builder_flow() {
    // This test uses httpbin.org, a public testing service.
    let url = "https://httpbin.org/get";

    let stream = Http3::json()
        .url(url)
        .headers(|| {
            let mut map = std::collections::HashMap::new();
            map.insert(http::HeaderName::from_static("x-custom-header"), "Cascade-Test");
            map
        })
        .api_key("test-api-key")
        .get(url);

    // The new API uses async collect, which consumes the stream.
    let body: serde_json::Value = stream.collect().await;

    // Basic validation on the collected body.
    assert!(body.is_object(), "Response body should be a JSON object");
    let headers = body.get("headers").expect("Response should have headers");
    let custom_header = headers.get("X-Custom-Header").expect("Missing X-Custom-Header");
    assert_eq!(custom_header, "Cascade-Test");
}
