//! Integration tests for the fluent HTTP/3 builder.

use fluent_ai_http3::{Http3, HttpChunk, HttpError};
use futures_util::StreamExt;
use http::{HeaderMap, HeaderValue};

#[tokio::test]
async fn test_fluent_builder_get_request() {
    // This test uses httpbin.org, a public testing service.
    let url = "https://httpbin.org/get";

    let stream = Http3::json()
        .headers(|| {
            let mut map = std::collections::HashMap::new();
            map.insert("X-Custom-Header".to_string(), HeaderValue::from_static("Cascade-Test"));
            map
        })
        .api_key("test-api-key")
        .get(url);

    let results: Vec<Result<HttpChunk, HttpError>> = stream.collect().await;

    let mut head_received = false;
    for result in results {
        match result {
            Ok(HttpChunk::Head(status, headers)) => {
                assert_eq!(status.as_u16(), 200);
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
