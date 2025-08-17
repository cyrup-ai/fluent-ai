//! HTTP method implementations
//!
//! This module provides terminal methods for executing HTTP requests with different
//! builder states and configurations. The methods are organized into logical modules:
//!
//! - `no_body_methods`: GET, DELETE, and download operations for BodyNotSet state
//! - `body_methods`: POST, PUT, PATCH operations for BodySet state  
//! - `jsonpath_methods`: JSONPath streaming operations for JsonPathStreaming state
//!
//! All methods maintain the streaming-first architecture and provide comprehensive
//! documentation with examples.

pub mod body_methods;
pub mod jsonpath_methods;
pub mod no_body_methods;

// Re-export all method implementations for backward compatibility
pub use body_methods::{BodyMethods, PatchMethod, PostMethod, PutMethod};
pub use jsonpath_methods::{JsonPathGetMethod, JsonPathMethods, JsonPathPostMethod};
pub use no_body_methods::{DeleteMethod, DownloadMethod, GetMethod, NoBodyMethods};

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::builder::core::Http3Builder;
    use crate::client::HttpClient;

    fn create_test_client() -> Arc<HttpClient> {
        Arc::new(HttpClient::default())
    }

    #[test]
    fn test_no_body_methods_compilation() {
        let client = create_test_client();

        // Test GET method compilation
        let _get_stream = Http3Builder::new(&client).get("https://example.com/api");

        // Test DELETE method compilation
        let _delete_stream =
            Http3Builder::new(&client).delete("https://example.com/api/resource/123");

        // Test download_file method compilation
        let _download_builder =
            Http3Builder::new(&client).download_file("https://example.com/file.zip");
    }

    #[test]
    fn test_body_methods_compilation() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct TestData {
            name: String,
            value: i32,
        }

        let client = create_test_client();
        let test_data = TestData {
            name: "test".to_string(),
            value: 42,
        };

        // Test POST method compilation
        let _post_stream = Http3Builder::json()
            .body(&test_data)
            .post("https://example.com/api");

        // Test PUT method compilation
        let _put_stream = Http3Builder::json()
            .body(&test_data)
            .put("https://example.com/api/resource/123");

        // Test PATCH method compilation
        let _patch_stream = Http3Builder::json()
            .body(&test_data)
            .patch("https://example.com/api/resource/123");
    }

    #[test]
    fn test_jsonpath_methods_compilation() {
        use fluent_ai_async::prelude::MessageChunk;
        use serde::{Deserialize, Serialize};

        #[derive(Serialize)]
        struct Query {
            filter: String,
        }

        #[derive(Deserialize, Default)]
        struct TestResult {
            id: u64,
            name: String,
        }

        // Implement required traits for TestResult
        impl MessageChunk for TestResult {
            fn bad_chunk(_error: String) -> Self {
                Self::default()
            }
        }

        impl MessageChunk for TestResult {
            fn bad_chunk(_error: String) -> Self {
                Self::default()
            }
        }

        let client = create_test_client();
        let query = Query {
            filter: "test".to_string(),
        };

        // Test JSONPath GET method compilation
        let _jsonpath_get_stream = Http3Builder::json()
            .array_stream("$.results[*]")
            .get::<TestResult>("https://example.com/api");

        // Test JSONPath POST method compilation
        let _jsonpath_post_stream = Http3Builder::json()
            .body(&query)
            .array_stream("$.results[*]")
            .post::<TestResult>("https://api.example.com/search");
    }

    #[test]
    fn test_method_debug_logging() {
        let client = create_test_client();

        // Test debug logging is properly configured
        let _debug_stream = Http3Builder::new(&client)
            .debug()
            .get("https://example.com/api");
    }

    #[test]
    fn test_builder_state_transitions() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct TestBody {
            data: String,
        }

        let client = create_test_client();
        let body = TestBody {
            data: "test".to_string(),
        };

        // Test BodyNotSet -> HttpStream
        let _no_body_stream = Http3Builder::new(&client).get("https://example.com/api");

        // Test BodySet -> HttpStream
        let _body_stream = Http3Builder::json()
            .body(&body)
            .post("https://example.com/api");

        // Test JsonPathStreaming -> JsonPathStream
        let _jsonpath_stream = Http3Builder::json()
            .array_stream("$[*]")
            .get::<serde_json::Value>("https://example.com/api");
    }
}
