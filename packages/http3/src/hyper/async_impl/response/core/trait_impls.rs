//! MessageChunk and Default implementations
//!
//! This module contains trait implementations for Response including MessageChunk and Default.

use fluent_ai_async::prelude::MessageChunk;

use super::types::Response;

impl MessageChunk for Response {
    fn is_chunk(&self) -> bool {
        false // Response is a complete message, not a chunk
    }

    fn chunk_type(&self) -> &'static str {
        "response"
    }

    fn is_error(&self) -> bool {
        self.status().is_client_error() || self.status().is_server_error()
    }

    fn error_message(&self) -> Option<String> {
        if self.is_error() {
            Some(format!(
                "HTTP {} {}",
                self.status().as_u16(),
                self.status().canonical_reason().unwrap_or("Unknown")
            ))
        } else {
            None
        }
    }

    fn bad_chunk(error: String) -> Self {
        // Create a minimal error response
        use std::pin::Pin;
        use std::time::Duration;

        use hyper::{Response as HyperResponse, StatusCode};
        use url::Url;

        use super::super::super::body::ResponseBody;
        use super::super::super::decoder::Accepts;

        let hyper_response = HyperResponse::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(ResponseBody::empty())
            .unwrap();

        let url = Url::parse("http://error.local").unwrap();

        Response::new(hyper_response, url, Accepts::default(), None, None)
    }
}

impl Default for Response {
    fn default() -> Self {
        use hyper::{Response as HyperResponse, StatusCode};
        use url::Url;

        use super::super::super::body::ResponseBody;
        use super::super::super::decoder::Accepts;

        let hyper_response = HyperResponse::builder()
            .status(StatusCode::OK)
            .body(ResponseBody::empty())
            .unwrap();

        let url = Url::parse("http://default.local").unwrap();

        Response::new(hyper_response, url, Accepts::default(), None, None)
    }
}
