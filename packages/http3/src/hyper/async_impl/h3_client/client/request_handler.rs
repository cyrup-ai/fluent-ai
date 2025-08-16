//! Request handler for H3 client
//!
//! Execute request method called from client.rs with complete implementation.

use bytes::Bytes;
use fluent_ai_async::AsyncStream;
use http::Request;

use super::types::H3Client;
use crate::response::HttpResponseChunk;

impl H3Client {
    /// Execute request - main entry point
    pub fn execute_request(&mut self, req: Request<Bytes>) -> AsyncStream<HttpResponseChunk> {
        // Validate request first
        if let Err(error) = Self::validate_request(&req) {
            return AsyncStream::with_channel(move |sender| {
                fluent_ai_async::emit!(sender, HttpResponseChunk::bad_chunk(error));
            });
        }

        // Choose execution path based on features
        #[cfg(feature = "cookies")]
        {
            if self.cookie_store.is_some() {
                self.send_request_with_cookies(req)
            } else {
                self.send_request_no_cookies(req)
            }
        }

        #[cfg(not(feature = "cookies"))]
        {
            self.send_request_no_cookies(req)
        }
    }

    /// Execute request with advanced features
    pub fn execute_request_with_options(
        &mut self,
        req: Request<Bytes>,
        _options: RequestOptions,
    ) -> AsyncStream<HttpResponseChunk> {
        self.execute_request_advanced(req)
    }
}

/// Request execution options
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    pub timeout_ms: Option<u64>,
    pub retry_count: Option<u32>,
    pub enable_push: bool,
}
