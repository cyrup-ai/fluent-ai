//! Public API for H3 client
//!
//! Main public request method converting String body to Bytes.

use bytes::Bytes;
use fluent_ai_async::AsyncStream;
use http::Request;

use super::types::H3Client;
use crate::response::HttpResponseChunk;

impl H3Client {
    /// Main public request method
    pub fn request(&mut self, req: Request<String>) -> AsyncStream<HttpResponseChunk> {
        // Convert String body to Bytes
        let (parts, body) = req.into_parts();
        let bytes_body = Bytes::from(body);
        let bytes_req = Request::from_parts(parts, bytes_body);

        // Choose appropriate execution path based on cookie support
        #[cfg(feature = "cookies")]
        {
            if self.cookie_store.is_some() {
                self.send_request_with_cookies(bytes_req)
            } else {
                self.send_request_no_cookies(bytes_req)
            }
        }

        #[cfg(not(feature = "cookies"))]
        {
            self.send_request_no_cookies(bytes_req)
        }
    }
}
