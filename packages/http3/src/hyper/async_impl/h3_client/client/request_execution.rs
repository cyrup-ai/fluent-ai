//! Request execution without cookie support
//!
//! Core request sending logic for HTTP/3 client without cookie handling.

use bytes::Bytes;
use fluent_ai_async::AsyncStream;
use http::{Request, Uri};

use super::types::H3Client;
use crate::response::HttpResponseChunk;

impl H3Client {
    /// Send HTTP request without cookie support
    pub fn send_request_no_cookies(
        &mut self,
        req: Request<Bytes>,
    ) -> AsyncStream<HttpResponseChunk> {
        let uri = req.uri().clone();
        self.execute_request_internal(req, uri)
    }

    /// Internal request execution
    fn execute_request_internal(
        &mut self,
        _req: Request<Bytes>,
        uri: Uri,
    ) -> AsyncStream<HttpResponseChunk> {
        // Delegate to connection establishment
        self.establish_connection(uri)
    }

    /// Validate request before sending
    fn validate_request(req: &Request<Bytes>) -> Result<(), String> {
        if req.uri().host().is_none() {
            return Err("Request URI must have a host".to_string());
        }

        if req.method().as_str().is_empty() {
            return Err("Request method cannot be empty".to_string());
        }

        Ok(())
    }

    /// Prepare request headers
    fn prepare_headers(req: &mut Request<Bytes>) {
        let headers = req.headers_mut();

        // Add default headers if not present
        if !headers.contains_key("user-agent") {
            if let Ok(user_agent_value) = "fluent-ai-http3/1.0".parse() {
                headers.insert("user-agent", user_agent_value);
            }
        }

        if !headers.contains_key("accept") {
            if let Ok(accept_value) = "*/*".parse() {
                headers.insert("accept", accept_value);
            }
        }
    }
}
