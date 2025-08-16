//! Request execution with cookie support
//!
//! Request sending logic with cookie handling for HTTP/3 client.

use bytes::Bytes;
use fluent_ai_async::AsyncStream;
use http::{Request, Uri};

use super::types::H3Client;
use crate::response::HttpResponseChunk;

#[cfg(feature = "cookies")]
impl H3Client {
    /// Send HTTP request with cookie support
    pub fn send_request_with_cookies(
        &mut self,
        mut req: Request<Bytes>,
    ) -> AsyncStream<HttpResponseChunk> {
        // Add cookies from store if available
        if let Some(ref cookie_store) = self.cookie_store {
            self.add_cookies_to_request(&mut req, cookie_store);
        }

        let uri = req.uri().clone();
        self.execute_request_internal(req, uri)
    }

    /// Add cookies from store to request
    fn add_cookies_to_request(
        &self,
        req: &mut Request<Bytes>,
        cookie_store: &dyn crate::common::cookie::CookieStore,
    ) {
        let uri = req.uri();
        let cookies = cookie_store.get_cookies_for_url(uri);

        if !cookies.is_empty() {
            let cookie_header = cookies.join("; ");
            req.headers_mut().insert(
                http::header::COOKIE,
                cookie_header.parse().unwrap_or_default(),
            );
        }
    }

    /// Process response cookies
    pub fn process_response_cookies(&self, response_headers: &http::HeaderMap, request_uri: &Uri) {
        if let Some(ref cookie_store) = self.cookie_store {
            for cookie_header in response_headers.get_all(http::header::SET_COOKIE) {
                if let Ok(cookie_str) = cookie_header.to_str() {
                    cookie_store.set_cookie_from_response(request_uri, cookie_str);
                }
            }
        }
    }
}
