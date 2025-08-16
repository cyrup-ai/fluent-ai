//! Debug trait implementations for Request types
//!
//! Provides Debug formatting for Request and RequestBuilder structs.

use std::fmt;

use super::builder::RequestBuilder;
use super::request::Request;

impl fmt::Debug for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt_request_fields(&self.method, &self.url, &self.headers, &self.version, f)
    }
}

impl fmt::Debug for RequestBuilder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.request {
            Ok(ref req) => fmt_request_fields(&req.method, &req.url, &req.headers, &req.version, f),
            Err(ref err) => f
                .debug_struct("RequestBuilder")
                .field("error", err)
                .finish(),
        }
    }
}

fn fmt_request_fields(
    method: &http::Method,
    url: &url::Url,
    headers: &http::HeaderMap,
    version: &http::Version,
    f: &mut fmt::Formatter,
) -> fmt::Result {
    f.debug_struct("Request")
        .field("method", method)
        .field("url", url)
        .field("headers", headers)
        .field("version", version)
        .finish()
}
