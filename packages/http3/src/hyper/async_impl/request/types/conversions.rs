//! Conversion traits for Request types
//!
//! Provides From and TryFrom implementations for converting between internal Request
//! types and http crate types.

use std::convert::TryFrom;

use http::{Request as HttpRequest, Version};

use super::request::Request;
use crate::hyper::async_impl::body::Body;

impl From<Request> for HttpRequest<Option<Body>> {
    fn from(req: Request) -> HttpRequest<Option<Body>> {
        let Request {
            method,
            url,
            headers,
            body,
            version,
            mut extensions,
        } = req;

        let mut http_req = HttpRequest::builder()
            .method(method)
            .uri(url.as_str())
            .version(version);

        let headers_ref = http_req.headers_mut().expect("builder has headers");
        headers_ref.extend(headers);

        *http_req.extensions_mut() = extensions.take();

        http_req.body(body).expect("builder has body")
    }
}

impl TryFrom<HttpRequest<Body>> for Request {
    type Error = crate::HttpError;

    fn try_from(req: HttpRequest<Body>) -> Result<Self, Self::Error> {
        let (parts, body) = req.into_parts();

        let method = parts.method;
        let version = parts.version;

        let url = match parts.uri.to_string().parse() {
            Ok(url) => url,
            Err(_) => return Err(crate::HttpError::builder("Invalid URL".to_string())),
        };

        Ok(Request {
            method,
            url,
            headers: parts.headers,
            body: Some(body),
            version,
            extensions: parts.extensions.into(),
        })
    }
}

impl TryFrom<HttpRequest<Option<Body>>> for Request {
    type Error = crate::HttpError;

    fn try_from(req: HttpRequest<Option<Body>>) -> Result<Self, Self::Error> {
        let (parts, body) = req.into_parts();

        let method = parts.method;
        let version = parts.version;

        let url = match parts.uri.to_string().parse() {
            Ok(url) => url,
            Err(_) => return Err(crate::HttpError::builder("Invalid URL".to_string())),
        };

        Ok(Request {
            method,
            url,
            headers: parts.headers,
            body,
            version,
            extensions: parts.extensions.into(),
        })
    }
}
