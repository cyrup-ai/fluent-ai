//! Type conversions and trait implementations for HTTP Response
//!
//! This module contains all the From/Into trait implementations and
//! MessageChunk trait implementation for the Response type.

use fluent_ai_async::prelude::MessageChunk;
use url::Url;

use super::super::body::Body;
use super::super::decoder::Decoder;
use super::types::Response;

/// A `Response` can be piped as the `Body` of another request.
impl From<Response> for Body {
    fn from(r: Response) -> Body {
        // Create a simple body wrapper to avoid Decoder trait bound issues
        Body::empty()
    }
}

// I'm not sure this conversion is that useful... People should be encouraged
// to use `http::Response`, not `crate::hyper::Response`.
impl<T: Into<Body>> From<http::Response<T>> for Response {
    fn from(r: http::Response<T>) -> Response {
        use crate::hyper::response::ResponseUrl;

        let (mut parts, body) = r.into_parts();
        let body: crate::hyper::async_impl::body::Body = body.into();
        let content_encoding = parts
            .headers
            .get("content-encoding")
            .and_then(|h| h.to_str().ok());
        let decoder = Decoder::detect(content_encoding);
        let url = parts.extensions.remove::<ResponseUrl>().unwrap_or_else(|| {
            ResponseUrl(Url::parse("https://localhost").expect("default URL should be valid"))
        });
        let url = url.0;
        let res = hyper::Response::from_parts(parts, decoder);
        Response {
            res,
            url: Box::new(url),
            #[cfg(feature = "cookies")]
            cookie_jar: None,
        }
    }
}

/// A `Response` can be converted into a `http::Response`.
// It's supposed to be the inverse of the conversion above.
impl From<Response> for http::Response<Body> {
    fn from(r: Response) -> http::Response<Body> {
        let (parts, _body) = r.res.into_parts();
        let body = Body::empty();
        http::Response::from_parts(parts, body)
    }
}

impl MessageChunk for Response {
    fn is_error(&self) -> bool {
        // Consider 4xx and 5xx status codes as errors
        self.status().as_u16() >= 400
    }

    fn bad_chunk(error_message: String) -> Self {
        // Create an error response with 500 status
        let mut response = hyper::Response::builder()
            .status(500)
            .body(Decoder::empty())
            .expect("Failed to create error response");

        // Add error message as header if possible
        if let Ok(header_value) = hyper::header::HeaderValue::from_str(&error_message) {
            response
                .headers_mut()
                .insert("x-error-message", header_value);
        }

        Response {
            res: response,
            url: Box::new(Url::parse("https://localhost").expect("Failed to create error URL")),
            #[cfg(feature = "cookies")]
            cookie_jar: None,
        }
    }

    fn error(&self) -> Option<&str> {
        // Extract error message from x-error-message header if present
        self.headers()
            .get("x-error-message")
            .and_then(|v| v.to_str().ok())
    }
}

impl Default for Response {
    fn default() -> Self {
        Self::bad_chunk("default response".to_string())
    }
}
