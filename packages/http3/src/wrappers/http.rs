//! HTTP protocol wrappers for implementing MessageChunk trait
//! Includes HTTP responses, frames, bodies, and protocol-specific types

use std::ops::{Deref, DerefMut};

use bytes::Bytes;
use fluent_ai_async::prelude::MessageChunk;
use http::Response;
use http_body::Frame;

/// Wrapper for HTTP headers to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct HeaderWrapper(pub http::HeaderMap);

impl MessageChunk for HeaderWrapper {
    fn bad_chunk(_error: String) -> Self {
        Self(http::HeaderMap::new())
    }

    fn error(&self) -> Option<&str> {
        if self.0.is_empty() {
            Some("Empty header map")
        } else {
            None
        }
    }
}

impl From<http::HeaderMap> for HeaderWrapper {
    fn from(headers: http::HeaderMap) -> Self {
        Self(headers)
    }
}

impl From<HeaderWrapper> for http::HeaderMap {
    fn from(wrapper: HeaderWrapper) -> Self {
        wrapper.0
    }
}

impl Deref for HeaderWrapper {
    type Target = http::HeaderMap;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for HeaderWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Wrapper for http_body::Frame<Bytes> to implement MessageChunk + Default
#[derive(Debug)]
pub struct FrameWrapper(pub Frame<Bytes>);

impl Deref for FrameWrapper {
    type Target = Frame<Bytes>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for FrameWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Frame<Bytes>> for FrameWrapper {
    fn from(frame: Frame<Bytes>) -> Self {
        Self(frame)
    }
}

impl From<FrameWrapper> for Frame<Bytes> {
    fn from(wrapper: FrameWrapper) -> Self {
        wrapper.0
    }
}

impl Default for FrameWrapper {
    fn default() -> Self {
        Self(Frame::data(Bytes::new()))
    }
}

impl MessageChunk for FrameWrapper {
    fn bad_chunk(error: String) -> Self {
        Self(Frame::data(Bytes::from(format!("ERROR: {}", error))))
    }

    fn is_error(&self) -> bool {
        if let Some(data) = self.0.data_ref() {
            data.starts_with(b"ERROR:")
        } else {
            false
        }
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            if let Some(data) = self.0.data_ref() {
                std::str::from_utf8(data).ok()
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Wrapper for BoxBody to implement MessageChunk + Default
#[derive(Debug)]
pub struct BoxBodyWrapper(
    pub http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>,
);

impl Default for BoxBodyWrapper {
    fn default() -> Self {
        use http_body_util::BodyExt;
        Self(
            http_body_util::Empty::new()
                .map_err(|never| match never {})
                .boxed(),
        )
    }
}

impl MessageChunk for BoxBodyWrapper {
    fn bad_chunk(error: String) -> Self {
        use http_body_util::BodyExt;
        let body = http_body_util::Full::new(Bytes::from(format!("ERROR: {}", error)))
            .map_err(|never| match never {})
            .boxed();
        Self(body)
    }

    fn is_error(&self) -> bool {
        false // Cannot easily inspect BoxBody contents
    }

    fn error(&self) -> Option<&str> {
        None
    }
}

/// Wrapper for http::Response to implement MessageChunk
#[derive(Debug)]
pub struct ResponseWrapper<B>(pub Response<B>);

impl<B: Default> MessageChunk for ResponseWrapper<B> {
    fn bad_chunk(error: String) -> Self {
        use http::{Response, StatusCode};
        let response = Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header("content-type", "text/plain")
            .body(B::default())
            .unwrap_or_else(|_| {
                // Use minimal safe response if even the fallback fails
                match Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(B::default())
                {
                    Ok(resp) => resp,
                    Err(_) => {
                        // Absolute fallback - create response directly without builder
                        let mut resp = Response::new(B::default());
                        *resp.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                        resp
                    }
                }
            });
        Self(response)
    }

    fn is_error(&self) -> bool {
        self.0.status().is_server_error() || self.0.status().is_client_error()
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            self.0.status().canonical_reason()
        } else {
            None
        }
    }
}

impl<B> Default for ResponseWrapper<B>
where
    B: Default,
{
    fn default() -> Self {
        Self(Response::new(B::default()))
    }
}

impl<B> From<Response<B>> for ResponseWrapper<B> {
    fn from(response: Response<B>) -> Self {
        Self(response)
    }
}

impl<B> From<ResponseWrapper<B>> for Response<B> {
    fn from(wrapper: ResponseWrapper<B>) -> Self {
        wrapper.0
    }
}

impl<B> Deref for ResponseWrapper<B> {
    type Target = Response<B>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B> DerefMut for ResponseWrapper<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
