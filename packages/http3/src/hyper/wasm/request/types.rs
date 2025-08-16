use std::time::Duration;

use bytes::Bytes;
use http::Method;
use url::Url;
#[cfg(target_arch = "wasm32")]
use web_sys::{Headers, Request as WebRequest, RequestCredentials};

use super::Body;
use http::header::HeaderMap;

/// A request which can be executed with `Client::execute()`.
pub struct Request {
    pub(super) method: Method,
    pub(super) url: Url,
    pub(super) headers: HeaderMap,
    pub(super) body: Option<Body>,
    pub(super) timeout: Option<Duration>,
    pub(super) cors: bool,
    #[cfg(target_arch = "wasm32")]
    pub(super) credentials: Option<RequestCredentials>,
    #[cfg(target_arch = "wasm32")]
    pub(super) cache: Option<RequestCache>,
}

impl Request {
    /// Constructs a new request.
    #[inline]
    pub fn new(method: Method, url: Url) -> Self {
        Request {
            method,
            url,
            headers: HeaderMap::new(),
            body: None,
            timeout: None,
            cors: true,
            credentials: None,
            cache: None,
        }
    }

    /// Get the method.
    #[inline]
    pub fn method(&self) -> &Method {
        &self.method
    }

    /// Get a mutable reference to the method.
    #[inline]
    pub fn method_mut(&mut self) -> &mut Method {
        &mut self.method
    }

    /// Get the url.
    #[inline]
    pub fn url(&self) -> &Url {
        &self.url
    }

    /// Get a mutable reference to the url.
    #[inline]
    pub fn url_mut(&mut self) -> &mut Url {
        &mut self.url
    }

    /// Get the headers.
    #[inline]
    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    /// Get a mutable reference to the headers.
    #[inline]
    pub fn headers_mut(&mut self) -> &mut HeaderMap {
        &mut self.headers
    }

    /// Get the body.
    #[inline]
    pub fn body(&self) -> Option<&Body> {
        self.body.as_ref()
    }

    /// Get a mutable reference to the body.
    #[inline]
    pub fn body_mut(&mut self) -> &mut Option<Body> {
        &mut self.body
    }

    /// Get the timeout.
    #[inline]
    pub fn timeout(&self) -> Option<&Duration> {
        self.timeout.as_ref()
    }

    /// Get a mutable reference to the timeout.
    #[inline]
    pub fn timeout_mut(&mut self) -> &mut Option<Duration> {
        &mut self.timeout
    }

    /// Attempts to clone the `Request`.
    ///
    /// None is returned if a body is which can not be cloned.
    pub fn try_clone(&self) -> Option<Request> {
        let body = match self.body.as_ref() {
            Some(body) => Some(body.try_clone()?),
            None => None,
        };

        Some(Self {
            method: self.method.clone(),
            url: self.url.clone(),
            headers: self.headers.clone(),
            body,
            timeout: self.timeout,
            cors: self.cors,
            credentials: self.credentials,
            cache: self.cache,
        })
    }
}
