//! Core Request type and methods
//!
//! Provides the fundamental Request struct with accessor methods and core functionality.

use std::time::Duration;

use http::{Extensions, Version};

use super::super::super::body::Body;
use http::header::HeaderMap;
use crate::hyper::config::RequestTimeout;
use crate::{Method, Url};

/// A request which can be executed with `Client::execute()`.
pub struct Request {
    method: Method,
    url: Url,
    headers: HeaderMap,
    body: Option<Body>,
    version: Version,
    extensions: Extensions,
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
            version: Version::default(),
            extensions: Extensions::new(),
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

    /// Get the http version.
    #[inline]
    pub fn version(&self) -> Version {
        self.version
    }

    /// Get a mutable reference to the http version.
    #[inline]
    pub fn version_mut(&mut self) -> &mut Version {
        &mut self.version
    }

    /// Get the request extensions.
    #[inline]
    pub fn extensions(&self) -> &Extensions {
        &self.extensions
    }

    /// Get a mutable reference to the request extensions.
    #[inline]
    pub fn extensions_mut(&mut self) -> &mut Extensions {
        &mut self.extensions
    }

    /// Get the timeout for this request.
    pub fn timeout(&self) -> Option<&Duration> {
        self.extensions().get::<RequestTimeout>().map(|rt| &rt.0)
    }

    /// Get a mutable reference to the timeout for this request.
    pub fn timeout_mut(&mut self) -> &mut Option<Duration> {
        &mut self
            .extensions_mut()
            .get_mut::<RequestTimeout>()
            .unwrap_or(&mut RequestTimeout(None))
            .0
    }

    /// Attempt to clone the Request.
    ///
    /// `None` is returned if the Request can not be cloned,
    /// i.e. if the request body is a stream.
    pub fn try_clone(&self) -> Option<Request> {
        let body = match self.body.as_ref() {
            Some(ref body) => Some(body.try_clone()?),
            None => None,
        };
        Some(Request {
            method: self.method.clone(),
            url: self.url.clone(),
            headers: self.headers.clone(),
            body,
            version: self.version,
            extensions: self.extensions.clone(),
        })
    }
}
