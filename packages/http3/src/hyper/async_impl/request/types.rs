use std::convert::TryFrom;
use std::fmt;
use std::time::Duration;

use super::super::body::Body;
use super::super::client::Client;
use super::super::response::Response;
use crate::hyper::config::{RequestConfig, RequestTimeout};
use crate::header::{HeaderMap, HeaderName, HeaderValue};
use crate::{Method, Url};
use http::{request::Parts, Extensions, Request as HttpRequest, Version};

/// A request which can be executed with `Client::execute()`.
pub struct Request {
    method: Method,
    url: Url,
    headers: HeaderMap,
    body: Option<Body>,
    version: Version,
    extensions: Extensions,
}

/// A builder to construct the properties of a `Request`.
///
/// To construct a `RequestBuilder`, refer to the `Client` documentation.
#[must_use = "RequestBuilder does nothing until you 'send' it"]
pub struct RequestBuilder {
    pub(super) client: Client,
    pub(super) request: crate::Result<Request>,
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
        self.extensions()
            .get::<RequestTimeout>()
            .map(|rt| &rt.0)
    }

    /// Get a mutable reference to the timeout for this request.
    pub fn timeout_mut(&mut self) -> &mut Option<Duration> {
        &mut self.extensions_mut()
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

impl RequestBuilder {
    pub(super) fn new(client: Client, request: crate::Result<Request>) -> RequestBuilder {
        RequestBuilder { client, request }
    }

    /// Add a `Header` to this Request.
    pub fn header<K, V>(mut self, key: K, value: V) -> RequestBuilder
    where
        HeaderName: TryFrom<K>,
        <HeaderName as TryFrom<K>>::Error: Into<http::Error>,
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.header_sensitive(key, value, false)
    }

    /// Add a `Header` to this Request with ability to define if `header_value` is sensitive.
    pub(super) fn header_sensitive<K, V>(mut self, key: K, value: V, sensitive: bool) -> RequestBuilder
    where
        HeaderName: TryFrom<K>,
        <HeaderName as TryFrom<K>>::Error: Into<http::Error>,
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        let mut error = None;
        if let Ok(ref mut req) = self.request {
            match <HeaderName as TryFrom<K>>::try_from(key) {
                Ok(key) => match <HeaderValue as TryFrom<V>>::try_from(value) {
                    Ok(mut value) => {
                        // We want to potentially make an non-sensitive header
                        // to be sensitive, not the reverse. So, don't turn off
                        // a previously sensitive header.
                        if sensitive {
                            value.set_sensitive(true);
                        }
                        req.headers_mut().append(key, value);
                    }
                    Err(_) => error = Some(crate::HttpError::builder("Header value error".to_string())),
                },
                Err(_) => error = Some(crate::HttpError::builder("Header name error".to_string())),
            };
        }
        if let Some(err) = error {
            self.request = Err(err);
        }
        self
    }
}

impl fmt::Debug for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt_request_fields(&mut f.debug_struct("Request"), self).finish()
    }
}

impl fmt::Debug for RequestBuilder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("RequestBuilder");
        match self.request {
            Ok(ref req) => fmt_request_fields(&mut builder, req).finish(),
            Err(ref err) => builder.field("error", err).finish(),
        }
    }
}

pub(super) fn fmt_request_fields<'a, 'b>(
    f: &'a mut fmt::DebugStruct<'a, 'b>,
    req: &Request,
) -> &'a mut fmt::DebugStruct<'a, 'b> {
    f.field("method", &req.method)
        .field("url", &req.url)
        .field("headers", &req.headers)
}

impl From<Request> for HttpRequest<Option<Body>> {
    fn from(req: Request) -> HttpRequest<Option<Body>> {
        let Request {
            method,
            url,
            headers,
            body,
            version,
            extensions,
        } = req;

        let mut http_req = HttpRequest::builder()
            .method(method)
            .uri(url.as_str())
            .version(version);

        let headers_map = http_req.headers_mut().unwrap();
        for (key, value) in headers.iter() {
            headers_map.append(key, value.clone());
        }

        let mut http_req = http_req.body(body).unwrap();
        *http_req.extensions_mut() = extensions;
        http_req
    }
}

impl TryFrom<HttpRequest<Body>> for Request {
    type Error = crate::HttpError;

    fn try_from(req: HttpRequest<Body>) -> Result<Self, Self::Error> {
        let (parts, body) = req.into_parts();
        let Parts {
            method,
            uri,
            version,
            headers,
            extensions,
            ..
        } = parts;

        let url = Url::parse(&uri.to_string())
            .map_err(|e| crate::HttpError::builder(format!("Invalid URL: {}", e)))?;

        Ok(Request {
            method,
            url,
            headers: HeaderMap::from(headers),
            body: Some(body),
            version,
            extensions,
        })
    }
}