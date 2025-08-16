//! Pre-built response constructors
//!
//! This module contains static factory methods for creating common response types.

use http::{HeaderMap, HeaderValue, StatusCode, Version};
use url::Url;

use super::types::Response;

impl Response {
    /// Create a new 200 OK response
    pub fn ok() -> ResponseBuilder {
        ResponseBuilder::new(StatusCode::OK)
    }

    /// Create a new 201 Created response
    pub fn created() -> ResponseBuilder {
        ResponseBuilder::new(StatusCode::CREATED)
    }

    /// Create a new 204 No Content response
    pub fn no_content() -> ResponseBuilder {
        ResponseBuilder::new(StatusCode::NO_CONTENT)
    }

    /// Create a new 400 Bad Request response
    pub fn bad_request() -> ResponseBuilder {
        ResponseBuilder::new(StatusCode::BAD_REQUEST)
    }

    /// Create a new 401 Unauthorized response
    pub fn unauthorized() -> ResponseBuilder {
        ResponseBuilder::new(StatusCode::UNAUTHORIZED)
    }

    /// Create a new 403 Forbidden response
    pub fn forbidden() -> ResponseBuilder {
        ResponseBuilder::new(StatusCode::FORBIDDEN)
    }

    /// Create a new 404 Not Found response
    pub fn not_found() -> ResponseBuilder {
        ResponseBuilder::new(StatusCode::NOT_FOUND)
    }

    /// Create a new 500 Internal Server Error response
    pub fn internal_server_error() -> ResponseBuilder {
        ResponseBuilder::new(StatusCode::INTERNAL_SERVER_ERROR)
    }

    /// Create a new 502 Bad Gateway response
    pub fn bad_gateway() -> ResponseBuilder {
        ResponseBuilder::new(StatusCode::BAD_GATEWAY)
    }

    /// Create a new 503 Service Unavailable response
    pub fn service_unavailable() -> ResponseBuilder {
        ResponseBuilder::new(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Builder for constructing Response instances
pub struct ResponseBuilder {
    status: StatusCode,
    headers: HeaderMap,
    version: Version,
}

impl ResponseBuilder {
    pub fn new(status: StatusCode) -> Self {
        Self {
            status,
            headers: HeaderMap::new(),
            version: Version::HTTP_11,
        }
    }

    pub fn header<K, V>(mut self, key: K, value: V) -> Self
    where
        K: http::header::IntoHeaderName,
        V: TryInto<HeaderValue>,
        V::Error: Into<http::Error>,
    {
        self.headers.insert(key, value.try_into().unwrap());
        self
    }

    pub fn version(mut self, version: Version) -> Self {
        self.version = version;
        self
    }

    pub fn body<T>(self, body: T) -> hyper::Response<T> {
        let mut builder = hyper::Response::builder()
            .status(self.status)
            .version(self.version);

        for (key, value) in self.headers.iter() {
            builder = builder.header(key, value);
        }

        builder.body(body).unwrap()
    }
}
