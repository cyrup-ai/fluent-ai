//! PUT HTTP Operations Module - Idempotent resource replacement with ETag support

use http::{HeaderMap, HeaderName, HeaderValue, Method};
use serde_json::Value;

use crate::operations::HttpOperation;
use crate::{HttpResult, client::HttpClient, request::HttpRequest, stream::HttpStream};

/// PUT operation implementation for idempotent resource replacement
#[derive(Clone)]
pub struct PutOperation {
    client: HttpClient,
    url: String,
    headers: HeaderMap,
    body: PutBody,
}

/// Supported PUT body types
#[derive(Clone)]
pub enum PutBody {
    /// JSON-encoded request body
    Json(Value),
    /// Binary data body
    Binary(Vec<u8>),
    /// Plain text body
    Text(String),
    /// Empty request body
    Empty,
}

impl PutOperation {
    /// Create a new PUT operation
    #[inline(always)]
    pub fn new(client: HttpClient, url: String) -> Self {
        Self {
            client,
            url,
            headers: HeaderMap::new(),
            body: PutBody::Empty,
        }
    }

    /// Add custom header
    #[inline(always)]
    pub fn header(mut self, key: &str, value: &str) -> HttpResult<Self> {
        let header_name = HeaderName::from_bytes(key.as_bytes())?;
        let header_value = HeaderValue::from_str(value)?;
        self.headers.insert(header_name, header_value);
        Ok(self)
    }

    /// Set JSON body with automatic Content-Type
    #[inline(always)]
    pub fn json(mut self, json_value: Value) -> Self {
        self.headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        self.body = PutBody::Json(json_value);
        self
    }

    /// Set binary body with custom Content-Type
    #[inline(always)]
    pub fn binary(mut self, data: Vec<u8>, content_type: &str) -> HttpResult<Self> {
        self.headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_str(content_type)?,
        );
        self.body = PutBody::Binary(data);
        Ok(self)
    }

    /// Set text body with Content-Type
    #[inline(always)]
    pub fn text(mut self, data: String, content_type: &str) -> HttpResult<Self> {
        self.headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_str(content_type)?,
        );
        self.body = PutBody::Text(data);
        Ok(self)
    }

    /// Add If-Match header for conditional replacement
    #[inline(always)]
    pub fn if_match(mut self, etag: &str) -> HttpResult<Self> {
        self.headers
            .insert(http::header::IF_MATCH, HeaderValue::from_str(etag)?);
        Ok(self)
    }
}

impl HttpOperation for PutOperation {
    type Output = HttpStream;

    fn execute(&self) -> Self::Output {
        let body_bytes = match &self.body {
            PutBody::Json(val) => match serde_json::to_vec(val) {
                Ok(bytes) => Some(bytes),
                Err(_) => Some(Vec::new()), // Fallback to empty body on serialization error
            },
            PutBody::Binary(data) => Some(data.clone()),
            PutBody::Text(text) => Some(text.clone().into_bytes()),
            PutBody::Empty => None,
        };

        let request = HttpRequest::new(
            self.method(),
            self.url.clone(),
            Some(self.headers.clone()),
            body_bytes,
            None,
        );

        self.client.execute_streaming(request)
    }

    #[inline(always)]
    fn method(&self) -> Method {
        Method::PUT
    }

    #[inline(always)]
    fn url(&self) -> &str {
        &self.url
    }
}
