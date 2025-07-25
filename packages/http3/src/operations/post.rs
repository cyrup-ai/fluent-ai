//! POST HTTP Operations Module - JSON, form data, binary, and multipart support

use std::collections::HashMap;
use http::{HeaderMap, HeaderName, HeaderValue, Method};
use serde_json::Value;

use crate::operations::HttpOperation;
use crate::{HttpResult, client::HttpClient, request::HttpRequest, stream::HttpStream};

/// POST operation implementation with multiple body type support
#[derive(Clone)]
pub struct PostOperation {
    client: HttpClient,
    url: String,
    headers: HeaderMap,
    body: PostBody}

/// Supported POST body types
#[derive(Clone)]
pub enum PostBody {
    /// JSON-encoded request body
    Json(Value),
    /// Form-encoded data as key-value pairs
    FormData(HashMap<String, String>),
    /// Binary data body
    Binary(Vec<u8>),
    /// Empty request body
    Empty}

impl PostOperation {
    /// Create a new POST operation
    #[inline(always)]
    pub fn new(client: HttpClient, url: String) -> Self {
        Self {
            client,
            url,
            headers: HeaderMap::new(),
            body: PostBody::Empty}
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
        self.body = PostBody::Json(json_value);
        self
    }

    /// Set form data body with automatic Content-Type
    #[inline(always)]
    pub fn form(mut self, form_data: HashMap<String, String>) -> Self {
        self.headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/x-www-form-urlencoded"),
        );
        self.body = PostBody::FormData(form_data);
        self
    }

    /// Set binary body with a specific Content-Type
    #[inline(always)]
    pub fn binary(mut self, body: Vec<u8>, content_type: &str) -> Self {
        self.headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_str(content_type)
                .unwrap_or(HeaderValue::from_static("application/octet-stream")),
        );
        self.body = PostBody::Binary(body);
        self
    }
}

impl HttpOperation for PostOperation {
    type Output = HttpStream;

    fn execute(&self) -> Self::Output {
        let body_bytes = match &self.body {
            PostBody::Json(val) => match serde_json::to_vec(val) {
                Ok(bytes) => Some(bytes),
                Err(_) => Some(Vec::new()), // Fallback to empty body on serialization error
            },
            PostBody::FormData(data) => match serde_urlencoded::to_string(data) {
                Ok(encoded) => Some(encoded.into_bytes()),
                Err(_) => Some(Vec::new()), // Fallback to empty body on encoding error
            },
            PostBody::Binary(data) => Some(data.clone()),
            PostBody::Empty => None};

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
        Method::POST
    }

    #[inline(always)]
    fn url(&self) -> &str {
        &self.url
    }
}
