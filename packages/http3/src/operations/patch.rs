//! PATCH HTTP Operations Module - JSON Patch (RFC 6902) and JSON Merge Patch (RFC 7396)

use http::{HeaderMap, HeaderName, HeaderValue, Method};
use serde_json::Value;

use crate::operations::HttpOperation;
use crate::{HttpResult, client::HttpClient, request::HttpRequest, stream::HttpStream};

/// PATCH operation implementation supporting multiple patch formats
#[derive(Clone)]
pub struct PatchOperation {
    client: HttpClient,
    url: String,
    headers: HeaderMap,
    body: PatchBody,
}

/// Supported PATCH types
#[derive(Clone)]
pub enum PatchBody {
    JsonPatch(Value),
    JsonMergePatch(Value),
}

impl PatchOperation {
    /// Create a new PATCH operation
    #[inline(always)]
    pub fn new(client: HttpClient, url: String) -> Self {
        Self {
            client,
            url,
            headers: HeaderMap::new(),
            body: PatchBody::JsonMergePatch(Value::Null),
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

    /// Set JSON Patch operations (RFC 6902)
    #[inline(always)]
    pub fn json_patch(mut self, patch: Value) -> Self {
        self.headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json-patch+json"),
        );
        self.body = PatchBody::JsonPatch(patch);
        self
    }

    /// Set JSON Merge Patch (RFC 7396)
    #[inline(always)]
    pub fn merge_patch(mut self, patch: Value) -> Self {
        self.headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/merge-patch+json"),
        );
        self.body = PatchBody::JsonMergePatch(patch);
        self
    }

    /// Add If-Match header for conditional patching
    #[inline(always)]
    pub fn if_match(mut self, etag: &str) -> HttpResult<Self> {
        self.headers
            .insert(http::header::IF_MATCH, HeaderValue::from_str(etag)?);
        Ok(self)
    }
}

impl HttpOperation for PatchOperation {
    type Output = HttpStream;

    fn execute(&self) -> Self::Output {
        let body_bytes = match &self.body {
            PatchBody::JsonPatch(val) => serde_json::to_vec(val).unwrap(), // Should not fail
            PatchBody::JsonMergePatch(val) => serde_json::to_vec(val).unwrap(), // Should not fail
        };

        let request = HttpRequest::new(
            self.method(),
            self.url.clone(),
            Some(self.headers.clone()),
            Some(body_bytes),
            None,
        );

        self.client.execute_streaming(request)
    }

    #[inline(always)]
    fn method(&self) -> Method {
        Method::PATCH
    }

    #[inline(always)]
    fn url(&self) -> &str {
        &self.url
    }
}
