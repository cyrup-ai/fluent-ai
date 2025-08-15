//! PATCH HTTP Operations Module - JSON Patch (RFC 6902) and JSON Merge Patch (RFC 7396)

use http::{HeaderMap, HeaderName, HeaderValue, Method};
use serde_json::Value;

use crate::operations::HttpOperation;
use crate::{client::HttpClient, error::HttpError, request::HttpRequest, stream::HttpStream};

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
    /// JSON Patch (RFC 6902) - Array of patch operations
    JsonPatch(Value),
    /// JSON Merge Patch (RFC 7396) - Object representing the patch
    JsonMergePatch(Value),
}

impl PatchOperation {
    /// Create a new PATCH operation
    ///
    /// # Arguments
    /// * `client` - The HTTP client to use for the request
    /// * `url` - The URL to send the PATCH request to
    ///
    /// # Returns
    /// A new `PatchOperation` instance with JSON Merge Patch as the default format
    #[must_use]
    pub fn new(client: HttpClient, url: String) -> Self {
        Self {
            client,
            url,
            headers: HeaderMap::new(),
            body: PatchBody::JsonMergePatch(Value::Null),
        }
    }

    /// Add a custom header
    ///
    /// # Arguments
    /// * `key` - The header name
    /// * `value` - The header value
    ///
    /// # Returns
    /// `Result<Self, HttpError>` for method chaining
    #[must_use]
    pub fn header(mut self, key: &str, value: &str) -> Result<Self, HttpError> {
        let header_name = HeaderName::from_bytes(key.as_bytes())?;
        let header_value = HeaderValue::from_str(value)?;
        self.headers.insert(header_name, header_value);
        Ok(self)
    }

    /// Set JSON Patch operations (RFC 6902)
    ///
    /// # Arguments
    /// * `patch` - The JSON patch operations to apply
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn json_patch(mut self, patch: Value) -> Self {
        self.headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json-patch+json"),
        );
        self.body = PatchBody::JsonPatch(patch);
        self
    }

    /// Set JSON Merge Patch (RFC 7396)
    ///
    /// # Arguments
    /// * `patch` - The JSON merge patch to apply
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn merge_patch(mut self, patch: Value) -> Self {
        self.headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/merge-patch+json"),
        );
        self.body = PatchBody::JsonMergePatch(patch);
        self
    }

    /// Add If-Match header for conditional patching
    ///
    /// # Arguments
    /// * `etag` - The entity tag to use for conditional requests
    ///
    /// # Returns
    /// `Result<Self, HttpError>` for method chaining
    #[must_use]
    pub fn if_match(mut self, etag: &str) -> Result<Self, HttpError> {
        self.headers
            .insert(http::header::IF_MATCH, HeaderValue::from_str(etag)?);
        Ok(self)
    }
}

impl HttpOperation for PatchOperation {
    type Output = HttpStream;

    fn execute(&self) -> Self::Output {
        let body_bytes = match &self.body {
            PatchBody::JsonPatch(val) | PatchBody::JsonMergePatch(val) => {
                serde_json::to_vec(val).unwrap_or_default()
            }
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

    fn method(&self) -> Method {
        Method::PATCH
    }

    fn url(&self) -> &str {
        &self.url
    }
}
