//! DELETE HTTP Operations Module - Resource deletion with conditional logic

use http::{HeaderMap, HeaderName, HeaderValue, Method};

use crate::operations::HttpOperation;
use crate::{client::HttpClient, error::HttpError, request::HttpRequest, stream::HttpStream};

/// DELETE operation implementation with conditional deletion support
#[derive(Clone)]
pub struct DeleteOperation {
    client: HttpClient,
    url: String,
    headers: HeaderMap,
}

impl DeleteOperation {
    /// Create a new DELETE operation
    #[inline(always)]
    pub fn new(client: HttpClient, url: String) -> Self {
        Self {
            client,
            url,
            headers: HeaderMap::new(),
        }
    }

    /// Add custom header
    #[inline(always)]
    pub fn header(mut self, key: &str, value: &str) -> Result<Self, HttpError> {
        let header_name = HeaderName::from_bytes(key.as_bytes())?;
        let header_value = HeaderValue::from_str(value)?;
        self.headers.insert(header_name, header_value);
        Ok(self)
    }

    /// Add If-Match header for conditional deletion
    #[inline(always)]
    pub fn if_match(mut self, etag: &str) -> Result<Self, HttpError> {
        self.headers
            .insert(http::header::IF_MATCH, HeaderValue::from_str(etag)?);
        Ok(self)
    }
}

impl HttpOperation for DeleteOperation {
    type Output = HttpStream;

    fn execute(&self) -> Self::Output {
        let request = HttpRequest::new(
            self.method(),
            self.url.clone(),
            Some(self.headers.clone()),
            None,
            None,
        );
        self.client.execute_streaming(request)
    }

    #[inline(always)]
    fn method(&self) -> Method {
        Method::DELETE
    }

    #[inline(always)]
    fn url(&self) -> &str {
        &self.url
    }
}
