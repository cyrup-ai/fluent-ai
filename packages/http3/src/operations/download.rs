//! Download Operations Module - File downloads with progress, resume, and throttling

use http::{HeaderMap, HeaderName, HeaderValue, Method};

use crate::{
    HttpResult, client::HttpClient, operations::HttpOperation, request::HttpRequest,
    stream::DownloadStream,
};

/// Download operation with progress tracking and resume capability
pub struct DownloadOperation {
    client: HttpClient,
    url: String,
    headers: HeaderMap,
    resume_from: Option<u64>,
}

impl DownloadOperation {
    /// Create a new download operation
    #[inline(always)]
    pub fn new(client: HttpClient, url: String) -> Self {
        Self {
            client,
            url,
            headers: HeaderMap::new(),
            resume_from: None,
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

    /// Set headers from a HeaderMap
    #[inline(always)]
    pub fn headers(mut self, headers: HeaderMap) -> Self {
        self.headers = headers;
        self
    }

    /// Set the byte offset to resume the download from
    #[inline(always)]
    pub fn resume_from(mut self, offset: u64) -> Self {
        self.resume_from = Some(offset);
        self
    }

    /// Execute the download and return a stream of chunks.
    pub fn execute_download(mut self) -> DownloadStream {
        if let Some(offset) = self.resume_from {
            let range_value = format!("bytes={}-", offset);
            if let Ok(header_value) = HeaderValue::from_str(&range_value) {
                self.headers.insert(http::header::RANGE, header_value);
            }
            // Silently skip invalid range header rather than panicking
        }

        let request = HttpRequest::new(
            self.method(),
            self.url.clone(),
            Some(self.headers),
            None,
            None,
        );
        self.client.download_file(request)
    }
}

impl HttpOperation for DownloadOperation {
    type Output = DownloadStream;

    fn execute(&self) -> Self::Output {
        // Cloning self to allow the operation to be executed.
        // This is a bit of a workaround for the ownership model.
        let op = self.clone();
        op.execute_download()
    }

    fn method(&self) -> Method {
        Method::GET
    }

    fn url(&self) -> &str {
        &self.url
    }
}

// Manually implement Clone because of HttpClient
impl Clone for DownloadOperation {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            url: self.url.clone(),
            headers: self.headers.clone(),
            resume_from: self.resume_from,
        }
    }
}
