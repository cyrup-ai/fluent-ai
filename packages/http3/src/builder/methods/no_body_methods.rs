//! Pure AsyncStream HTTP methods - NO Futures, NO middleware
//! ALL methods return AsyncStream<T, CAP> directly from fluent_ai_async

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit};
use http::Method;

use crate::builder::core::{BodyNotSet, Http3Builder};
use crate::{DownloadBuilder, HttpChunk};

/// Trait for HTTP methods that don't require a body - pure AsyncStream
pub trait NoBodyMethods {
    /// Execute a GET request - returns AsyncStream directly
    fn get(self, url: &str) -> AsyncStream<HttpChunk, 1024>;

    /// Execute a DELETE request - returns AsyncStream directly
    fn delete(self, url: &str) -> AsyncStream<HttpChunk, 1024>;

    /// Create a download builder for file downloads
    fn download_file(self, url: &str) -> DownloadBuilder;
}

/// GET method implementation
pub struct GetMethod;

/// DELETE method implementation
pub struct DeleteMethod;

/// Download method implementation
pub struct DownloadMethod;

impl NoBodyMethods for Http3Builder<BodyNotSet> {
    #[inline]
    fn get(self, url: &str) -> AsyncStream<HttpChunk, 1024> {
        self.execute_with_method(Method::GET, url)
    }

    #[inline]
    fn delete(self, url: &str) -> AsyncStream<HttpChunk, 1024> {
        self.execute_with_method(Method::DELETE, url)
    }

    #[inline]
    fn download_file(self, url: &str) -> DownloadBuilder {
        DownloadBuilder::new(self.client, url)
    }
}

impl Http3Builder<BodyNotSet> {
    /// Execute a GET request - returns AsyncStream directly
    ///
    /// # Arguments
    /// * `url` - The URL to send the GET request to
    ///
    /// # Returns
    /// `AsyncStream<HttpChunk, 1024>` for pure streaming
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let stream = Http3Builder::json()
    ///     .get("https://api.example.com/users");
    /// ```
    #[must_use]
    #[inline]
    pub fn get(self, url: &str) -> AsyncStream<HttpChunk, 1024> {
        self.execute_with_method(Method::GET, url)
    }

    /// Execute a DELETE request - returns AsyncStream directly
    ///
    /// # Arguments
    /// * `url` - The URL to send the DELETE request to
    ///
    /// # Returns
    /// `AsyncStream<HttpChunk, 1024>` for pure streaming
    #[must_use]
    #[inline]
    pub fn delete(self, url: &str) -> AsyncStream<HttpChunk, 1024> {
        self.execute_with_method(Method::DELETE, url)
    }

    /// Create a download builder for file downloads
    ///
    /// # Arguments
    /// * `url` - The URL to download from
    ///
    /// # Returns
    /// `DownloadBuilder` for configuring and executing the download
    #[must_use]
    #[inline]
    pub fn download_file(self, url: &str) -> DownloadBuilder {
        if self.debug_enabled {
            log::debug!("HTTP3 Builder: Download {}", url);
        }

        DownloadBuilder::new(self.client, url)
    }

    /// Internal method to execute request with specified HTTP method - pure AsyncStream
    #[inline]
    fn execute_with_method(self, method: Method, url: &str) -> AsyncStream<HttpChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Parse URL and validate
            let uri = match url.parse() {
                Ok(uri) => uri,
                Err(e) => {
                    emit!(
                        sender,
                        HttpChunk::bad_chunk(format!("Invalid URL {}: {}", url, e))
                    );
                    return;
                }
            };

            // Create request with method and URL
            let mut request = self.request;
            *request.method_mut() = method;
            *request.uri_mut() = uri;

            if self.debug_enabled {
                log::debug!("HTTP3 Builder: {} {}", method, url);
            }

            // Execute using direct H2/H3/Quiche protocols - NO middleware
            let protocol_stream = self.client.execute_direct_streaming(request);

            // Forward all chunks from protocol stream
            for chunk in protocol_stream {
                emit!(sender, chunk);
            }
        })
    }
}
