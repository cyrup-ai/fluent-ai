//! Pure AsyncStream HTTP methods - NO Futures, NO middleware
//! ALL methods return AsyncStream<T, CAP> directly from fluent_ai_async

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit};
use http::Method;
use url::Url;

use crate::prelude::*;
use crate::builder::fluent::DownloadBuilder;
use crate::builder::core::{Http3Builder, BodyNotSet};
/// Trait for HTTP methods that don't require a body - pure AsyncStream
pub trait NoBodyMethods {
    /// Execute a GET request - returns AsyncStream directly
    fn get<T>(self, url: &str) -> AsyncStream<T, 1024>
    where
        T: serde::de::DeserializeOwned + MessageChunk + Default + Send + 'static;

    /// Execute a DELETE request - returns AsyncStream directly
    fn delete<T>(self, url: &str) -> AsyncStream<T, 1024>
    where
        T: serde::de::DeserializeOwned + MessageChunk + Default + Send + 'static;

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
    fn get<T>(self, url: &str) -> AsyncStream<T, 1024>
    where
        T: serde::de::DeserializeOwned + MessageChunk + Default + Send + 'static,
    {
        let request = self.request
            .with_method(Method::GET)
            .with_url(Url::parse(url).unwrap_or_else(|_| {
                log::error!("Invalid URL: {}", url);
                Url::parse("http://invalid").unwrap()
            }));
        
        // Execute request using canonical method and apply real-time deserialization
        let http_response = self.client.execute(request);
        
        // Transform HttpBodyChunk stream to typed stream with real-time deserialization
        AsyncStream::with_channel(move |sender| {
            let body_stream = http_response.into_body_stream();
            for body_chunk in body_stream {
                // Real-time JSON deserialization of each chunk
                let deserialized_obj = match serde_json::from_slice::<T>(&body_chunk.data) {
                    Ok(obj) => obj,
                    Err(_) => T::bad_chunk("JSON deserialization failed".to_string())
                };
                
                fluent_ai_async::emit!(sender, deserialized_obj);
            }
        })
    }

    #[inline]
    fn delete<T>(self, url: &str) -> AsyncStream<T, 1024>
    where
        T: serde::de::DeserializeOwned + MessageChunk + Default + Send + 'static,
    {
        let request = self.request
            .with_method(Method::DELETE)
            .with_url(Url::parse(url).unwrap_or_else(|_| {
                log::error!("Invalid URL: {}", url);
                Url::parse("http://invalid").unwrap()
            }));
        
        // Execute request using canonical method and apply real-time deserialization
        let http_response = self.client.execute(request);
        
        // Transform HttpBodyChunk stream to typed stream with real-time deserialization
        AsyncStream::with_channel(move |sender| {
            let body_stream = http_response.into_body_stream();
            for body_chunk in body_stream {
                // Real-time JSON deserialization of each chunk
                let deserialized_obj = match serde_json::from_slice::<T>(&body_chunk.data) {
                    Ok(obj) => obj,
                    Err(_) => T::bad_chunk("JSON deserialization failed".to_string())
                };
                
                fluent_ai_async::emit!(sender, deserialized_obj);
            }
        })
    }

    #[inline]
    fn download_file(self, url: &str) -> DownloadBuilder {
        // Execute request and convert to download stream
        let request = self.request
            .with_method(Method::GET)
            .with_url(Url::parse(url).unwrap());
        let response = self.client.execute(request);
        
        // Convert response to download stream
        let download_stream = AsyncStream::with_channel(move |sender| {
            let body_stream = response.into_body_stream();
            let mut downloaded = 0u64;
            
            for chunk in body_stream {
                downloaded += chunk.data.len() as u64;
                let download_chunk = crate::http::response::HttpDownloadChunk::Data {
                    chunk: chunk.data.to_vec(),
                    downloaded,
                    total_size: None,
                };
                emit!(sender, download_chunk);
            }
            
            emit!(sender, crate::http::response::HttpDownloadChunk::Complete);
        });
        
        DownloadBuilder::new(download_stream)
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

        // Execute request and convert to download stream
        let request = self.request
            .with_method(Method::GET)
            .with_url(Url::parse(url).unwrap());
        let response = self.client.execute(request);
        
        // Convert response to download stream
        let download_stream = AsyncStream::with_channel(move |sender| {
            let body_stream = response.into_body_stream();
            let mut downloaded = 0u64;
            
            for chunk in body_stream {
                downloaded += chunk.data.len() as u64;
                let download_chunk = crate::http::response::HttpDownloadChunk::Data {
                    chunk: chunk.data.to_vec(),
                    downloaded,
                    total_size: None,
                };
                emit!(sender, download_chunk);
            }
            
            emit!(sender, crate::http::response::HttpDownloadChunk::Complete);
        });
        
        DownloadBuilder::new(download_stream)
    }

    /// Internal method to execute request with specified HTTP method - pure AsyncStream
    #[inline]
    fn execute_with_method(self, method: Method, url: &str) -> AsyncStream<HttpChunk, 1024> {
        let url_owned = url.to_string();
        AsyncStream::with_channel(move |sender| {
            // Parse URL and validate
            let uri = match url_owned.parse() {
                Ok(uri) => uri,
                Err(e) => {
                    emit!(
                        sender,
                        HttpChunk::bad_chunk(format!("Invalid URL {}: {}", url_owned, e))
                    );
                    return;
                }
            };

            // Create request with method and URL
            let method_str = method.to_string();
            let mut request = self.request;
            request = request.with_method(method);
            request = request.with_url(uri);

            if self.debug_enabled {
                log::debug!("HTTP3 Builder: {} {}", method_str, url_owned);
            }

            // Execute using canonical method and extract chunks from HttpResponse
            let http_response = self.client.execute(request);

            // Forward all chunks from HttpResponse body stream
            let body_stream = http_response.into_body_stream();
            for body_chunk in body_stream {
                let http_chunk = HttpChunk::from(body_chunk);
                emit!(sender, http_chunk);
            }
        })
    }
}
