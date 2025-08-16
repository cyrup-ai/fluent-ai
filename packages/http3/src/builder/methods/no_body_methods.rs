//! HTTP methods for requests without body
//!
//! This module contains terminal methods for HTTP requests that don't require a body:
//! GET, DELETE, and download_file operations for the BodyNotSet builder state.

use http::Method;

use crate::builder::core::{BodyNotSet, Http3Builder};
use crate::{DownloadBuilder, HttpStream};

impl Http3Builder<BodyNotSet> {
    /// Execute a GET request
    ///
    /// # Arguments
    /// * `url` - The URL to send the GET request to
    ///
    /// # Returns
    /// `HttpStream` for streaming the response
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .get("https://api.example.com/users");
    /// ```
    #[must_use]
    pub fn get(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::GET)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: GET {url}");
        }

        self.client.execute_streaming(self.request)
    }

    /// Execute a DELETE request
    ///
    /// # Arguments
    /// * `url` - The URL to send the DELETE request to
    ///
    /// # Returns
    /// `HttpStream` for streaming the response
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .delete("https://api.example.com/users/123");
    /// ```
    #[must_use]
    pub fn delete(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::DELETE)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: DELETE {url}");
        }

        self.client.execute_streaming(self.request)
    }

    /// Initiate a file download
    ///
    /// Creates a specialized download stream with progress tracking and
    /// file writing capabilities.
    ///
    /// # Arguments
    /// * `url` - The URL to download from
    ///
    /// # Returns
    /// `DownloadBuilder` for configuring the download
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let download = Http3Builder::new(&client)
    ///     .download_file("https://example.com/large-file.zip")
    ///     .destination("/tmp/downloaded-file.zip")
    ///     .start();
    /// ```
    #[must_use]
    pub fn download_file(mut self, url: &str) -> DownloadBuilder {
        self.request = self
            .request
            .set_method(Method::GET)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: DOWNLOAD {url}");
        }

        let stream = self.client.download_file(self.request);
        DownloadBuilder::new(stream)
    }
}
