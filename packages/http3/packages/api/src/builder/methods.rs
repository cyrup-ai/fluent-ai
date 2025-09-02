//! HTTP method implementations
//!
//! Terminal methods for executing HTTP requests (GET, POST, PUT, PATCH, DELETE)
//! with appropriate request configurations and response streaming.

use fluent_ai_async::AsyncStream;
use http::Method;
use serde::de::DeserializeOwned;
use url::Url;

use crate::builder::core::{BodyNotSet, BodySet, Http3Builder, JsonPathStreaming};
use fluent_ai_http3_client::operations::HttpOperation;

// Re-export types from client package - using direct types, no confusing aliases
pub use fluent_ai_http3_client::http::response::{HttpChunk, HttpBodyChunk};
pub use fluent_ai_http3_client::builder::fluent::DownloadBuilder;

/// SSRF Protection - Always block dangerous URLs (no config required)
fn is_dangerous_url(url_str: &str) -> bool {
    if let Ok(parsed) = url::Url::parse(url_str) {
        if let Some(host_str) = parsed.host_str() {
            // Block localhost variants
            if host_str == "localhost" || host_str == "0.0.0.0" || host_str.ends_with(".local") {
                return true;
            }
            // Block private IP ranges if it's an IP
            if let Ok(ip) = host_str.parse::<std::net::IpAddr>() {
                match ip {
                    std::net::IpAddr::V4(ipv4) => {
                        let octets = ipv4.octets();
                        // RFC1918 + loopback + metadata
                        return octets[0] == 10 
                            || (octets[0] == 172 && octets[1] >= 16 && octets[1] <= 31)
                            || (octets[0] == 192 && octets[1] == 168)
                            || octets[0] == 127
                            || (octets[0] == 169 && octets[1] == 254);
                    }
                    std::net::IpAddr::V6(ipv6) => {
                        return ipv6.is_loopback() || ipv6.segments()[0] == 0xfe80;
                    }
                }
            }
        }
    }
    false
}

// Terminal methods for BodyNotSet (no body required)
impl Http3Builder<BodyNotSet> {
    /// Execute a GET request
    ///
    /// # Arguments
    /// * `url` - The URL to send the GET request to
    ///
    /// # Returns
    /// `AsyncStream<T, 1024>` for streaming deserialized response data
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .get::<User>("https://api.example.com/users");
    /// ```
    #[must_use]
    pub fn get<T>(mut self, url: &str) -> AsyncStream<T, 1024> 
    where 
        T: serde::de::DeserializeOwned + fluent_ai_async::prelude::MessageChunk + Default + Send + 'static,
    {
        // SSRF Protection - Always block dangerous URLs (no config required)
        if is_dangerous_url(url) {
            return AsyncStream::with_channel(move |sender| {
                fluent_ai_async::emit!(sender, T::bad_chunk("URL blocked for security".to_string()));
            });
        }

        // Handle URL parsing with proper error handling
        let parsed_url = match url.parse::<Url>() {
            Ok(url) => url,
            Err(parse_error) => {
                // Convert to owned string for the closure
                let url_string = url.to_string();
                // Return error stream for invalid URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Invalid URL '{}': {}", url_string, parse_error)));
                });
            }
        };

        self.request = self
            .request
            .with_method(Method::GET)
            .with_url(parsed_url);

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: GET {url}");
        }

        // Use the correct HttpClient.execute() pattern
        let response = self.client.execute(self.request);
        
        // Transform HttpResponse to deserialized T stream
        AsyncStream::with_channel(move |sender| {
            // Convert HttpResponse to body stream and deserialize
            let body_stream = response.into_body_stream();
            let mut body_data = Vec::new();
            
            // Collect all body chunks
            let body_chunks = body_stream.collect();
            for chunk in body_chunks {
                body_data.extend_from_slice(&chunk.data);
            }
            
            // Deserialize the complete response body
            match serde_json::from_slice::<T>(&body_data) {
                Ok(deserialized) => fluent_ai_async::emit!(sender, deserialized),
                Err(e) => fluent_ai_async::emit!(sender, T::bad_chunk(format!("JSON deserialization failed: {}", e))),
            }
        })
    }

    /// Execute a DELETE request
    ///
    /// # Arguments
    /// * `url` - The URL to send the DELETE request to
    ///
    /// # Returns
    /// `AsyncStream<HttpChunk, 1024>` for streaming the response
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .delete("https://api.example.com/users/123");
    /// ```
    #[must_use]
    pub fn delete(mut self, url: &str) -> AsyncStream<HttpChunk, 1024> {
        // SSRF Protection - Always block dangerous URLs (no config required)
        if is_dangerous_url(url) {
            return AsyncStream::with_channel(move |sender| {
                fluent_ai_async::emit!(sender, HttpChunk::Error("URL blocked for security".to_string()));
            });
        }

        // Handle URL parsing with proper error handling
        let parsed_url = match url.parse::<Url>() {
            Ok(url) => url,
            Err(parse_error) => {
                // Convert to owned string for the closure
                let url_string = url.to_string();
                // Return error stream for invalid URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, HttpChunk::Error(format!("Invalid URL '{}': {}", url_string, parse_error)));
                });
            }
        };

        self.request = self
            .request
            .with_method(Method::DELETE)
            .with_url(parsed_url);

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: DELETE {url}");
        }

        // Use the correct HttpClient.execute() pattern
        let response = self.client.execute(self.request);
        
        // Transform HttpResponse stream to HttpChunk stream
        AsyncStream::with_channel(move |sender| {
            // Convert HttpResponse to body stream and emit as HttpChunks
            let body_stream = response.into_body_stream();
            let body_chunks = body_stream.collect();
            
            for body_chunk in body_chunks {
                let http_chunk = HttpChunk::Body(body_chunk.data);
                fluent_ai_async::emit!(sender, http_chunk);
            }
            
            // Emit End marker
            fluent_ai_async::emit!(sender, HttpChunk::End);
        })
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
        // SSRF Protection - Always block dangerous URLs (no config required)
        if is_dangerous_url(url) {
            let error_stream = fluent_ai_async::AsyncStream::with_channel(move |sender| {
                use fluent_ai_http3_client::http::response::HttpDownloadChunk;
                fluent_ai_async::emit!(sender, HttpDownloadChunk::Error { message: "URL blocked for security".to_string() });
            });
            return DownloadBuilder::new(error_stream);
        }

        // Handle URL parsing with proper error handling
        let parsed_url = match url.parse::<Url>() {
            Ok(url) => url,
            Err(parse_error) => {
                // Convert to owned string for the closure
                let url_string = url.to_string();
                // Return error stream for invalid URLs
                let error_stream = fluent_ai_async::AsyncStream::with_channel(move |sender| {
                    use fluent_ai_http3_client::http::response::HttpDownloadChunk;
                    fluent_ai_async::emit!(sender, HttpDownloadChunk::Error { message: format!("Invalid URL '{}': {}", url_string, parse_error) });
                });
                return DownloadBuilder::new(error_stream);
            }
        };

        self.request = self
            .request
            .with_method(Method::GET)
            .with_url(parsed_url);

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: DOWNLOAD {url}");
        }

        // Use the standard HttpClient.execute() pattern
        let response = self.client.execute(self.request);
        
        // Extract total size from response headers before converting to stream
        let total_size = response.headers()
            .get(http::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        
        // Convert HttpResponse body stream to download stream
        let download_stream = fluent_ai_async::AsyncStream::with_channel(move |sender| {
            use fluent_ai_http3_client::http::response::{HttpBodyChunk, HttpDownloadChunk};
            
            let body_stream = response.into_body_stream();
            for chunk in body_stream {
                let download_chunk = match chunk.is_final {
                    true => HttpDownloadChunk::Complete,
                    false => HttpDownloadChunk::Data {
                        chunk: chunk.data.to_vec(),
                        downloaded: chunk.offset + chunk.data.len() as u64,
                        total_size,
                    }
                };
                
                fluent_ai_async::emit!(sender, download_chunk);
                
                if chunk.is_final {
                    break;
                }
            }
        });
        
        DownloadBuilder::new(download_stream)
    }
}

// Terminal methods for BodySet (body has been set)
impl Http3Builder<BodySet> {
    /// Execute a POST request
    ///
    /// # Arguments
    /// * `url` - The URL to send the POST request to
    ///
    /// # Returns
    /// `AsyncStream<T, 1024>` for streaming deserialized response data
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::{Serialize, Deserialize};
    ///
    /// #[derive(Serialize)]
    /// struct CreateUser {
    ///     name: String,
    ///     email: String,
    /// }
    ///
    /// #[derive(Deserialize)]
    /// struct UserResponse {
    ///     id: u32,
    ///     name: String,
    /// }
    ///
    /// let user = CreateUser {
    ///     name: "John Doe".to_string(),
    ///     email: "john@example.com".to_string(),
    /// };
    ///
    /// let response = Http3Builder::json()
    ///     .body(&user)
    ///     .post::<UserResponse>("https://api.example.com/users");
    /// ```
    pub fn post<T>(mut self, url: &str) -> AsyncStream<T, 1024> 
    where 
        T: serde::de::DeserializeOwned + fluent_ai_async::prelude::MessageChunk + Default + Send + 'static,
    {
        // SSRF Protection - Always block dangerous URLs (no config required)
        if is_dangerous_url(url) {
            return AsyncStream::with_channel(move |sender| {
                fluent_ai_async::emit!(sender, T::bad_chunk("URL blocked for security".to_string()));
            });
        }

        // Handle URL parsing with proper error handling
        let parsed_url = match url.parse::<Url>() {
            Ok(url) => url,
            Err(parse_error) => {
                // Convert to owned string for the closure
                let url_string = url.to_string();
                // Return error stream for invalid URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Invalid URL '{}': {}", url_string, parse_error)));
                });
            }
        };

        self.request = self
            .request
            .with_method(Method::POST)
            .with_url(parsed_url);

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: POST {}", url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        // Use the correct HttpClient.execute() pattern
        let response = self.client.execute(self.request);
        
        // Transform HttpResponse stream to deserialized T stream
        AsyncStream::with_channel(move |sender| {
            // Convert HttpResponse to body stream and deserialize
            let body_stream = response.into_body_stream();
            let mut body_data = Vec::new();
            
            // Collect all body chunks
            let body_chunks = body_stream.collect();
            for chunk in body_chunks {
                body_data.extend_from_slice(&chunk.data);
            }
            
            // Deserialize the complete response body
            match serde_json::from_slice::<T>(&body_data) {
                Ok(deserialized) => fluent_ai_async::emit!(sender, deserialized),
                Err(e) => fluent_ai_async::emit!(sender, T::bad_chunk(format!("JSON deserialization failed: {}", e))),
            }
        })
    }

    /// Execute a PUT request
    ///
    /// # Arguments
    /// * `url` - The URL to send the PUT request to
    ///
    /// # Returns
    /// `AsyncStream<HttpChunk, 1024>` for streaming the response
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::Serialize;
    ///
    /// #[derive(Serialize)]
    /// struct UpdateUser {
    ///     name: String,
    ///     email: String,
    /// }
    ///
    /// let user = UpdateUser {
    ///     name: "Jane Doe".to_string(),
    ///     email: "jane@example.com".to_string(),
    /// };
    ///
    /// let response = Http3Builder::json()
    ///     .body(&user)
    ///     .put("https://api.example.com/users/123");
    /// ```
    pub fn put<T>(mut self, url: &str) -> AsyncStream<T, 1024> 
    where 
        T: serde::de::DeserializeOwned + fluent_ai_async::prelude::MessageChunk + Default + Send + 'static,
    {
        // SSRF Protection - Always block dangerous URLs (no config required)
        if is_dangerous_url(url) {
            return AsyncStream::with_channel(move |sender| {
                fluent_ai_async::emit!(sender, T::bad_chunk("URL blocked for security".to_string()));
            });
        }

        // Handle URL parsing with proper error handling
        let parsed_url = match url.parse::<Url>() {
            Ok(url) => url,
            Err(parse_error) => {
                // Convert to owned string for the closure
                let url_string = url.to_string();
                // Return error stream for invalid URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Invalid URL '{}': {}", url_string, parse_error)));
                });
            }
        };

        self.request = self
            .request
            .with_method(Method::PUT)
            .with_url(parsed_url);

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: PUT {}", url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        // Use the correct HttpClient.execute() pattern
        let response = self.client.execute(self.request);
        
        // Transform HttpResponse stream to deserialized T stream
        AsyncStream::with_channel(move |sender| {
            // Convert HttpResponse to body stream and deserialize
            let body_stream = response.into_body_stream();
            let mut body_data = Vec::new();
            
            // Collect all body chunks
            let body_chunks = body_stream.collect();
            for chunk in body_chunks {
                body_data.extend_from_slice(&chunk.data);
            }
            
            // Deserialize the complete response body
            match serde_json::from_slice::<T>(&body_data) {
                Ok(deserialized) => fluent_ai_async::emit!(sender, deserialized),
                Err(e) => fluent_ai_async::emit!(sender, T::bad_chunk(format!("JSON deserialization failed: {}", e))),
            }
        })
    }

    /// Execute a PATCH request
    ///
    /// # Arguments
    /// * `url` - The URL to send the PATCH request to
    ///
    /// # Returns
    /// `AsyncStream<HttpChunk, 1024>` for streaming the response
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::Serialize;
    ///
    /// #[derive(Serialize)]
    /// struct PatchUser {
    ///     email: String,
    /// }
    ///
    /// let update = PatchUser {
    ///     email: "newemail@example.com".to_string(),
    /// };
    ///
    /// let response = Http3Builder::json()
    ///     .body(&update)
    ///     .patch("https://api.example.com/users/123");
    /// ```
    pub fn patch(mut self, url: &str) -> AsyncStream<HttpChunk, 1024> {
        // SSRF Protection - Always block dangerous URLs (no config required)
        if is_dangerous_url(url) {
            return AsyncStream::with_channel(move |sender| {
                fluent_ai_async::emit!(sender, HttpChunk::Error("URL blocked for security".to_string()));
            });
        }

        // Handle URL parsing with proper error handling
        let parsed_url = match url.parse::<Url>() {
            Ok(url) => url,
            Err(parse_error) => {
                // Convert to owned string for the closure
                let url_string = url.to_string();
                // Return error stream for invalid URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, HttpChunk::Error(format!("Invalid URL '{}': {}", url_string, parse_error)));
                });
            }
        };

        self.request = self
            .request
            .with_method(Method::PATCH)
            .with_url(parsed_url);

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: PATCH {}", url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        // Use the correct HttpClient.execute() pattern
        let response = self.client.execute(self.request);
        
        // Transform HttpResponse stream to HttpChunk stream
        AsyncStream::with_channel(move |sender| {
            // Convert HttpResponse to body stream and emit as HttpChunks
            let body_stream = response.into_body_stream();
            let body_chunks = body_stream.collect();
            
            for body_chunk in body_chunks {
                let http_chunk = HttpChunk::Body(body_chunk.data);
                fluent_ai_async::emit!(sender, http_chunk);
            }
            
            // Emit End marker
            fluent_ai_async::emit!(sender, HttpChunk::End);
        })
    }
}

// Terminal methods for JsonPathStreaming state
impl Http3Builder<JsonPathStreaming> {
    /// Execute a GET request with JSONPath streaming
    ///
    /// Returns a stream of deserialized objects matching the JSONPath expression.
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let users = Http3Builder::json()
    ///     .array_stream("$.users[*]")
    ///     .get::<User>("https://api.example.com/data");
    /// ```
    pub fn get<T>(mut self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + fluent_ai_async::prelude::MessageChunk + Default + Send + 'static,
    {
        // SSRF Protection - Always block dangerous URLs (no config required)
        if is_dangerous_url(url) {
            return AsyncStream::with_channel(move |sender| {
                fluent_ai_async::emit!(sender, T::bad_chunk("URL blocked for security".to_string()));
            });
        }

        // Handle URL parsing with proper error handling
        let parsed_url = match url.parse::<Url>() {
            Ok(url) => url,
            Err(parse_error) => {
                // Convert to owned string for the closure
                let url_string = url.to_string();
                // Return error stream for invalid URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Invalid URL '{}': {}", url_string, parse_error)));
                });
            }
        };

        self.request = self
            .request
            .with_method(Method::GET)
            .with_url(parsed_url);

        let jsonpath_expr = &self.state.jsonpath_expr;

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: GET {} (JSONPath: {})", url, jsonpath_expr);
        }

        // Terse delegation to JSONPath infrastructure
        let response = self.client.execute(self.request);
        fluent_ai_http3_client::jsonpath::process_response(response, jsonpath_expr)
    }

    /// Execute a POST request with JSONPath streaming
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    pub fn post<T>(mut self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + fluent_ai_async::prelude::MessageChunk + Default + Send + 'static,
    {
        // SSRF Protection - Always block dangerous URLs (no config required)
        if is_dangerous_url(url) {
            return AsyncStream::with_channel(move |sender| {
                fluent_ai_async::emit!(sender, T::bad_chunk("URL blocked for security".to_string()));
            });
        }

        // Handle URL parsing with proper error handling
        let parsed_url = match url.parse::<Url>() {
            Ok(url) => url,
            Err(parse_error) => {
                // Convert to owned string for the closure
                let url_string = url.to_string();
                // Return error stream for invalid URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Invalid URL '{}': {}", url_string, parse_error)));
                });
            }
        };

        self.request = self
            .request
            .with_method(Method::POST)
            .with_url(parsed_url);

        let jsonpath_expr = &self.state.jsonpath_expr;

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: POST {} (JSONPath: {})", url, jsonpath_expr);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        // Terse delegation to JSONPath infrastructure
        let response = self.client.execute(self.request);
        fluent_ai_http3_client::jsonpath::process_response(response, jsonpath_expr)
    }

    /// Execute a PUT request with JSONPath streaming
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    pub fn put<T>(mut self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + fluent_ai_async::prelude::MessageChunk + Default + Send + 'static,
    {
        // SSRF Protection - Always block dangerous URLs (no config required)
        if is_dangerous_url(url) {
            return AsyncStream::with_channel(move |sender| {
                fluent_ai_async::emit!(sender, T::bad_chunk("URL blocked for security".to_string()));
            });
        }

        // Handle URL parsing with proper error handling
        let parsed_url = match url.parse::<Url>() {
            Ok(url) => url,
            Err(parse_error) => {
                // Convert to owned string for the closure
                let url_string = url.to_string();
                // Return error stream for invalid URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Invalid URL '{}': {}", url_string, parse_error)));
                });
            }
        };

        self.request = self
            .request
            .with_method(Method::PUT)
            .with_url(parsed_url);

        let jsonpath_expr = &self.state.jsonpath_expr;

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: PUT {} (JSONPath: {})", url, jsonpath_expr);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        // Terse delegation to JSONPath infrastructure
        let response = self.client.execute(self.request);
        fluent_ai_http3_client::jsonpath::process_response(response, jsonpath_expr)
    }

    /// Execute a PATCH request with JSONPath streaming
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    pub fn patch<T>(mut self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + fluent_ai_async::prelude::MessageChunk + Default + Send + 'static,
    {
        // SSRF Protection - Always block dangerous URLs (no config required)
        if is_dangerous_url(url) {
            return AsyncStream::with_channel(move |sender| {
                fluent_ai_async::emit!(sender, T::bad_chunk("URL blocked for security".to_string()));
            });
        }

        // Handle URL parsing with proper error handling
        let parsed_url = match url.parse::<Url>() {
            Ok(url) => url,
            Err(parse_error) => {
                // Convert to owned string for the closure
                let url_string = url.to_string();
                // Return error stream for invalid URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Invalid URL '{}': {}", url_string, parse_error)));
                });
            }
        };

        self.request = self
            .request
            .with_method(Method::PATCH)
            .with_url(parsed_url);

        let jsonpath_expr = &self.state.jsonpath_expr;

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: PATCH {} (JSONPath: {})", url, jsonpath_expr);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        // Terse delegation to JSONPath infrastructure
        let response = self.client.execute(self.request);
        fluent_ai_http3_client::jsonpath::process_response(response, jsonpath_expr)
    }

    /// Execute a DELETE request with JSONPath streaming
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    pub fn delete<T>(mut self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + fluent_ai_async::prelude::MessageChunk + Default + Send + 'static,
    {
        // SSRF Protection - Always block dangerous URLs (no config required)
        if is_dangerous_url(url) {
            return AsyncStream::with_channel(move |sender| {
                fluent_ai_async::emit!(sender, T::bad_chunk("URL blocked for security".to_string()));
            });
        }

        // Handle URL parsing with proper error handling
        let parsed_url = match url.parse::<Url>() {
            Ok(url) => url,
            Err(parse_error) => {
                // Convert to owned string for the closure
                let url_string = url.to_string();
                // Return error stream for invalid URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Invalid URL '{}': {}", url_string, parse_error)));
                });
            }
        };

        self.request = self
            .request
            .with_method(Method::DELETE)
            .with_url(parsed_url);

        let jsonpath_expr = &self.state.jsonpath_expr;

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: DELETE {} (JSONPath: {})", url, jsonpath_expr);
        }

        // Terse delegation to JSONPath infrastructure
        let response = self.client.execute(self.request);
        fluent_ai_http3_client::jsonpath::process_response(response, jsonpath_expr)
    }
}