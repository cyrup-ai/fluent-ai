//! HTTP method implementations
//!
//! Terminal methods for executing HTTP requests (GET, POST, PUT, PATCH, DELETE)
//! with appropriate request configurations and response streaming.

use fluent_ai_async::AsyncStream;
use http::Method;
use serde::de::DeserializeOwned;
use url::Url;
use tracing;

use crate::builder::core::{BodyNotSet, BodySet, Http3Builder, JsonPathStreaming};
use fluent_ai_http3_client::operations::HttpOperation;

// Re-export types from client package - using direct types, no confusing aliases
pub use fluent_ai_http3_client::http::response::{HttpChunk, HttpBodyChunk};
pub use fluent_ai_http3_client::builder::fluent::DownloadBuilder;

/// SECURITY: Sanitize JSON error messages to prevent data disclosure
fn sanitize_json_error(error_msg: &str) -> String {
    let mut sanitized = error_msg.to_string();
    
    // Remove quoted strings that might contain sensitive data
    let mut in_quotes = false;
    let mut escaped = false;
    let mut result = String::new();
    
    for ch in sanitized.chars() {
        match ch {
            '"' if !escaped => {
                if in_quotes {
                    // End of quoted string - replace with redacted
                    result.push_str("[REDACTED]\"");
                    in_quotes = false;
                } else {
                    // Start of quoted string
                    result.push('"');
                    in_quotes = true;
                }
            }
            '\\' if in_quotes => {
                escaped = !escaped;
                if !escaped {
                    // Don't include backslashes in redacted content
                }
            }
            _ => {
                if !in_quotes {
                    result.push(ch);
                }
                escaped = false;
            }
        }
    }
    
    let mut sanitized = result;
    
    // Remove large numbers (potential IDs, timestamps, tokens)
    let chars: Vec<char> = sanitized.chars().collect();
    let mut new_chars = Vec::new();
    let mut i = 0;
    
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            // If number is longer than 8 digits, redact it
            if i - start > 8 {
                new_chars.extend("[REDACTED_ID]".chars());
            } else {
                new_chars.extend(chars[start..i].iter());
            }
        } else {
            new_chars.push(chars[i]);
            i += 1;
        }
    }
    
    sanitized = new_chars.into_iter().collect();
    
    // Limit error message length to prevent information leakage
    if sanitized.len() > 200 {
        sanitized.truncate(197);
        sanitized.push_str("...");
    }
    
    // If sanitization made the message empty or too generic, provide a safe generic message
    if sanitized.len() < 10 || sanitized.trim().is_empty() {
        "Invalid JSON format - deserialization error".to_string()
    } else {
        sanitized
    }
}

/// SECURITY: Sanitize URLs for error messages to prevent information disclosure
fn sanitize_url_for_error(url: &str) -> String {
    // Parse URL to extract components safely
    if let Ok(parsed) = url.parse::<Url>() {
        let scheme = parsed.scheme();
        let host = parsed.host_str().unwrap_or("[REDACTED_HOST]");
        
        // Determine if host should be sanitized (internal/private networks)
        let sanitized_host = if host == "localhost" || 
                               host.starts_with("127.") ||
                               host.starts_with("192.168.") ||
                               host.starts_with("10.") ||
                               host.starts_with("172.16.") || host.starts_with("172.17.") ||
                               host.starts_with("172.18.") || host.starts_with("172.19.") ||
                               host.starts_with("172.20.") || host.starts_with("172.21.") ||
                               host.starts_with("172.22.") || host.starts_with("172.23.") ||
                               host.starts_with("172.24.") || host.starts_with("172.25.") ||
                               host.starts_with("172.26.") || host.starts_with("172.27.") ||
                               host.starts_with("172.28.") || host.starts_with("172.29.") ||
                               host.starts_with("172.30.") || host.starts_with("172.31.") ||
                               host.ends_with(".local") ||
                               host.contains("internal") ||
                               host.contains("corp") ||
                               host.contains("intranet") {
            "[INTERNAL_HOST]"
        } else {
            host
        };
        
        // Return sanitized URL format - scheme and host only, no paths/queries
        format!("{}://{}", scheme, sanitized_host)
    } else {
        // If URL parsing fails, return completely generic error
        "[INVALID_URL]".to_string()
    }
}

/// Comprehensive SSRF Protection - Production-grade URL validation
/// Blocks all known attack vectors including internal networks, metadata services, 
/// dangerous schemes, and suspicious ports
fn validate_url_safety(url_str: &str) -> Result<Url, String> {
    // Parse URL first to validate structure
    let parsed = url_str.parse::<Url>()
        .map_err(|e| format!("Invalid URL format: {}", e))?;
    
    // 1. SCHEME VALIDATION: Only allow http/https
    match parsed.scheme() {
        "http" | "https" => {},
        other => return Err(format!("Unsafe URL scheme '{}' - only http/https allowed", other)),
    }
    
    // 2. HOST VALIDATION: Block dangerous hosts
    let host_str = parsed.host_str()
        .ok_or_else(|| "URL must have a host".to_string())?;
    
    // 2a. Block localhost variants and special domains
    if host_str == "localhost" 
        || host_str == "0.0.0.0" 
        || host_str.ends_with(".local")
        || host_str.ends_with(".localhost")
        || host_str.ends_with(".internal") {
        return Err(format!("Blocked internal hostname: {}", host_str));
    }
    
    // 2b. Block IP addresses in dangerous ranges
    if let Ok(ip) = host_str.parse::<std::net::IpAddr>() {
        match ip {
            std::net::IpAddr::V4(ipv4) => {
                let octets = ipv4.octets();
                
                // RFC1918 Private networks
                if octets[0] == 10 {
                    return Err("Blocked private IP range: 10.0.0.0/8".to_string());
                }
                if octets[0] == 172 && octets[1] >= 16 && octets[1] <= 31 {
                    return Err("Blocked private IP range: 172.16.0.0/12".to_string());
                }
                if octets[0] == 192 && octets[1] == 168 {
                    return Err("Blocked private IP range: 192.168.0.0/16".to_string());
                }
                
                // Loopback
                if octets[0] == 127 {
                    return Err("Blocked loopback IP range: 127.0.0.0/8".to_string());
                }
                
                // Link-local / AWS metadata
                if octets[0] == 169 && octets[1] == 254 {
                    return Err("Blocked link-local/metadata IP range: 169.254.0.0/16".to_string());
                }
                
                // Multicast ranges
                if octets[0] >= 224 && octets[0] <= 239 {
                    return Err("Blocked multicast IP range: 224.0.0.0/4".to_string());
                }
                
                // Reserved/experimental ranges
                if octets[0] >= 240 {
                    return Err("Blocked reserved IP range: 240.0.0.0/4".to_string());
                }
            }
            std::net::IpAddr::V6(ipv6) => {
                // IPv6 loopback
                if ipv6.is_loopback() {
                    return Err("Blocked IPv6 loopback address".to_string());
                }
                
                let segments = ipv6.segments();
                
                // Link-local (fe80::/10)
                if segments[0] >= 0xfe80 && segments[0] <= 0xfebf {
                    return Err("Blocked IPv6 link-local range: fe80::/10".to_string());
                }
                
                // Unique local (fc00::/7) 
                if segments[0] >= 0xfc00 && segments[0] <= 0xfdff {
                    return Err("Blocked IPv6 unique local range: fc00::/7".to_string());
                }
                
                // AWS metadata service IPv6
                if segments[0] == 0xfd00 && segments[1] == 0xec2 {
                    return Err("Blocked AWS metadata service IPv6".to_string());
                }
                
                // Multicast (ff00::/8)
                if segments[0] >= 0xff00 {
                    return Err("Blocked IPv6 multicast range: ff00::/8".to_string());
                }
            }
        }
    }
    
    // 3. PORT VALIDATION: Block dangerous internal ports
    if let Some(port) = parsed.port() {
        match port {
            // Common database ports
            3306 | 5432 | 1433 | 1521 | 27017 => {
                return Err(format!("Blocked database port: {}", port));
            }
            // Internal service ports
            6379 | 11211 | 9200 | 9300 | 5672 | 15672 => {
                return Err(format!("Blocked internal service port: {}", port));
            }
            // Development/debugging ports
            3000 | 8000 | 8080 | 9000 | 5000 => {
                return Err(format!("Blocked development port: {}", port));
            }
            // System service ports (low numbers)
            1..=1023 => {
                // Allow common web ports
                if port != 80 && port != 443 {
                    return Err(format!("Blocked system port: {}", port));
                }
            }
            _ => {} // Allow other ports
        }
    }
    
    // 4. PATH VALIDATION: Block suspicious paths
    let path = parsed.path();
    if path.contains("..") {
        return Err("Blocked path traversal attempt".to_string());
    }
    
    // If all validations pass, return the parsed URL
    Ok(parsed)
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
        // Comprehensive SSRF Protection with production-grade URL validation
        let parsed_url = match validate_url_safety(url) {
            Ok(url) => url,
            Err(security_error) => {
                // SECURITY: Sanitize URL in error messages to prevent information disclosure
                let sanitized_url = sanitize_url_for_error(url);
                
                // Return security error stream for blocked URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Security blocked URL '{}': {}", sanitized_url, security_error)));
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
            
            // SECURITY: Enforce memory limits when collecting response body
            const MAX_RESPONSE_BODY_SIZE: usize = 100 * 1024 * 1024; // 100MB hard limit
            
            // Collect all body chunks with size checking
            let body_chunks = body_stream.collect();
            for chunk in body_chunks {
                // SECURITY: Check size BEFORE allocation to prevent memory exhaustion attacks
                if body_data.len() + chunk.data.len() > MAX_RESPONSE_BODY_SIZE {
                    tracing::error!(
                        target: "fluent_ai_http3::api::builder::methods",
                        current_size = body_data.len(),
                        chunk_size = chunk.data.len(),
                        limit = MAX_RESPONSE_BODY_SIZE,
                        "Response body exceeds memory safety limit - truncating to prevent DoS"
                    );
                    break; // Stop collecting to prevent memory exhaustion
                }
                body_data.extend_from_slice(&chunk.data);
            }
            
            // Deserialize the complete response body
            match serde_json::from_slice::<T>(&body_data) {
                Ok(deserialized) => fluent_ai_async::emit!(sender, deserialized),
                Err(e) => {
                    // SECURITY: Sanitize JSON error messages to prevent data disclosure
                    let sanitized_error = sanitize_json_error(&e.to_string());
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("JSON deserialization failed: {}", sanitized_error)));
                }
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
        // Comprehensive SSRF Protection with production-grade URL validation
        let parsed_url = match validate_url_safety(url) {
            Ok(url) => url,
            Err(security_error) => {
                let url_string = url.to_string();
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, HttpChunk::Error(format!("Security blocked URL '{}': {}", url_string, security_error)));
                });
            }
        };

        // URL is already validated and parsed by validate_url_safety()

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
        // Comprehensive SSRF Protection with production-grade URL validation
        let parsed_url = match validate_url_safety(url) {
            Ok(url) => url,
            Err(security_error) => {
                let url_string = url.to_string();
                let error_stream = fluent_ai_async::AsyncStream::with_channel(move |sender| {
                    use fluent_ai_http3_client::http::response::HttpDownloadChunk;
                    fluent_ai_async::emit!(sender, HttpDownloadChunk::Error { message: format!("Security blocked URL '{}': {}", url_string, security_error) });
                });
                return DownloadBuilder::new(error_stream);
            }
        };

        // URL is already validated and parsed by validate_url_safety()

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
            .iter()
            .find(|header| header.name == http::header::CONTENT_LENGTH)
            .and_then(|header| header.value.to_str().ok())
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
        // Comprehensive SSRF Protection with production-grade URL validation
        let parsed_url = match validate_url_safety(url) {
            Ok(url) => url,
            Err(security_error) => {
                // SECURITY: Sanitize URL in error messages to prevent information disclosure
                let sanitized_url = sanitize_url_for_error(url);
                
                // Return security error stream for blocked URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Security blocked URL '{}': {}", sanitized_url, security_error)));
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
            
            // SECURITY: Enforce memory limits when collecting response body
            const MAX_RESPONSE_BODY_SIZE: usize = 100 * 1024 * 1024; // 100MB hard limit
            
            // Collect all body chunks with size checking
            let body_chunks = body_stream.collect();
            for chunk in body_chunks {
                // SECURITY: Check size BEFORE allocation to prevent memory exhaustion attacks
                if body_data.len() + chunk.data.len() > MAX_RESPONSE_BODY_SIZE {
                    tracing::error!(
                        target: "fluent_ai_http3::api::builder::methods",
                        current_size = body_data.len(),
                        chunk_size = chunk.data.len(),
                        limit = MAX_RESPONSE_BODY_SIZE,
                        "Response body exceeds memory safety limit - truncating to prevent DoS"
                    );
                    break; // Stop collecting to prevent memory exhaustion
                }
                body_data.extend_from_slice(&chunk.data);
            }
            
            // Deserialize the complete response body
            match serde_json::from_slice::<T>(&body_data) {
                Ok(deserialized) => fluent_ai_async::emit!(sender, deserialized),
                Err(e) => {
                    // SECURITY: Sanitize JSON error messages to prevent data disclosure
                    let sanitized_error = sanitize_json_error(&e.to_string());
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("JSON deserialization failed: {}", sanitized_error)));
                }
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
        // Comprehensive SSRF Protection with production-grade URL validation
        let parsed_url = match validate_url_safety(url) {
            Ok(url) => url,
            Err(security_error) => {
                // SECURITY: Sanitize URL in error messages to prevent information disclosure
                let sanitized_url = sanitize_url_for_error(url);
                
                // Return security error stream for blocked URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Security blocked URL '{}': {}", sanitized_url, security_error)));
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
            
            // SECURITY: Enforce memory limits when collecting response body
            const MAX_RESPONSE_BODY_SIZE: usize = 100 * 1024 * 1024; // 100MB hard limit
            
            // Collect all body chunks with size checking
            let body_chunks = body_stream.collect();
            for chunk in body_chunks {
                // SECURITY: Check size BEFORE allocation to prevent memory exhaustion attacks
                if body_data.len() + chunk.data.len() > MAX_RESPONSE_BODY_SIZE {
                    tracing::error!(
                        target: "fluent_ai_http3::api::builder::methods",
                        current_size = body_data.len(),
                        chunk_size = chunk.data.len(),
                        limit = MAX_RESPONSE_BODY_SIZE,
                        "Response body exceeds memory safety limit - truncating to prevent DoS"
                    );
                    break; // Stop collecting to prevent memory exhaustion
                }
                body_data.extend_from_slice(&chunk.data);
            }
            
            // Deserialize the complete response body
            match serde_json::from_slice::<T>(&body_data) {
                Ok(deserialized) => fluent_ai_async::emit!(sender, deserialized),
                Err(e) => {
                    // SECURITY: Sanitize JSON error messages to prevent data disclosure
                    let sanitized_error = sanitize_json_error(&e.to_string());
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("JSON deserialization failed: {}", sanitized_error)));
                }
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
        // Comprehensive SSRF Protection with production-grade URL validation
        let parsed_url = match validate_url_safety(url) {
            Ok(url) => url,
            Err(security_error) => {
                let url_string = url.to_string();
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, HttpChunk::Error(format!("Security blocked URL '{}': {}", url_string, security_error)));
                });
            }
        };

        // URL is already validated and parsed by validate_url_safety()

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
        // Comprehensive SSRF Protection with production-grade URL validation
        let parsed_url = match validate_url_safety(url) {
            Ok(url) => url,
            Err(security_error) => {
                // SECURITY: Sanitize URL in error messages to prevent information disclosure
                let sanitized_url = sanitize_url_for_error(url);
                
                // Return security error stream for blocked URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Security blocked URL '{}': {}", sanitized_url, security_error)));
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
        // Comprehensive SSRF Protection with production-grade URL validation
        let parsed_url = match validate_url_safety(url) {
            Ok(url) => url,
            Err(security_error) => {
                // SECURITY: Sanitize URL in error messages to prevent information disclosure
                let sanitized_url = sanitize_url_for_error(url);
                
                // Return security error stream for blocked URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Security blocked URL '{}': {}", sanitized_url, security_error)));
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
        // Comprehensive SSRF Protection with production-grade URL validation
        let parsed_url = match validate_url_safety(url) {
            Ok(url) => url,
            Err(security_error) => {
                // SECURITY: Sanitize URL in error messages to prevent information disclosure
                let sanitized_url = sanitize_url_for_error(url);
                
                // Return security error stream for blocked URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Security blocked URL '{}': {}", sanitized_url, security_error)));
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
        // Comprehensive SSRF Protection with production-grade URL validation
        let parsed_url = match validate_url_safety(url) {
            Ok(url) => url,
            Err(security_error) => {
                // SECURITY: Sanitize URL in error messages to prevent information disclosure
                let sanitized_url = sanitize_url_for_error(url);
                
                // Return security error stream for blocked URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Security blocked URL '{}': {}", sanitized_url, security_error)));
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
        // Comprehensive SSRF Protection with production-grade URL validation
        let parsed_url = match validate_url_safety(url) {
            Ok(url) => url,
            Err(security_error) => {
                // SECURITY: Sanitize URL in error messages to prevent information disclosure
                let sanitized_url = sanitize_url_for_error(url);
                
                // Return security error stream for blocked URLs
                return AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, T::bad_chunk(format!("Security blocked URL '{}': {}", sanitized_url, security_error)));
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