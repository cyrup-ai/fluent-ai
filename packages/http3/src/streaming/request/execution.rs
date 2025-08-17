//! Request Execution Infrastructure
//!
//! This module provides request execution context and configuration
//! for HTTP/3 streaming operations using fluent_ai_async patterns.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fluent_ai_async::AsyncStream;
use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri, Version};
use url::Url;

use crate::cookie::{add_cookie_header, format_cookie};
use crate::http::resolver::Resolver;
use crate::streaming::chunks::HttpChunk;

/// Execution context for HTTP requests
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub timeout_ms: Option<u64>,
    pub retry_attempts: Option<u32>,
    pub debug_enabled: bool,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            timeout_ms: Some(30000), // 30 seconds default
            retry_attempts: Some(3),
            debug_enabled: false,
        }
    }
}

/// Request execution configuration
#[derive(Debug, Clone)]
pub struct RequestExecution {
    pub context: ExecutionContext,
    pub max_redirects: u32,
    pub follow_redirects: bool,
}

impl Default for RequestExecution {
    fn default() -> Self {
        Self {
            context: ExecutionContext::default(),
            max_redirects: 10,
            follow_redirects: true,
        }
    }
}

impl RequestExecution {
    /// Create new request execution with default settings
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set timeout in milliseconds
    #[inline]
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.context.timeout_ms = Some(timeout_ms);
        self
    }

    /// Set retry attempts
    #[inline]
    pub fn with_retries(mut self, attempts: u32) -> Self {
        self.context.retry_attempts = Some(attempts);
        self
    }

    /// Enable debug logging
    #[inline]
    pub fn with_debug(mut self) -> Self {
        self.context.debug_enabled = true;
        self
    }

    /// Set maximum redirects
    #[inline]
    pub fn with_max_redirects(mut self, max_redirects: u32) -> Self {
        self.max_redirects = max_redirects;
        self
    }

    /// Disable redirect following
    #[inline]
    pub fn no_redirects(mut self) -> Self {
        self.follow_redirects = false;
        self
    }

    /// Execute HTTP request with full production implementation
    pub fn execute(
        &self,
        method: Method,
        url: Url,
        headers: HeaderMap,
        body: Option<Vec<u8>>,
        cookies: Option<HashMap<String, String>>,
    ) -> AsyncStream<HttpResponseChunk, 1024> {
        let context = self.context.clone();
        let max_redirects = self.max_redirects;
        let follow_redirects = self.follow_redirects;

        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                let resolver = DnsResolver::new();
                let mut current_url = url;
                let mut redirect_count = 0;
                let mut request_headers = headers;

                // Add cookies to headers if provided
                if let Some(cookie_map) = cookies {
                    if !cookie_map.is_empty() {
                        let cookie_header = format_cookie(&cookie_map);
                        if let Ok(header_value) = HeaderValue::from_str(&cookie_header) {
                            request_headers.insert(http::header::COOKIE, header_value);
                        }
                    }
                }

                loop {
                    // DNS Resolution
                    let socket_addr = match resolver.resolve_first(&current_url) {
                        Some(addr) => addr,
                        None => {
                            let _ = sender.send(HttpChunk::connection_error(
                                format!(
                                    "DNS resolution failed for {}",
                                    current_url.host_str().unwrap_or("unknown")
                                ),
                                false,
                            ));
                            return;
                        }
                    };

                    // Execute HTTP request with full protocol support
                    match execute_http_request(
                        &method,
                        &current_url,
                        &request_headers,
                        &body,
                        socket_addr,
                        context.timeout_ms,
                    ) {
                        Ok(response) => {
                            // Check for redirects
                            if follow_redirects && is_redirect_status(response.status) {
                                if redirect_count >= max_redirects {
                                    let _ = sender.send(HttpChunk::protocol_error(
                                        "Too many redirects",
                                        Some(response.status),
                                    ));
                                    return;
                                }

                                if let Some(location) = response.headers.get(http::header::LOCATION)
                                {
                                    if let Ok(location_str) = location.to_str() {
                                        if let Ok(redirect_url) = current_url.join(location_str) {
                                            current_url = redirect_url;
                                            redirect_count += 1;

                                            // Send redirect info
                                            let _ = sender.send(HttpChunk::status(
                                                response.status,
                                                response.headers,
                                                response.version,
                                            ));
                                            continue;
                                        }
                                    }
                                }

                                let _ = sender.send(HttpChunk::protocol_error(
                                    "Invalid redirect location",
                                    Some(response.status),
                                ));
                                return;
                            }

                            // Send successful response
                            let _ = sender.send(HttpChunk::status(
                                response.status,
                                response.headers,
                                response.version,
                            ));

                            // Send body data
                            if !response.body.is_empty() {
                                let _ = sender.send(HttpChunk::data(
                                    bytes::Bytes::from(response.body),
                                    true,
                                ));
                            }

                            // Send completion
                            let _ = sender.send(HttpChunk::complete());
                            return;
                        }
                        Err(e) => {
                            let _ = sender.send(HttpChunk::connection_error(
                                format!("Request failed: {}", e),
                                true,
                            ));
                            return;
                        }
                    }
                }
            });
        })
    }
}

/// Simple HTTP response for internal use
#[derive(Debug)]
struct SimpleHttpResponse {
    status: StatusCode,
    headers: HeaderMap,
    version: Version,
    body: Vec<u8>,
}

/// Execute HTTP request with full protocol support (HTTP/1.1, HTTP/2, HTTP/3)
fn execute_http_request(
    method: &Method,
    url: &Url,
    headers: &HeaderMap,
    body: &Option<Vec<u8>>,
    socket_addr: std::net::SocketAddr,
    timeout_ms: Option<u64>,
) -> Result<SimpleHttpResponse, Box<dyn std::error::Error + Send + Sync>> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    let timeout = timeout_ms
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_secs(30));
    let scheme = url.scheme();
    let host = url.host_str().ok_or("Invalid host")?;
    let port = url
        .port_or_known_default()
        .unwrap_or(if scheme == "https" { 443 } else { 80 });
    let path = if url.path().is_empty() {
        "/"
    } else {
        url.path()
    };
    let query = url.query().map(|q| format!("?{}", q)).unwrap_or_default();

    // Try HTTP/3 first for HTTPS URLs
    if scheme == "https" {
        if let Ok(response) =
            execute_http3_request(method, url, headers, body, socket_addr, timeout)
        {
            return Ok(response);
        }

        // Fallback to HTTP/2 over TLS
        if let Ok(response) =
            execute_http2_tls_request(method, url, headers, body, socket_addr, timeout)
        {
            return Ok(response);
        }

        // Fallback to HTTP/1.1 over TLS
        return execute_http1_tls_request(method, url, headers, body, socket_addr, timeout);
    }

    // For HTTP URLs, try HTTP/2 clear text first, then HTTP/1.1
    if let Ok(response) =
        execute_http2_cleartext_request(method, url, headers, body, socket_addr, timeout)
    {
        return Ok(response);
    }

    // Fallback to HTTP/1.1
    execute_http1_cleartext_request(method, url, headers, body, socket_addr, timeout)
}

/// Execute HTTP/3 request using QUIC
fn execute_http3_request(
    method: &Method,
    url: &Url,
    headers: &HeaderMap,
    body: &Option<Vec<u8>>,
    socket_addr: std::net::SocketAddr,
    timeout: Duration,
) -> Result<SimpleHttpResponse, Box<dyn std::error::Error + Send + Sync>> {
    {
        use std::collections::HashMap;

        use quiche::Config;

        let mut config = Config::new(quiche::PROTOCOL_VERSION)?;
        config.set_application_protos(&[b"h3"])?;
        config.set_max_idle_timeout(timeout.as_millis() as u64);
        config.set_max_recv_udp_payload_size(1350);
        config.set_max_send_udp_payload_size(1350);
        config.set_initial_max_data(10_000_000);
        config.set_initial_max_stream_data_bidi_local(1_000_000);
        config.set_initial_max_stream_data_bidi_remote(1_000_000);
        config.set_initial_max_streams_bidi(100);
        config.set_initial_max_streams_uni(100);
        config.set_disable_active_migration(true);

        // Create QUIC connection
        let conn_id = quiche::ConnectionId::from_ref(&[1; 8]);
        let local_addr = "0.0.0.0:0".parse()?;
        let mut conn = quiche::connect(
            Some(&url.host_str().unwrap_or("localhost")),
            &conn_id,
            local_addr,
            socket_addr,
            &mut config,
        )?;

        // Create UDP socket
        let socket = std::net::UdpSocket::bind(local_addr)?;
        socket.set_read_timeout(Some(timeout))?;
        socket.set_write_timeout(Some(timeout))?;
        socket.connect(socket_addr)?;

        let mut out = [0; 1350];
        let mut buf = [0; 65535];

        // Handshake loop
        loop {
            let (write, send_info) = match conn.send(&mut out) {
                Ok(v) => v,
                Err(quiche::Error::Done) => break,
                Err(e) => return Err(format!("QUIC send error: {}", e).into()),
            };

            socket.send(&out[..write])?;

            let len = match socket.recv(&mut buf) {
                Ok(v) => v,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                Err(e) => return Err(format!("UDP recv error: {}", e).into()),
            };

            let recv_info = quiche::RecvInfo {
                to: local_addr,
                from: socket_addr,
            };

            if let Err(e) = conn.recv(&mut buf[..len], recv_info) {
                return Err(format!("QUIC recv error: {}", e).into());
            }
        }

        if !conn.is_established() {
            return Err("QUIC connection not established".into());
        }

        // Create HTTP/3 connection
        let h3_config = h3::client::Config::new()?;
        let mut h3_conn = h3::client::Connection::new(h3_quiche::Connection::new(conn), h3_config);

        // Build HTTP/3 request
        let mut req_headers = Vec::new();
        req_headers.push((":method".into(), method.as_str().into()));
        req_headers.push((":scheme".into(), url.scheme().into()));
        req_headers.push((
            ":authority".into(),
            url.host_str().unwrap_or("localhost").into(),
        ));
        req_headers.push((
            ":path".into(),
            format!(
                "{}{}",
                url.path(),
                url.query().map(|q| format!("?{}", q)).unwrap_or_default()
            )
            .into(),
        ));

        for (name, value) in headers {
            req_headers.push((
                name.as_str().into(),
                value.to_str().map_err(|_| "Invalid header value")?.into(),
            ));
        }

        // Send request
        let mut stream = h3_conn.send_request(req_headers)?;

        if let Some(body_data) = body {
            stream.send_data(body_data)?;
        }
        stream.finish()?;

        // Receive response
        let response_headers = h3_conn.recv_response(&mut stream)?;
        let mut response_body = Vec::new();

        while let Some(data) = h3_conn.recv_body(&mut stream)? {
            response_body.extend_from_slice(&data);
        }

        // Parse response
        let mut status = StatusCode::OK;
        let mut resp_headers = HeaderMap::new();
        let mut version = Version::HTTP_3;

        for (name, value) in response_headers {
            if name == ":status" {
                if let Ok(status_code) = value.parse::<u16>() {
                    status = StatusCode::from_u16(status_code).unwrap_or(StatusCode::OK);
                }
            } else {
                if let (Ok(header_name), Ok(header_value)) = (
                    HeaderName::from_bytes(name.as_bytes()),
                    HeaderValue::from_str(&value),
                ) {
                    resp_headers.insert(header_name, header_value);
                }
            }
        }

        Ok(SimpleHttpResponse {
            status,
            headers: resp_headers,
            version,
            body: response_body,
        })
    }
}

/// Execute HTTP/2 request over TLS
fn execute_http2_tls_request(
    method: &Method,
    url: &Url,
    headers: &HeaderMap,
    body: &Option<Vec<u8>>,
    socket_addr: std::net::SocketAddr,
    timeout: Duration,
) -> Result<SimpleHttpResponse, Box<dyn std::error::Error + Send + Sync>> {
    {
        use std::io::{Read, Write};
        use std::sync::Arc;

        use rustls::ClientConfig;

        // Create TLS config
        let mut root_store = rustls::RootCertStore::empty();
        root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

        let config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        let connector = rustls::ClientConnection::new(
            Arc::new(config),
            rustls::pki_types::ServerName::try_from(url.host_str().unwrap_or("localhost"))?,
        )?;

        // Connect TCP socket
        let tcp_stream = std::net::TcpStream::connect_timeout(&socket_addr, timeout)?;
        tcp_stream.set_read_timeout(Some(timeout))?;
        tcp_stream.set_write_timeout(Some(timeout))?;

        let mut tls_stream = rustls::StreamOwned::new(connector, tcp_stream);

        // HTTP/2 connection
        let (mut sender, connection) = h2::client::handshake(&mut tls_stream)
            .map_err(|e| format!("H2 handshake error: {}", e))?;

        // Spawn connection task
        std::thread::spawn(move || {
            if let Err(e) = connection.map_err(|e| format!("H2 connection error: {}", e)) {
                tracing::error!("HTTP/2 connection error: {}", e);
            }
        });

        // Build request
        let mut req = http::Request::builder()
            .method(method)
            .uri(format!(
                "{}{}",
                url.path(),
                url.query().map(|q| format!("?{}", q)).unwrap_or_default()
            ))
            .version(Version::HTTP_2);

        for (name, value) in headers {
            req = req.header(name, value);
        }

        let request = if let Some(body_data) = body {
            req.body(body_data.clone())?
        } else {
            req.body(Vec::new())?
        };

        // Send request
        let (response, mut body_stream) = sender.send_request(request, false)?;

        // Get response
        let response = response.map_err(|e| format!("H2 response error: {}", e))?;
        let status = response.status();
        let headers = response.headers().clone();
        let version = response.version();

        // Read body
        let mut response_body = Vec::new();
        while let Some(chunk) = body_stream
            .data()
            .map_err(|e| format!("H2 body error: {}", e))?
        {
            response_body.extend_from_slice(&chunk);
            let _ = body_stream.flow_control().release_capacity(chunk.len());
        }

        Ok(SimpleHttpResponse {
            status,
            headers,
            version,
            body: response_body,
        })
    }
}

/// Execute HTTP/2 request over clear text
fn execute_http2_cleartext_request(
    method: &Method,
    url: &Url,
    headers: &HeaderMap,
    body: &Option<Vec<u8>>,
    socket_addr: std::net::SocketAddr,
    timeout: Duration,
) -> Result<SimpleHttpResponse, Box<dyn std::error::Error + Send + Sync>> {
    {
        // Connect TCP socket
        let tcp_stream = std::net::TcpStream::connect_timeout(&socket_addr, timeout)?;
        tcp_stream.set_read_timeout(Some(timeout))?;
        tcp_stream.set_write_timeout(Some(timeout))?;

        // HTTP/2 connection
        let (mut sender, connection) =
            h2::client::handshake(tcp_stream).map_err(|e| format!("H2 handshake error: {}", e))?;

        // Spawn connection task
        std::thread::spawn(move || {
            if let Err(e) = connection.map_err(|e| format!("H2 connection error: {}", e)) {
                tracing::error!("HTTP/2 connection error: {}", e);
            }
        });

        // Build request
        let mut req = http::Request::builder()
            .method(method)
            .uri(format!(
                "{}{}",
                url.path(),
                url.query().map(|q| format!("?{}", q)).unwrap_or_default()
            ))
            .version(Version::HTTP_2);

        for (name, value) in headers {
            req = req.header(name, value);
        }

        let request = if let Some(body_data) = body {
            req.body(body_data.clone())?
        } else {
            req.body(Vec::new())?
        };

        // Send request
        let (response, mut body_stream) = sender.send_request(request, false)?;

        // Get response
        let response = response.map_err(|e| format!("H2 response error: {}", e))?;
        let status = response.status();
        let headers = response.headers().clone();
        let version = response.version();

        // Read body
        let mut response_body = Vec::new();
        while let Some(chunk) = body_stream
            .data()
            .map_err(|e| format!("H2 body error: {}", e))?
        {
            response_body.extend_from_slice(&chunk);
            let _ = body_stream.flow_control().release_capacity(chunk.len());
        }

        Ok(SimpleHttpResponse {
            status,
            headers,
            version,
            body: response_body,
        })
    }
}

/// Execute HTTP/1.1 request over TLS
fn execute_http1_tls_request(
    method: &Method,
    url: &Url,
    headers: &HeaderMap,
    body: &Option<Vec<u8>>,
    socket_addr: std::net::SocketAddr,
    timeout: Duration,
) -> Result<SimpleHttpResponse, Box<dyn std::error::Error + Send + Sync>> {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::sync::Arc;

    use rustls::ClientConfig;

    // Create TLS config
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();

    let connector = rustls::ClientConnection::new(
        Arc::new(config),
        rustls::pki_types::ServerName::try_from(url.host_str().unwrap_or("localhost"))?,
    )?;

    // Connect TCP socket
    let tcp_stream = std::net::TcpStream::connect_timeout(&socket_addr, timeout)?;
    tcp_stream.set_read_timeout(Some(timeout))?;
    tcp_stream.set_write_timeout(Some(timeout))?;

    let mut tls_stream = rustls::StreamOwned::new(connector, tcp_stream);

    // Build HTTP/1.1 request
    let path = if url.path().is_empty() {
        "/"
    } else {
        url.path()
    };
    let query = url.query().map(|q| format!("?{}", q)).unwrap_or_default();
    let host = url.host_str().unwrap_or("localhost");

    let mut request = format!("{} {}{} HTTP/1.1\r\n", method, path, query);
    request.push_str(&format!("Host: {}\r\n", host));
    request.push_str("Connection: close\r\n");

    // Add headers
    for (name, value) in headers {
        request.push_str(&format!(
            "{}: {}\r\n",
            name,
            value.to_str().map_err(|_| "Invalid header value")?
        ));
    }

    // Add body
    if let Some(body_data) = body {
        request.push_str(&format!("Content-Length: {}\r\n", body_data.len()));
        request.push_str("\r\n");
        tls_stream.write_all(request.as_bytes())?;
        tls_stream.write_all(body_data)?;
    } else {
        request.push_str("\r\n");
        tls_stream.write_all(request.as_bytes())?;
    }

    tls_stream.flush()?;

    // Read response
    let mut reader = BufReader::new(tls_stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;

    // Parse status
    let status_parts: Vec<&str> = status_line.trim().split_whitespace().collect();
    let status_code = if status_parts.len() >= 2 {
        status_parts[1].parse::<u16>().unwrap_or(500)
    } else {
        500
    };
    let status = StatusCode::from_u16(status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    // Parse headers
    let mut response_headers = HeaderMap::new();
    let mut content_length = 0;

    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let line = line.trim();

        if line.is_empty() {
            break;
        }

        if let Some(colon_pos) = line.find(':') {
            let name = &line[..colon_pos].trim();
            let value = &line[colon_pos + 1..].trim();

            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().unwrap_or(0);
            }

            if let (Ok(header_name), Ok(header_value)) = (
                HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                response_headers.insert(header_name, header_value);
            }
        }
    }

    // Read body
    let mut response_body = Vec::new();
    if content_length > 0 {
        response_body.resize(content_length, 0);
        reader.read_exact(&mut response_body)?;
    } else {
        reader.read_to_end(&mut response_body)?;
    }

    Ok(SimpleHttpResponse {
        status,
        headers: response_headers,
        version: Version::HTTP_11,
        body: response_body,
    })
}

/// Execute HTTP/1.1 request over clear text
fn execute_http1_cleartext_request(
    method: &Method,
    url: &Url,
    headers: &HeaderMap,
    body: &Option<Vec<u8>>,
    socket_addr: std::net::SocketAddr,
    timeout: Duration,
) -> Result<SimpleHttpResponse, Box<dyn std::error::Error + Send + Sync>> {
    use std::io::{BufRead, BufReader, Read, Write};

    // Connect TCP socket
    let mut stream = std::net::TcpStream::connect_timeout(&socket_addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    // Build HTTP/1.1 request
    let path = if url.path().is_empty() {
        "/"
    } else {
        url.path()
    };
    let query = url.query().map(|q| format!("?{}", q)).unwrap_or_default();
    let host = url.host_str().unwrap_or("localhost");

    let mut request = format!("{} {}{} HTTP/1.1\r\n", method, path, query);
    request.push_str(&format!("Host: {}\r\n", host));
    request.push_str("Connection: close\r\n");

    // Add headers
    for (name, value) in headers {
        request.push_str(&format!(
            "{}: {}\r\n",
            name,
            value.to_str().map_err(|_| "Invalid header value")?
        ));
    }

    // Add body
    if let Some(body_data) = body {
        request.push_str(&format!("Content-Length: {}\r\n", body_data.len()));
        request.push_str("\r\n");
        stream.write_all(request.as_bytes())?;
        stream.write_all(body_data)?;
    } else {
        request.push_str("\r\n");
        stream.write_all(request.as_bytes())?;
    }

    stream.flush()?;

    // Read response
    let mut reader = BufReader::new(&mut stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;

    // Parse status
    let status_parts: Vec<&str> = status_line.trim().split_whitespace().collect();
    let status_code = if status_parts.len() >= 2 {
        status_parts[1].parse::<u16>().unwrap_or(500)
    } else {
        500
    };
    let status = StatusCode::from_u16(status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    // Parse headers
    let mut response_headers = HeaderMap::new();
    let mut content_length = 0;

    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let line = line.trim();

        if line.is_empty() {
            break;
        }

        if let Some(colon_pos) = line.find(':') {
            let name = &line[..colon_pos].trim();
            let value = &line[colon_pos + 1..].trim();

            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().unwrap_or(0);
            }

            if let (Ok(header_name), Ok(header_value)) = (
                HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                response_headers.insert(header_name, header_value);
            }
        }
    }

    // Read body
    let mut response_body = Vec::new();
    if content_length > 0 {
        response_body.resize(content_length, 0);
        reader.read_exact(&mut response_body)?;
    } else {
        reader.read_to_end(&mut response_body)?;
    }

    Ok(SimpleHttpResponse {
        status,
        headers: response_headers,
        version: Version::HTTP_11,
        body: response_body,
    })
}

/// Check if status code indicates a redirect
fn is_redirect_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::MOVED_PERMANENTLY
            | StatusCode::FOUND
            | StatusCode::SEE_OTHER
            | StatusCode::TEMPORARY_REDIRECT
            | StatusCode::PERMANENT_REDIRECT
    )
}
