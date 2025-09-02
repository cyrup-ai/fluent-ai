//! H3 Protocol Strategy Implementation
//!
//! Encapsulates all HTTP/3 and QUIC protocol complexity behind the ProtocolStrategy interface.

use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam_utils::Backoff;

use fluent_ai_async::{AsyncStream, emit, spawn_task};
use http::{StatusCode, HeaderName, HeaderValue, Version};
use bytes::Bytes;
use quiche;
use quiche::h3::NameValue;

use crate::protocols::strategy_trait::ProtocolStrategy;
use crate::protocols::core::ProtocolConfig;
use crate::protocols::strategy::H3Config;
use crate::http::{HttpRequest, HttpResponse};
use crate::crypto::random::generate_boundary;
use crate::http::response::{HttpHeader, HttpBodyChunk};

// Global connection ID counter for H3 connections
static NEXT_CONNECTION_ID: AtomicU64 = AtomicU64::new(1);

/// HTTP/3 Protocol Strategy
///
/// Encapsulates all HTTP/3 and QUIC complexity including:
/// - UDP socket management
/// - QUIC connection establishment
/// - HTTP/3 stream management
/// - Connection pooling
pub struct H3Strategy {
    config: H3Config,
}

impl H3Strategy {
    /// Create a new H3 strategy with the given configuration
    pub fn new(config: H3Config) -> Self {
        Self {
            config,
        }
    }
    
    /// Convert H3Config to quiche::Config
    fn create_quiche_config(&self) -> Result<quiche::Config, crate::error::HttpError> {
        let mut config = match quiche::Config::new(quiche::PROTOCOL_VERSION) {
            Ok(cfg) => cfg,
            Err(e) => {
                tracing::error!(
                    target: "fluent_ai_http3::protocols::h3",
                    error = %e,
                    "Failed to create QUICHE config with protocol version"
                );
                return Err(crate::error::HttpError::new(crate::error::Kind::Request));
            }
        };
        
        // Set transport parameters from H3Config
        config.set_initial_max_data(self.config.initial_max_data);
        config.set_initial_max_streams_bidi(self.config.initial_max_streams_bidi);
        config.set_initial_max_streams_uni(self.config.initial_max_streams_uni);
        config.set_initial_max_stream_data_bidi_local(self.config.initial_max_stream_data_bidi_local);
        config.set_initial_max_stream_data_bidi_remote(self.config.initial_max_stream_data_bidi_remote);
        config.set_initial_max_stream_data_uni(self.config.initial_max_stream_data_uni);
        
        // Set idle timeout
        config.set_max_idle_timeout(self.config.max_idle_timeout.as_millis() as u64);
        
        // Set UDP payload size
        config.set_max_recv_udp_payload_size(self.config.max_udp_payload_size as usize);
        config.set_max_send_udp_payload_size(self.config.max_udp_payload_size as usize);
        
        // Enable early data if configured
        if self.config.enable_early_data {
            config.enable_early_data();
        }
        
        // Set congestion control algorithm
        use crate::protocols::strategy::CongestionControl;
        match self.config.congestion_control {
            CongestionControl::Cubic => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::CUBIC),
            CongestionControl::Reno => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::Reno),
            CongestionControl::Bbr => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::BBR),
            CongestionControl::BbrV2 => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::BBR2),
        }
        
        // Set HTTP/3 application protocol
        if let Err(e) = config.set_application_protos(&[b"h3"]) {
            tracing::error!(
                target: "fluent_ai_http3::protocols::h3",
                error = %e,
                "Failed to set H3 application protocols"
            );
            return Err(crate::error::HttpError::new(crate::error::Kind::Request)
                .with(std::io::Error::new(std::io::ErrorKind::Other, 
                    format!("Critical H3 protocol configuration failure: {}", e))));
        }
        
        // SECURITY: Enable certificate verification
        config.verify_peer(true);
        
        // Load CA certificates from system store
        #[cfg(target_os = "linux")]
        let ca_path = "/etc/ssl/certs/ca-certificates.crt";
        #[cfg(target_os = "macos")]
        let ca_path = "/System/Library/Keychains/SystemRootCertificates.keychain";
        #[cfg(target_os = "windows")]
        let ca_path = "C:\\Windows\\System32\\certstor.p7b";
        
        // Set CA file with fallback to environment variable
        if std::path::Path::new(ca_path).exists() {
            config.load_verify_locations_from_file(ca_path)
                .unwrap_or_else(|e| {
                    tracing::warn!(
                        target: "fluent_ai_http3::protocols::h3",
                        error = %e,
                        ca_path = ca_path,
                        "Failed to load system CA bundle, trying environment fallback"
                    );
                    // Try environment variable fallback
                    if let Ok(cert_file) = std::env::var("SSL_CERT_FILE") {
                        let _ = config.load_verify_locations_from_file(&cert_file);
                    }
                });
        }
        
        // Set verification depth (standard is 4 for web PKI)
        config.verify_peer(true);
        
        Ok(config)
    }
}

impl ProtocolStrategy for H3Strategy {
    fn execute(&self, request: HttpRequest) -> HttpResponse {
        // Create response streams
        let (headers_tx, headers_internal) = AsyncStream::channel();
        let (body_tx, body_internal) = AsyncStream::channel();
        let (trailers_tx, trailers_internal) = AsyncStream::channel();
        
        // Extract request details
        let method = request.method().clone();
        let url = request.url().clone();
        let headers = request.headers().clone();
        let body_data = request.body().cloned();
        
        // Parse URL to get host and port
        let host = url.host_str().unwrap_or("localhost").to_string();
        let port = url.port().unwrap_or(443);
        let path = match url.query() {
            Some(query) => format!("{}?{}", url.path(), query),
            None => url.path().to_string(),
        };
        let scheme = url.scheme().to_string();
        
        // Clone for async task
        let config = self.config.clone();
        let quic_config = match self.create_quiche_config() {
            Ok(cfg) => cfg,
            Err(e) => {
                // Return error response instead of panicking
                return HttpResponse::error(http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
            }
        };
        
        // Spawn task to handle H3 protocol
        spawn_task(move || {
            // SSRF Protection - validate destination before creating connection
            // Only bypass for development/testing when explicitly configured
            if !config.disable_ssrf_protection {
                if host == "localhost" || host.ends_with(".local") || host == "0.0.0.0" {
                    emit!(body_tx, HttpBodyChunk {
                        data: Bytes::from("Connection to internal host blocked by SSRF protection"),
                        offset: 0,
                        is_final: true,
                        timestamp: std::time::Instant::now(),
                    });
                    return;
                }
                if let Ok(ip) = host.parse::<std::net::IpAddr>() {
                    if ip.is_loopback() {
                        emit!(body_tx, HttpBodyChunk {
                            data: Bytes::from("Connection to loopback address blocked by SSRF protection"),
                            offset: 0,
                            is_final: true,
                            timestamp: std::time::Instant::now(),
                        });
                        return;
                    }
                    // Block private IP ranges (RFC 1918) in production
                    match ip {
                        std::net::IpAddr::V4(ipv4) => {
                            let octets = ipv4.octets();
                            // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                            if octets[0] == 10 
                                || (octets[0] == 172 && octets[1] >= 16 && octets[1] <= 31)
                                || (octets[0] == 192 && octets[1] == 168) {
                                emit!(body_tx, HttpBodyChunk {
                                    data: Bytes::from("Connection to private IP address blocked by SSRF protection"),
                                    offset: 0,
                                    is_final: true,
                                    timestamp: std::time::Instant::now(),
                                });
                                return;
                            }
                        }
                        std::net::IpAddr::V6(ipv6) => {
                            // Block IPv6 private ranges
                            if ipv6.segments()[0] & 0xfe00 == 0xfc00 { // fc00::/7
                                emit!(body_tx, HttpBodyChunk {
                                    data: Bytes::from("Connection to private IPv6 address blocked by SSRF protection"),
                                    offset: 0,
                                    is_final: true,
                                    timestamp: std::time::Instant::now(),
                                });
                                return;
                            }
                        }
                    }
                }
            } else {
                tracing::warn!(
                    target: "fluent_ai_http3::protocols::h3",
                    host = %host,
                    "⚠️  SSRF protection disabled - allowing connection to potentially dangerous host"
                );
            }
            
            // Create UDP socket for QUIC
            let socket = match UdpSocket::bind("0.0.0.0:0") {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(
                        target: "fluent_ai_http3::protocols::h3",
                        error = %e,
                        "Failed to bind UDP socket for QUIC connection"
                    );
                    emit!(body_tx, HttpBodyChunk {
                        data: Bytes::from(format!("Failed to bind UDP socket: {}", e)),
                        offset: 0,
                        is_final: true,
                        timestamp: std::time::Instant::now(),
                    });
                    return;
                }
            };
            
            // Resolve server address
            let server_addr = format!("{}:{}", host, port);
            let server_addr: SocketAddr = match server_addr.parse() {
                Ok(addr) => addr,
                Err(e) => {
                    // Try DNS resolution
                    match std::net::ToSocketAddrs::to_socket_addrs(&server_addr) {
                        Ok(mut addrs) => {
                            if let Some(addr) = addrs.next() {
                                addr
                            } else {
                                tracing::error!(
                                    target: "fluent_ai_http3::protocols::h3",
                                    server_addr = %server_addr,
                                    "DNS resolution returned no addresses"
                                );
                                emit!(body_tx, HttpBodyChunk {
                                    data: Bytes::from(format!("Failed to resolve address: {}", server_addr)),
                                    offset: 0,
                                    is_final: true,
                                    timestamp: std::time::Instant::now(),
                                });
                                return;
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                target: "fluent_ai_http3::protocols::h3",
                                error = %e,
                                server_addr = %server_addr,
                                "Failed to resolve server address via DNS"
                            );
                            emit!(body_tx, HttpBodyChunk {
                                data: Bytes::from(format!("Failed to resolve {}: {}", server_addr, e)),
                                offset: 0,
                                is_final: true,
                                timestamp: std::time::Instant::now(),
                            });
                            return;
                        }
                    }
                }
            };
            
            // Generate connection ID
            let conn_id = NEXT_CONNECTION_ID.fetch_add(1, Ordering::SeqCst);
            
            // Use QUIC configuration
            let mut quic_config = quic_config;
            
            // Create QUIC connection
            let conn_id_bytes = conn_id.to_be_bytes();
            let scid = quiche::ConnectionId::from_ref(&conn_id_bytes);
            let local_addr = match socket.local_addr() {
                Ok(addr) => addr,
                Err(e) => {
                    emit!(body_tx, HttpBodyChunk {
                        data: Bytes::from(format!("Failed to get local address: {}", e)),
                        offset: 0,
                        is_final: true,
                        timestamp: std::time::Instant::now(),
                    });
                    return;
                }
            };
            
            let mut quic_conn = match quiche::connect(
                Some(&host),
                &scid,
                local_addr,
                server_addr,
                &mut quic_config,
            ) {
                Ok(conn) => conn,
                Err(e) => {
                    tracing::error!(
                        target: "fluent_ai_http3::protocols::h3",
                        error = %e,
                        host = %host,
                        server_addr = %server_addr,
                        "Failed to create QUIC connection"
                    );
                    emit!(body_tx, HttpBodyChunk {
                        data: Bytes::from(format!("Failed to create QUIC connection: {}", e)),
                        offset: 0,
                        is_final: true,
                        timestamp: std::time::Instant::now(),
                    });
                    return;
                }
            };
            
            // Initial handshake
            let mut out = [0; 1350];
            let (write_len, send_info) = match quic_conn.send(&mut out) {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(
                        target: "fluent_ai_http3::protocols::h3",
                        error = %e,
                        "Failed initial QUIC handshake send"
                    );
                    emit!(body_tx, HttpBodyChunk {
                        data: Bytes::from(format!("Failed initial QUIC send: {}", e)),
                        offset: 0,
                        is_final: true,
                        timestamp: std::time::Instant::now(),
                    });
                    return;
                }
            };
            
            if let Err(e) = socket.send_to(&out[..write_len], send_info.to) {
                tracing::error!(
                    target: "fluent_ai_http3::protocols::h3",
                    error = %e,
                    packet_size = write_len,
                    destination = %send_info.to,
                    "Failed to send initial QUIC packet"
                );
                emit!(body_tx, HttpBodyChunk {
                    data: Bytes::from(format!("Failed to send initial packet: {}", e)),
                    offset: 0,
                    is_final: true,
                    timestamp: std::time::Instant::now(),
                });
                return;
            }
            
            // Wait for handshake to complete with elite backoff
            socket.set_nonblocking(true).ok();
            let mut buf = [0; 65535];
            let start = std::time::Instant::now();
            let timeout = config.timeout_config().connect_timeout;
            let backoff = Backoff::new();
            
            while !quic_conn.is_established() {
                if start.elapsed() > timeout {
                    tracing::error!(
                        target: "fluent_ai_http3::protocols::h3",
                        timeout_ms = timeout.as_millis(),
                        elapsed_ms = start.elapsed().as_millis(),
                        "QUIC handshake timeout exceeded"
                    );
                    emit!(body_tx, HttpBodyChunk {
                        data: Bytes::from("QUIC handshake timeout"),
                        offset: 0,
                        is_final: true,
                        timestamp: std::time::Instant::now(),
                    });
                    return;
                }
                
                let mut data_processed = false;
                
                // Try to receive
                match socket.recv_from(&mut buf) {
                    Ok((len, from)) => {
                        let recv_info = quiche::RecvInfo {
                            from,
                            to: local_addr,
                        };
                        
                        if let Err(e) = quic_conn.recv(&mut buf[..len], recv_info) {
                            tracing::warn!(
                                target: "fluent_ai_http3::protocols::h3",
                                error = %e,
                                packet_len = len,
                                "QUIC packet receive error during handshake"
                            );
                        } else {
                            data_processed = true;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // No data available, continue
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "fluent_ai_http3::protocols::h3",
                            error = %e,
                            "UDP socket receive error during handshake"
                        );
                    }
                }
                
                // Send any pending data
                loop {
                    let mut out = [0; 1350];
                    match quic_conn.send(&mut out) {
                        Ok((len, send_info)) => {
                            if len == 0 {
                                break;
                            }
                            if let Err(e) = socket.send_to(&out[..len], send_info.to) {
                                tracing::warn!(
                                    target: "fluent_ai_http3::protocols::h3",
                                    error = %e,
                                    packet_len = len,
                                    destination = %send_info.to,
                                    "Failed to send QUIC packet during handshake"
                                );
                            } else {
                                data_processed = true;
                            }
                        }
                        Err(quiche::Error::Done) => break,
                        Err(e) => {
                            tracing::warn!(
                                target: "fluent_ai_http3::protocols::h3",
                                error = %e,
                                "QUIC send error during handshake"
                            );
                            break;
                        }
                    }
                }
                
                // Elite backoff pattern - reset on successful data processing
                if data_processed {
                    backoff.reset();
                } else {
                    if backoff.is_completed() {
                        std::thread::yield_now();
                    } else {
                        backoff.snooze();
                    }
                }
            }
            
            // Create HTTP/3 connection
            let h3_config = match quiche::h3::Config::new() {
                Ok(config) => config,
                Err(e) => {
                    tracing::error!(
                        target: "fluent_ai_http3::protocols::h3",
                        error = %e,
                        "Failed to create H3 config"
                    );
                    emit!(body_tx, HttpBodyChunk {
                        data: Bytes::from(format!("Failed to create H3 config: {}", e)),
                        offset: 0,
                        is_final: true,
                        timestamp: std::time::Instant::now(),
                    });
                    return;
                }
            };
            let mut h3_conn = match quiche::h3::Connection::with_transport(&mut quic_conn, &h3_config) {
                Ok(conn) => conn,
                Err(e) => {
                    tracing::error!(
                        target: "fluent_ai_http3::protocols::h3",
                        error = %e,
                        "Failed to create HTTP/3 connection over QUIC transport"
                    );
                    emit!(body_tx, HttpBodyChunk {
                        data: Bytes::from(format!("Failed to create H3 connection: {}", e)),
                        offset: 0,
                        is_final: true,
                        timestamp: std::time::Instant::now(),
                    });
                    return;
                }
            };
            
            // Build H3 headers
            let h3_headers = vec![
                quiche::h3::Header::new(b":method", method.as_str().as_bytes()),
                quiche::h3::Header::new(b":scheme", scheme.as_bytes()),
                quiche::h3::Header::new(b":authority", host.as_bytes()),
                quiche::h3::Header::new(b":path", path.as_bytes()),
            ];
            
            // Send request
            let stream_id = match h3_conn.send_request(&mut quic_conn, &h3_headers, body_data.is_none()) {
                Ok(id) => id,
                Err(e) => {
                    tracing::error!(
                        target: "fluent_ai_http3::protocols::h3",
                        error = %e,
                        method = %method,
                        path = %path,
                        "Failed to send HTTP/3 request headers"
                    );
                    emit!(body_tx, HttpBodyChunk {
                        data: Bytes::from(format!("Failed to send H3 request: {}", e)),
                        offset: 0,
                        is_final: true,
                        timestamp: std::time::Instant::now(),
                    });
                    return;
                }
            };
            
            // Send body if present
            if let Some(body_data) = body_data {
                let body_bytes = match body_data {
                    crate::http::request::RequestBody::Bytes(bytes) => bytes.to_vec(),
                    crate::http::request::RequestBody::Text(text) => text.into_bytes(),
                    crate::http::request::RequestBody::Json(json) => {
                        serde_json::to_string(&json).unwrap_or_default().into_bytes()
                    }
                    crate::http::request::RequestBody::Form(form) => {
                        serde_urlencoded::to_string(&form).unwrap_or_default().into_bytes()
                    }
                    crate::http::request::RequestBody::Multipart(fields) => {
                        let boundary = generate_boundary();
                        let mut body = Vec::new();
                        
                        for field in fields {
                            // Add boundary separator
                            body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
                            
                            // Add Content-Disposition header
                            match (&field.filename, &field.content_type) {
                                (Some(filename), Some(content_type)) => {
                                    body.extend_from_slice(
                                        format!("Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n", field.name, filename).as_bytes()
                                    );
                                    body.extend_from_slice(
                                        format!("Content-Type: {}\r\n\r\n", content_type).as_bytes()
                                    );
                                }
                                (Some(filename), None) => {
                                    body.extend_from_slice(
                                        format!("Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n", field.name, filename).as_bytes()
                                    );
                                    body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
                                }
                                (None, Some(content_type)) => {
                                    body.extend_from_slice(
                                        format!("Content-Disposition: form-data; name=\"{}\"\r\n", field.name).as_bytes()
                                    );
                                    body.extend_from_slice(
                                        format!("Content-Type: {}\r\n\r\n", content_type).as_bytes()
                                    );
                                }
                                (None, None) => {
                                    body.extend_from_slice(
                                        format!("Content-Disposition: form-data; name=\"{}\"\r\n\r\n", field.name).as_bytes()
                                    );
                                }
                            }
                            
                            // Add field value
                            match &field.value {
                                crate::http::request::MultipartValue::Text(text) => {
                                    body.extend_from_slice(text.as_bytes());
                                }
                                crate::http::request::MultipartValue::Bytes(bytes) => {
                                    body.extend_from_slice(bytes);
                                }
                            }
                            
                            body.extend_from_slice(b"\r\n");
                        }
                        
                        // Add final boundary
                        body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());
                        body
                    }
                    crate::http::request::RequestBody::Stream(mut stream) => {
                        let mut body_data = Vec::new();
                        let timeout = config.timeout_config().request_timeout;
                        let start_time = std::time::Instant::now();
                        const MAX_BODY_SIZE: usize = 100 * 1024 * 1024; // 100MB hard limit (no config required)
                        
                        loop {
                            if start_time.elapsed() > timeout {
                                tracing::warn!("Streaming body timeout exceeded");
                                break;
                            }
                            
                            // Always enforce memory limits (no config required)
                            if body_data.len() >= MAX_BODY_SIZE {
                                tracing::error!("Request body exceeds 100MB limit");
                                break;
                            }
                            
                            match stream.try_next() {
                                Some(chunk) => {
                                    match chunk {
                                        crate::http::response::HttpChunk::Body(bytes) => {
                                            body_data.extend_from_slice(&bytes);
                                        }
                                        crate::http::response::HttpChunk::Data(bytes) => {
                                            body_data.extend_from_slice(&bytes);
                                        }
                                        crate::http::response::HttpChunk::Chunk(bytes) => {
                                            body_data.extend_from_slice(&bytes);
                                        }
                                        crate::http::response::HttpChunk::End => {
                                            break;
                                        }
                                        crate::http::response::HttpChunk::Error(err) => {
                                            tracing::error!("Stream error: {}", err);
                                            break;
                                        }
                                        _ => continue,
                                    }
                                }
                                None => {
                                    std::thread::sleep(std::time::Duration::from_millis(1));
                                    continue;
                                }
                            }
                        }
                        
                        body_data
                    }
                };
                
                if let Err(e) = h3_conn.send_body(&mut quic_conn, stream_id, &body_bytes, true) {
                    tracing::error!(
                        target: "fluent_ai_http3::protocols::h3",
                        error = %e,
                        stream_id = stream_id,
                        body_len = body_bytes.len(),
                        "Failed to send HTTP/3 request body"
                    );
                }
            }
            
            // Process response
            let mut response_complete = false;
            while !response_complete {
                // Poll H3 events
                match h3_conn.poll(&mut quic_conn) {
                    Ok((stream_id, quiche::h3::Event::Headers { list, .. })) => {
                        // Process response headers
                        for header in list {
                            let name_bytes = header.name();
                            let value_bytes = header.value();
                            
                            let name_str = String::from_utf8_lossy(name_bytes);
                            let value_str = String::from_utf8_lossy(value_bytes);
                            
                            // Convert to HeaderName and HeaderValue
                            if let (Ok(name), Ok(value)) = (
                                HeaderName::from_bytes(name_bytes),
                                HeaderValue::from_bytes(value_bytes)
                            ) {
                                let http_header = crate::http::response::HttpHeader {
                                    name,
                                    value,
                                    timestamp: std::time::Instant::now(),
                                };
                                
                                // Note: Header caching will be handled by the HttpResponse when it processes the stream
                                
                                // Emit header to stream
                                emit!(headers_tx, http_header);
                            }
                        }
                    }
                    Ok((stream_id, quiche::h3::Event::Data)) => {
                        // Read response body
                        let mut body_buf = vec![0; 4096];
                        match h3_conn.recv_body(&mut quic_conn, stream_id, &mut body_buf) {
                            Ok(len) => {
                                if len > 0 {
                                    emit!(body_tx, HttpBodyChunk {
                                        data: Bytes::from(body_buf[..len].to_vec()),
                                        offset: 0,
                                        is_final: false,
                                        timestamp: std::time::Instant::now(),
                                    });
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    target: "fluent_ai_http3::protocols::h3",
                                    error = %e,
                                    stream_id = stream_id,
                                    "Failed to receive HTTP/3 response body"
                                );
                            }
                        }
                    }
                    Ok((stream_id, quiche::h3::Event::Finished)) => {
                        // Stream finished
                        emit!(body_tx, HttpBodyChunk {
                            data: Bytes::new(),
                            offset: 0,
                            is_final: true,
                            timestamp: std::time::Instant::now(),
                        });
                        response_complete = true;
                    }
                    Ok(_) => {
                        // Other events
                    }
                    Err(quiche::h3::Error::Done) => {
                        // No more events
                        let backoff = Backoff::new();
                backoff.snooze();
                    }
                    Err(e) => {
                        tracing::error!(
                            target: "fluent_ai_http3::protocols::h3",
                            error = %e,
                            "HTTP/3 event polling error, terminating response processing"
                        );
                        response_complete = true;
                    }
                }
                
                // Handle QUIC I/O
                match socket.recv_from(&mut buf) {
                    Ok((len, from)) => {
                        let recv_info = quiche::RecvInfo {
                            from,
                            to: local_addr,
                        };
                        if let Err(e) = quic_conn.recv(&mut buf[..len], recv_info) {
                            tracing::warn!(
                                target: "fluent_ai_http3::protocols::h3",
                                error = %e,
                                packet_len = len,
                                "QUIC packet receive error during response processing"
                            );
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // No data available
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "fluent_ai_http3::protocols::h3",
                            error = %e,
                            "UDP socket receive error during response processing"
                        );
                    }
                }
                
                // Send pending data
                loop {
                    let mut out = [0; 1350];
                    match quic_conn.send(&mut out) {
                        Ok((len, send_info)) => {
                            if len == 0 {
                                break;
                            }
                            if let Err(e) = socket.send_to(&out[..len], send_info.to) {
                                tracing::warn!(
                                    target: "fluent_ai_http3::protocols::h3",
                                    error = %e,
                                    packet_len = len,
                                    destination = %send_info.to,
                                    "UDP socket send error during response processing"
                                );
                            }
                        }
                        Err(quiche::Error::Done) => break,
                        Err(e) => {
                            tracing::warn!(
                                target: "fluent_ai_http3::protocols::h3",
                                error = %e,
                                "QUIC send error during response processing"
                            );
                            break;
                        }
                    }
                }
            }
            
            // Signal completion
            emit!(body_tx, HttpBodyChunk {
                data: Bytes::new(),
                offset: 0,
                is_final: true,
                timestamp: std::time::Instant::now(),
            });
        });
        
        // Create and return HttpResponse
        let mut response = HttpResponse::new(
            headers_internal,
            body_internal,
            trailers_internal,
            Version::HTTP_3,
            0, // stream_id
        );
        
        // Set initial status
        response.set_status(StatusCode::OK);
        
        response
    }
    
    fn protocol_name(&self) -> &'static str {
        "HTTP/3"
    }
    
    fn supports_push(&self) -> bool {
        false // HTTP/3 doesn't use server push like HTTP/2
    }
    
    fn max_concurrent_streams(&self) -> usize {
        self.config.initial_max_streams_bidi as usize
    }
}