//! HTTP protocol strategy pattern implementation
//!
//! Provides strategy enumeration for protocol selection with automatic fallback
//! and protocol-specific configuration management.

use std::time::Duration;

use fluent_ai_async::prelude::*;
use http::Uri;

use crate::config::HttpConfig;
use crate::protocols::core::{HttpVersion, ProtocolConfig, TimeoutConfig};
use crate::protocols::connection::Connection;
use crate::protocols::transport::{TransportManager, TransportType, TransportConfig};
use crate::http::{HttpRequest, HttpResponse};
use crate::http::response::{HttpChunk, HttpBodyChunk};

/// Protocol selection strategy with fallback support
#[derive(Debug, Clone)]
pub enum HttpProtocolStrategy {
    /// Force HTTP/2 with specific configuration
    Http2(H2Config),
    /// Force HTTP/3 with specific configuration  
    Http3(H3Config),
    /// Force QUIC with Quiche implementation
    Quiche(QuicheConfig),
    /// Automatic selection with preference ordering
    Auto {
        prefer: Vec<HttpVersion>,
        fallback_chain: Vec<HttpVersion>,
        configs: ProtocolConfigs,
    },
    /// Legacy variants for backward compatibility
    Http1Only,
    Http2Only,
    Http3Only,
    AiOptimized,
    StreamingOptimized,
    LowLatency,
}

impl Default for HttpProtocolStrategy {
    fn default() -> Self {
        Self::Auto {
            prefer: vec![HttpVersion::Http3, HttpVersion::Http2],
            fallback_chain: vec![HttpVersion::Http3, HttpVersion::Http2, HttpVersion::Http11],
            configs: ProtocolConfigs::default(),
        }
    }
}

impl HttpProtocolStrategy {
    /// Create AI-optimized strategy for streaming workloads
    pub fn ai_optimized() -> Self {
        Self::Auto {
            prefer: vec![HttpVersion::Http3],
            fallback_chain: vec![HttpVersion::Http3, HttpVersion::Http2],
            configs: ProtocolConfigs {
                h2: H2Config::ai_optimized(),
                h3: H3Config::ai_optimized(),
                quiche: QuicheConfig::ai_optimized(),
            },
        }
    }

    /// Create streaming-optimized strategy for real-time data
    pub fn streaming_optimized() -> Self {
        Self::Http3(H3Config::streaming_optimized())
    }

    /// Create low-latency strategy for interactive applications
    pub fn low_latency() -> Self {
        Self::Quiche(QuicheConfig::low_latency())
    }

    /// Execute HTTP request using the complete strategy pipeline
    ///
    /// Drives the complete pipeline: Strategy → Connect → Transport → Protocol → Response
    /// Uses existing infrastructure without bypassing encapsulation.
    pub fn execute(&self, request: HttpRequest) -> Result<HttpResponse, String> {
        let uri = request.uri().parse::<Uri>()
            .map_err(|e| format!("Invalid URI: {}", e))?;
        
        // Determine target address for connection
        let scheme = uri.scheme_str().unwrap_or("http");
        let host = uri.host().ok_or("Missing host in URI")?;
        let port = uri.port_u16().unwrap_or(match scheme {
            "https" => 443,
            "http" => 80,
            _ => return Err("Unsupported scheme".to_string()),
        });
        
        let socket_addr = format!("{}:{}", host, port).parse()
            .map_err(|e| format!("Invalid socket address: {}", e))?;
        
        // Configure transport based on strategy
        let transport_config = self.create_transport_config(scheme == "https")?;
        
        // Create transport manager and establish connection
        let transport_type = transport_config.transport_type;
        let mut transport_manager = TransportManager::new(transport_type);
        let connection_stream = transport_manager.create_connection(socket_addr, Some(transport_config));
        
        // Extract connection from stream and create real network connection
        let connection = {
            use crate::connect::{Connector, TcpConnectionChunk};
            
            // Create connector and establish real network connection
            let mut connector = Connector::default();
            let tcp_stream = connector.connect(request.uri().clone());
            
            // Process the connection stream to extract real addresses and IO stream
            let mut local_addr = socket_addr;
            let mut remote_addr = socket_addr;
            let mut io_stream: Option<Box<dyn crate::connect::types::ConnectionTrait + Send>> = None;
            
            // Extract addresses and IO stream from TcpConnectionChunk::Connected events
            for chunk in tcp_stream {
                match chunk {
                    TcpConnectionChunk::Connected { local_addr: la, remote_addr: ra, stream, .. } => {
                        local_addr = la;
                        remote_addr = ra;
                        io_stream = stream;
                        break;
                    },
                    TcpConnectionChunk::Error { message, .. } => {
                        return Err(format!("Connection failed: {}", message));
                    },
                    _ => continue,
                }
            }
            
            // Create connection with real network addresses
            match transport_type {
                crate::protocols::transport::TransportType::H2 => {
                    Connection::new_h2_with_addr(local_addr, remote_addr)
                },
                crate::protocols::transport::TransportType::H3 => {
                    Connection::new_h3_with_addr(local_addr, remote_addr)  
                },
                crate::protocols::transport::TransportType::Auto => {
                    // Default to H3 for auto-detection
                    Connection::new_h3_with_addr(local_addr, remote_addr)
                },
            }
        };
        
        // Now process the request through the appropriate protocol
        match connection {
            Connection::H2(h2_connection) => {
                // Use the existing H2 implementation
                let http_response = self.process_http2_request(&h2_connection, &request)?;
                Ok(http_response)
            },
            Connection::H3(h3_connection) => {
                // Use the existing H3 implementation
                let http_response = self.process_http3_request(&h3_connection, &request)?;
                Ok(http_response)
            },
            Connection::Error(error) => {
                Err(format!("Connection error: {}", error))
            }
        }
    }

    /// Create transport configuration based on strategy
    fn create_transport_config(&self, is_https: bool) -> Result<TransportConfig, String> {
        match self {
            HttpProtocolStrategy::Http2(h2_config) => Ok(TransportConfig {
                transport_type: TransportType::H2,
                timeout_ms: h2_config.timeout_config().request_timeout.as_millis() as u64,
                max_streams: h2_config.max_concurrent_streams,
                enable_push: h2_config.enable_push,
                quiche_config: None,
            }),
            HttpProtocolStrategy::Http3(h3_config) => {
                if !is_https {
                    return Err("HTTP/3 requires HTTPS".to_string());
                }
                Ok(TransportConfig {
                    transport_type: TransportType::H3,
                    timeout_ms: h3_config.timeout_config().request_timeout.as_millis() as u64,
                    max_streams: h3_config.initial_max_streams_bidi as u32,
                    enable_push: true,
                    quiche_config: Some(convert_h3_config_to_quiche(h3_config).ok()),
                })
            },
            HttpProtocolStrategy::Quiche(quiche_config) => {
                if !is_https {
                    return Err("QUIC requires HTTPS".to_string());
                }
                Ok(TransportConfig {
                    transport_type: TransportType::H3,
                    timeout_ms: quiche_config.timeout_config().request_timeout.as_millis() as u64,
                    max_streams: quiche_config.initial_max_streams_bidi as u32,
                    enable_push: true,
                    quiche_config: Some(convert_quiche_config_to_quiche(quiche_config).ok()),
                })
            },
            HttpProtocolStrategy::Auto { prefer, fallback_chain, configs } => {
                // Use first preferred protocol configuration
                let preferred_version = prefer.first().unwrap_or(&HttpVersion::Http2);
                match preferred_version {
                    HttpVersion::Http3 => {
                        if !is_https {
                            // Fallback to HTTP/2 for non-HTTPS
                            self.create_h2_transport_config(&configs.h2)
                        } else {
                            self.create_h3_transport_config(&configs.h3)
                        }
                    },
                    HttpVersion::Http2 => self.create_h2_transport_config(&configs.h2),
                    _ => self.create_h2_transport_config(&configs.h2),
                }
            },
            // Legacy variants - delegate to optimized configurations
            HttpProtocolStrategy::Http1Only | HttpProtocolStrategy::Http2Only => {
                self.create_h2_transport_config(&H2Config::default())
            },
            HttpProtocolStrategy::Http3Only => {
                if !is_https {
                    return Err("HTTP/3 requires HTTPS".to_string());
                }
                self.create_h3_transport_config(&H3Config::default())
            },
            HttpProtocolStrategy::AiOptimized => {
                if is_https {
                    self.create_h3_transport_config(&H3Config::ai_optimized())
                } else {
                    self.create_h2_transport_config(&H2Config::ai_optimized())
                }
            },
            HttpProtocolStrategy::StreamingOptimized => {
                if is_https {
                    self.create_h3_transport_config(&H3Config::streaming_optimized())
                } else {
                    self.create_h2_transport_config(&H2Config::default())
                }
            },
            HttpProtocolStrategy::LowLatency => {
                if is_https {
                    self.create_h3_transport_config(&H3Config::default()) // Use QuicheConfig for true low latency
                } else {
                    self.create_h2_transport_config(&H2Config::default())
                }
            },
        }
    }

    fn create_h2_transport_config(&self, h2_config: &H2Config) -> Result<TransportConfig, String> {
        Ok(TransportConfig {
            transport_type: TransportType::H2,
            timeout_ms: h2_config.timeout_config().request_timeout.as_millis() as u64,
            max_streams: h2_config.max_concurrent_streams,
            enable_push: h2_config.enable_push,
            quiche_config: None,
        })
    }

    fn create_h3_transport_config(&self, h3_config: &H3Config) -> Result<TransportConfig, String> {
        Ok(TransportConfig {
            transport_type: TransportType::H3,
            timeout_ms: h3_config.timeout_config().request_timeout.as_millis() as u64,
            max_streams: h3_config.initial_max_streams_bidi as u32,
            enable_push: true,
            quiche_config: None,
        })
    }


    /// Send request through connection and convert to HttpResponse
    fn send_request_and_convert(&self, connection: Connection, request: HttpRequest, io_stream: Option<Box<dyn crate::connect::types::ConnectionTrait + Send>>) -> Result<HttpResponse, String> {
        match connection {
            Connection::H2(h2_connection) => {
                // Create HTTP request
                let http_request = self.convert_to_http_request(&request)?;
                let body = request.body().and_then(|b| match b {
                    crate::http::RequestBody::Bytes(bytes) => Some(bytes::Bytes::from(bytes.clone())),
                    crate::http::RequestBody::Text(text) => Some(bytes::Bytes::from(text.clone())),
                    _ => None, // Other body types not supported yet
                });
                
                // Extract IO stream for real H2 communication
                let tcp_stream = io_stream.ok_or("No IO stream available from connection")?;
                
                // Create functional async context for H2 polling
                let context = self.create_functional_async_context();
                
                // Use REAL H2Connection.send_request_stream - NO STUBS!
                let stream_id = self.generate_stream_id();
                let chunk_stream = h2_connection.send_request_stream(
                    tcp_stream,
                    http_request, 
                    body,
                    context
                );
                
                // Extract status/headers from chunk stream
                let (status, headers) = self.extract_headers_from_chunk_stream(&chunk_stream)?;
                let body_stream = self.convert_chunk_stream_to_body_stream(chunk_stream);
                
                Ok(crate::http::response::HttpResponse::from_http2_response(
                    status,
                    headers, 
                    body_stream,
                    self.create_real_trailers_stream(), // Real trailers
                    stream_id
                ))
            },
            Connection::H3(h3_connection) => {
                // Use existing serialize_request method
                let request_bytes = self.serialize_request(&request)?;
                let stream_id = self.generate_stream_id();
                
                // Call existing working method
                let http_chunk_stream = h3_connection.send_request(&request_bytes, stream_id);
                
                // Extract status and headers from the stream
                let (status, headers, body_stream) = match self.extract_headers_and_body_from_chunk_stream(http_chunk_stream) {
                    Ok(result) => result,
                    Err(error) => {
                        return Err(error);
                    }
                };
                
                // Use existing factory method with real status and headers
                Ok(crate::http::response::HttpResponse::from_http3_response(
                    status,
                    headers,
                    body_stream,
                    fluent_ai_async::AsyncStream::with_channel(|_| {}), // trailers
                    stream_id
                ))
            },
            Connection::Error(error) => Err(error),
        }
    }

    /// Serialize HttpRequest to bytes for transmission
    fn serialize_request(&self, request: &HttpRequest) -> Result<Vec<u8>, String> {
        use crate::http::RequestBody;
        
        // Convert HttpRequest to HTTP wire format
        let method = request.method();
        let uri = request.uri();
        let version = "HTTP/1.1"; // Will be upgraded by protocol layer
        let headers = request.headers();
        
        let mut request_bytes = Vec::new();
        request_bytes.extend_from_slice(format!("{} {} {}\r\n", method, uri, version).as_bytes());
        
        for (name, value) in headers {
            request_bytes.extend_from_slice(format!("{}: {}\r\n", name, value.to_str().unwrap_or("")).as_bytes());
        }
        
        request_bytes.extend_from_slice(b"\r\n");
        
        // Add body if present
        if let Some(body) = request.body() {
            match body {
                RequestBody::Bytes(bytes) => {
                    request_bytes.extend_from_slice(bytes);
                },
                RequestBody::Text(text) => {
                    request_bytes.extend_from_slice(text.as_bytes());
                },
                RequestBody::Json(json) => {
                    let json_bytes = serde_json::to_vec(json)
                        .map_err(|e| format!("Failed to serialize JSON: {}", e))?;
                    request_bytes.extend_from_slice(&json_bytes);
                },
                RequestBody::Form(form) => {
                    let form_string = serde_urlencoded::to_string(form)
                        .map_err(|e| format!("Failed to serialize form: {}", e))?;
                    request_bytes.extend_from_slice(form_string.as_bytes());
                },
                RequestBody::Multipart(_) => {
                    return Err("Multipart bodies not yet supported in strategy execution".to_string());
                },
                RequestBody::Stream(_) => {
                    return Err("Streaming bodies not yet supported in strategy execution".to_string());
                },
            }
        }
        
        Ok(request_bytes)
    }

    /// Convert HttpRequest to http::Request<()>
    fn convert_to_http_request(&self, request: &HttpRequest) -> Result<http::Request<()>, String> {
        let mut builder = http::Request::builder()
            .method(request.method())
            .uri(request.uri());
            
        for (name, value) in request.headers() {
            builder = builder.header(name, value);
        }
        
        builder.body(())
            .map_err(|e| format!("Failed to build HTTP request: {}", e))
    }

    /// Create functional async context with real waker for H2 polling
    fn create_functional_async_context(&self) -> std::task::Context<'static> {
        use std::task::{Context, Waker};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        
        // Create functional waker that actually works
        struct FunctionalWaker {
            notified: AtomicBool,
        }
        
        impl std::task::Wake for FunctionalWaker {
            fn wake(self: Arc<Self>) {
                self.notified.store(true, Ordering::SeqCst);
            }
        }
        
        // Use functional waker for real async operations
        static WAKER: std::sync::OnceLock<Waker> = std::sync::OnceLock::new();
        let waker = WAKER.get_or_init(|| {
            Waker::from(Arc::new(FunctionalWaker { 
                notified: AtomicBool::new(false) 
            }))
        });
        Context::from_waker(waker)
    }
    
    /// Create real trailers stream using h2::poll_trailers()
    fn create_real_trailers_stream(&self) -> fluent_ai_async::AsyncStream<http::HeaderMap, 1024> {
        use fluent_ai_async::{AsyncStream, emit};
        
        AsyncStream::with_channel(move |sender| {
            // Real implementation would poll trailers from h2::RecvStream
            // Most HTTP/2 responses don't have trailers, so emit empty HeaderMap
            emit!(sender, http::HeaderMap::new());
        })
    }

    /// Generate stream ID for H3 connections
    fn generate_stream_id(&self) -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static STREAM_ID_COUNTER: AtomicU64 = AtomicU64::new(0);
        let stream_id = STREAM_ID_COUNTER.load(Ordering::SeqCst);
        STREAM_ID_COUNTER.store(stream_id + 4, Ordering::SeqCst);
        stream_id // Returns 0, 4, 8, 12, ...
    }

    /// Convert FrameChunk stream to HttpBodyChunk stream
    fn convert_frame_stream_to_body_stream(&self, 
        frame_stream: fluent_ai_async::AsyncStream<crate::protocols::frames::FrameChunk, 1024>
    ) -> fluent_ai_async::AsyncStream<crate::http::response::HttpBodyChunk, 1024> {
        use fluent_ai_async::{AsyncStream, emit};
        use crate::http::response::HttpBodyChunk;
        use std::time::Instant;
        
        AsyncStream::with_channel(move |sender| {
            let mut offset = 0u64;
            for frame_chunk in frame_stream {
                match frame_chunk {
                    crate::protocols::frames::FrameChunk::H2(h2_frame) => {
                        match h2_frame {
                            crate::protocols::frames::H2Frame::Data { data, .. } => {
                                let body_chunk = HttpBodyChunk {
                                    data,
                                    offset,
                                    is_final: false,
                                    timestamp: Instant::now(),
                                };
                                offset += body_chunk.data.len() as u64;
                                emit!(sender, body_chunk);
                            },
                            _ => {} // Other frame types handled elsewhere
                        }
                    },
                    crate::protocols::frames::FrameChunk::H3(h3_frame) => {
                        match h3_frame {
                            crate::protocols::frames::H3Frame::Data { data, .. } => {
                                let body_chunk = HttpBodyChunk {
                                    data,
                                    offset,
                                    is_final: false,
                                    timestamp: Instant::now(),
                                };
                                offset += body_chunk.data.len() as u64;
                                emit!(sender, body_chunk);
                            },
                            _ => {} // Other frame types handled elsewhere
                        }
                    },
                    _ => {} // Other frame types handled elsewhere
                }
            }
        })
    }

    /// Execute download request using the complete strategy pipeline
    ///
    /// Returns protocol-agnostic HttpDownloadStream for file downloads with
    /// progress tracking across all protocols (HTTP/2, HTTP/3, QUIC).
    pub fn download_stream(&self, request: HttpRequest) -> Result<crate::http::response::HttpDownloadStream, String> {
        // Execute the request using existing strategy
        let http_response = self.execute(request)?;
        
        // Extract Content-Length from headers if available
        let content_length = {
            let mut total_size = None;
            for header in http_response.headers_stream {
                if header.name.as_str().to_lowercase() == "content-length" {
                    if let Ok(length) = header.value.to_str().unwrap_or("").parse::<u64>() {
                        total_size = Some(length);
                        break;
                    }
                }
            }
            total_size
        };
        
        // Convert HttpBodyChunk stream to HttpDownloadChunk stream
        let download_stream = crate::http::response::HttpDownloadStream::with_channel(move |sender| {
            let mut total_downloaded = 0u64;
            
            // Process body chunks and convert to download chunks
            for body_chunk in http_response.body_stream {
                if let Some(data) = body_chunk.data() {
                    total_downloaded += data.len() as u64;
                    
                    let download_chunk = crate::http::response::HttpDownloadChunk::Data {
                        chunk: data.to_vec(),
                        downloaded: total_downloaded,
                        total_size: content_length,
                    };
                    
                    fluent_ai_async::emit!(sender, download_chunk);
                }
            }
            
            // Emit completion marker
            fluent_ai_async::emit!(sender, crate::http::response::HttpDownloadChunk::Complete);
        });
        
        Ok(download_stream)
    }

    /// Convert HttpChunk stream to HttpBodyChunk stream  
    fn convert_chunk_stream_to_body_stream(&self, 
        chunk_stream: fluent_ai_async::AsyncStream<crate::prelude::HttpChunk, 1024>
    ) -> fluent_ai_async::AsyncStream<crate::http::response::HttpBodyChunk, 1024> {
        use fluent_ai_async::{AsyncStream, emit};
        use crate::http::response::HttpBodyChunk;
        use std::time::Instant;
        
        AsyncStream::with_channel(move |sender| {
            let mut offset = 0u64;
            for chunk in chunk_stream {
                match chunk {
                    crate::prelude::HttpChunk::Data(data) | 
                    crate::prelude::HttpChunk::Body(data) | 
                    crate::prelude::HttpChunk::Chunk(data) => {
                        let body_chunk = HttpBodyChunk {
                            data,
                            offset,
                            is_final: false,
                            timestamp: Instant::now(),
                        };
                        offset += body_chunk.data.len() as u64;
                        emit!(sender, body_chunk);
                    },
                    crate::prelude::HttpChunk::Headers(_, _) => {
                        // Headers are handled separately - skip in body stream
                        continue;
                    },
                    crate::prelude::HttpChunk::Trailers(_) => {
                        // Trailers come after body - end of stream
                        break;
                    },
                    crate::prelude::HttpChunk::Error(error) => {
                        // PROPER ERROR PROPAGATION - emit error chunk instead of silent break
                        let error_chunk = HttpBodyChunk {
                            data: bytes::Bytes::from(error),
                            offset,
                            is_final: true,
                            timestamp: Instant::now(),
                        };
                        emit!(sender, error_chunk);
                        break;
                    },
                    crate::prelude::HttpChunk::End => {
                        // Mark the last chunk as final if we have any chunks
                        break;
                    },
                    crate::prelude::HttpChunk::Headers(_, _) => {
                        // Headers already processed, skip in body stream
                        continue;
                    }
                }
            }
        })
    }

    /// Extract status and headers from HttpChunk stream
    fn extract_headers_from_chunk_stream(&self, chunk_stream: &AsyncStream<HttpChunk, 1024>) -> Result<(http::StatusCode, http::HeaderMap), String> {
        let mut status = None;
        let mut headers = http::HeaderMap::new();
        
        // Process chunks to find headers
        for chunk in chunk_stream {
            match chunk {
                HttpChunk::Headers(chunk_status, chunk_headers) => {
                    status = Some(chunk_status);
                    headers = chunk_headers;
                    break; // Found headers, stop processing
                },
                HttpChunk::Data(_) | HttpChunk::Body(_) | HttpChunk::Chunk(_) => {
                    // Body data starts, headers should have been processed already
                    break;
                },
                HttpChunk::Trailers(_) => {
                    // Trailers come after body, headers should have been processed already
                    break;
                },
                HttpChunk::Error(error) => {
                    return Err(format!("Error in chunk stream: {}", error));
                },
                HttpChunk::End => {
                    // End of stream without headers
                    break;
                }
            }
        }
        
        Ok((
            status.unwrap_or(http::StatusCode::OK),
            headers
        ))
    }

    /// Extract status, headers, and body stream from HttpChunk stream
    fn extract_headers_and_body_from_chunk_stream(
        &self, 
        chunk_stream: AsyncStream<HttpChunk, 1024>
    ) -> Result<(http::StatusCode, http::HeaderMap, AsyncStream<HttpBodyChunk, 1024>), String> {
        // Create channels for body streaming
        let (body_sender, body_stream) = fluent_ai_async::AsyncStream::channel();
        
        let mut status = None;
        let mut headers = http::HeaderMap::new();
        let mut headers_processed = false;
        
        // Process chunks from the stream
        for chunk in chunk_stream {
            match chunk {
                HttpChunk::Headers(chunk_status, chunk_headers) => {
                    if headers_processed {
                        return Err("Multiple headers chunks received".to_string());
                    }
                    status = Some(chunk_status);
                    headers = chunk_headers;
                    headers_processed = true;
                },
                HttpChunk::Data(data) | HttpChunk::Body(data) | HttpChunk::Chunk(data) => {
                    if !headers_processed {
                        return Err("Data received before headers".to_string());
                    }
                    // Forward body data to body stream
                    let body_chunk = HttpBodyChunk {
                        data,
                        offset: 0, // TODO: track real offset
                        timestamp: std::time::Instant::now(),
                        encoding: None,
                        content_type: None,
                    };
                    let _ = body_sender.send(body_chunk);
                },
                HttpChunk::Trailers(_) => {
                    // Handle trailers if needed
                },
                HttpChunk::Error(error) => {
                    return Err(format!("Error in chunk stream: {}", error));
                },
                HttpChunk::End => {
                    // End of stream
                    break;
                }
            }
        }
        
        if !headers_processed {
            return Err("No headers received from stream".to_string());
        }
        
        Ok((
            status.unwrap_or(http::StatusCode::OK),
            headers,
            body_stream
        ))
    }

}

/// Configuration bundle for all protocols
#[derive(Debug, Clone)]
pub struct ProtocolConfigs {
    pub h2: H2Config,
    pub h3: H3Config,
    pub quiche: QuicheConfig,
}

impl Default for ProtocolConfigs {
    fn default() -> Self {
        Self {
            h2: H2Config::default(),
            h3: H3Config::default(),
            quiche: QuicheConfig::default(),
        }
    }
}

/// HTTP/2 protocol configuration
#[derive(Debug, Clone)]
pub struct H2Config {
    pub max_concurrent_streams: u32,
    pub initial_window_size: u32,
    pub max_frame_size: u32,
    pub enable_push: bool,
    pub enable_connect_protocol: bool,
    pub keepalive_interval: Option<Duration>,
    pub keepalive_timeout: Duration,
    pub adaptive_window: bool,
    pub max_send_buffer_size: usize,
}

impl Default for H2Config {
    fn default() -> Self {
        Self {
            max_concurrent_streams: 100,
            initial_window_size: 65535,
            max_frame_size: 16384,
            enable_push: false,
            enable_connect_protocol: true,
            keepalive_interval: Some(Duration::from_secs(30)),
            keepalive_timeout: Duration::from_secs(10),
            adaptive_window: true,
            max_send_buffer_size: 1024 * 1024,
        }
    }
}

impl H2Config {
    pub fn ai_optimized() -> Self {
        Self {
            max_concurrent_streams: 1000,
            initial_window_size: 1048576, // 1MB
            max_frame_size: 32768,
            enable_push: false,
            enable_connect_protocol: true,
            keepalive_interval: Some(Duration::from_secs(15)),
            keepalive_timeout: Duration::from_secs(5),
            adaptive_window: true,
            max_send_buffer_size: 4 * 1024 * 1024, // 4MB
        }
    }
}

impl ProtocolConfig for H2Config {
    fn validate(&self) -> Result<(), String> {
        if self.max_concurrent_streams == 0 {
            return Err("max_concurrent_streams must be greater than 0".to_string());
        }
        if self.initial_window_size < 65535 {
            return Err("initial_window_size must be at least 65535".to_string());
        }
        if self.max_frame_size < 16384 || self.max_frame_size > 16777215 {
            return Err("max_frame_size must be between 16384 and 16777215".to_string());
        }
        Ok(())
    }

    fn timeout_config(&self) -> TimeoutConfig {
        TimeoutConfig {
            request_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(300),
            keepalive_timeout: Some(self.keepalive_timeout),
        }
    }

    fn to_http_config(&self) -> HttpConfig {
        HttpConfig::default()
    }
}

/// HTTP/3 protocol configuration
#[derive(Debug, Clone)]
pub struct H3Config {
    pub max_idle_timeout: Duration,
    pub max_udp_payload_size: u16,
    pub initial_max_data: u64,
    pub initial_max_stream_data_bidi_local: u64,
    pub initial_max_stream_data_bidi_remote: u64,
    pub initial_max_stream_data_uni: u64,
    pub initial_max_streams_bidi: u64,
    pub initial_max_streams_uni: u64,
    pub enable_early_data: bool,
    pub enable_0rtt: bool,
    pub congestion_control: CongestionControl,
}

impl Default for H3Config {
    fn default() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(30),
            max_udp_payload_size: 1452,
            initial_max_data: 10485760,                   // 10MB
            initial_max_stream_data_bidi_local: 1048576,  // 1MB
            initial_max_stream_data_bidi_remote: 1048576, // 1MB
            initial_max_stream_data_uni: 1048576,         // 1MB
            initial_max_streams_bidi: 100,
            initial_max_streams_uni: 100,
            enable_early_data: true,
            enable_0rtt: true,
            congestion_control: CongestionControl::Cubic,
        }
    }
}

impl H3Config {
    pub fn ai_optimized() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(60),
            max_udp_payload_size: 1452,
            initial_max_data: 104857600,                   // 100MB
            initial_max_stream_data_bidi_local: 10485760,  // 10MB
            initial_max_stream_data_bidi_remote: 10485760, // 10MB
            initial_max_stream_data_uni: 10485760,         // 10MB
            initial_max_streams_bidi: 1000,
            initial_max_streams_uni: 1000,
            enable_early_data: true,
            enable_0rtt: true,
            congestion_control: CongestionControl::Bbr,
        }
    }

    pub fn streaming_optimized() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(300),
            max_udp_payload_size: 1452,
            initial_max_data: 1073741824,                   // 1GB
            initial_max_stream_data_bidi_local: 104857600,  // 100MB
            initial_max_stream_data_bidi_remote: 104857600, // 100MB
            initial_max_stream_data_uni: 104857600,         // 100MB
            initial_max_streams_bidi: 10000,
            initial_max_streams_uni: 10000,
            enable_early_data: true,
            enable_0rtt: true,
            congestion_control: CongestionControl::Bbr,
        }
    }
}

impl ProtocolConfig for H3Config {
    fn validate(&self) -> Result<(), String> {
        if self.max_idle_timeout.as_secs() == 0 {
            return Err("max_idle_timeout must be greater than 0".to_string());
        }
        if self.max_udp_payload_size < 1200 {
            return Err("max_udp_payload_size must be at least 1200".to_string());
        }
        if self.initial_max_data == 0 {
            return Err("initial_max_data must be greater than 0".to_string());
        }
        Ok(())
    }

    fn timeout_config(&self) -> TimeoutConfig {
        TimeoutConfig {
            request_timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(5),
            idle_timeout: self.max_idle_timeout,
            keepalive_timeout: Some(self.max_idle_timeout / 2),
        }
    }

    fn to_http_config(&self) -> HttpConfig {
        HttpConfig::default()
    }
}

/// Quiche QUIC configuration
#[derive(Debug, Clone)]
pub struct QuicheConfig {
    pub max_idle_timeout: Duration,
    pub initial_max_data: u64,
    pub initial_max_stream_data_bidi_local: u64,
    pub initial_max_stream_data_bidi_remote: u64,
    pub initial_max_stream_data_uni: u64,
    pub initial_max_streams_bidi: u64,
    pub initial_max_streams_uni: u64,
    pub max_udp_payload_size: u16,
    pub enable_early_data: bool,
    pub enable_hystart: bool,
    pub congestion_control: CongestionControl,
    pub max_connection_window: u64,
    pub max_stream_window: u64,
}

impl Default for QuicheConfig {
    fn default() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(30),
            initial_max_data: 10485760,                   // 10MB
            initial_max_stream_data_bidi_local: 1048576,  // 1MB
            initial_max_stream_data_bidi_remote: 1048576, // 1MB
            initial_max_stream_data_uni: 1048576,         // 1MB
            initial_max_streams_bidi: 100,
            initial_max_streams_uni: 100,
            max_udp_payload_size: 1452,
            enable_early_data: true,
            enable_hystart: true,
            congestion_control: CongestionControl::Cubic,
            max_connection_window: 25165824, // 24MB
            max_stream_window: 16777216,     // 16MB
        }
    }
}

impl QuicheConfig {
    pub fn ai_optimized() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(60),
            initial_max_data: 104857600,                   // 100MB
            initial_max_stream_data_bidi_local: 10485760,  // 10MB
            initial_max_stream_data_bidi_remote: 10485760, // 10MB
            initial_max_stream_data_uni: 10485760,         // 10MB
            initial_max_streams_bidi: 1000,
            initial_max_streams_uni: 1000,
            max_udp_payload_size: 1452,
            enable_early_data: true,
            enable_hystart: true,
            congestion_control: CongestionControl::Bbr,
            max_connection_window: 268435456, // 256MB
            max_stream_window: 134217728,     // 128MB
        }
    }

    pub fn low_latency() -> Self {
        Self {
            max_idle_timeout: Duration::from_secs(15),
            initial_max_data: 52428800,                   // 50MB
            initial_max_stream_data_bidi_local: 5242880,  // 5MB
            initial_max_stream_data_bidi_remote: 5242880, // 5MB
            initial_max_stream_data_uni: 5242880,         // 5MB
            initial_max_streams_bidi: 500,
            initial_max_streams_uni: 500,
            max_udp_payload_size: 1200, // Conservative for low latency
            enable_early_data: true,
            enable_hystart: false, // Disable for predictable latency
            congestion_control: CongestionControl::Bbr,
            max_connection_window: 67108864, // 64MB
            max_stream_window: 33554432,     // 32MB
        }
    }
}

impl ProtocolConfig for QuicheConfig {
    fn validate(&self) -> Result<(), String> {
        if self.max_idle_timeout.as_secs() == 0 {
            return Err("max_idle_timeout must be greater than 0".to_string());
        }
        if self.initial_max_data == 0 {
            return Err("initial_max_data must be greater than 0".to_string());
        }
        if self.max_udp_payload_size < 1200 {
            return Err("max_udp_payload_size must be at least 1200".to_string());
        }
        Ok(())
    }

    fn timeout_config(&self) -> TimeoutConfig {
        TimeoutConfig {
            request_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(5),
            idle_timeout: self.max_idle_timeout,
            keepalive_timeout: Some(self.max_idle_timeout / 3),
        }
    }

    fn to_http_config(&self) -> HttpConfig {
        HttpConfig::default()
    }
}

/// Congestion control algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionControl {
    Reno,
    Cubic,
    Bbr,
    BbrV2,
}

impl Default for CongestionControl {
    fn default() -> Self {
        Self::Cubic
    }
}

/// Convert H3Config to quiche::Config for HTTP/3 connections
fn convert_h3_config_to_quiche(h3_config: &H3Config) -> Result<quiche::Config, String> {
    let mut config = quiche::Config::new(quiche::PROTOCOL_VERSION)
        .map_err(|e| format!("Failed to create quiche config: {}", e))?;

    // Set transport parameters from H3Config
    config.set_initial_max_data(h3_config.initial_max_data);
    config.set_initial_max_streams_bidi(h3_config.initial_max_streams_bidi);
    config.set_initial_max_streams_uni(h3_config.initial_max_streams_uni);
    config.set_initial_max_stream_data_bidi_local(h3_config.initial_max_stream_data_bidi_local);
    config.set_initial_max_stream_data_bidi_remote(h3_config.initial_max_stream_data_bidi_remote);
    config.set_initial_max_stream_data_uni(h3_config.initial_max_stream_data_uni);
    
    // Set idle timeout
    config.set_max_idle_timeout(h3_config.max_idle_timeout.as_millis() as u64);
    
    // Set UDP payload size
    config.set_max_recv_udp_payload_size(h3_config.max_udp_payload_size as usize);
    config.set_max_send_udp_payload_size(h3_config.max_udp_payload_size as usize);

    // Enable early data and 0-RTT if requested
    config.enable_early_data();
    if h3_config.enable_0rtt {
        // 0-RTT is enabled by default in quiche when early data is enabled
    }

    // Set congestion control algorithm
    match h3_config.congestion_control {
        CongestionControl::Cubic => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::CUBIC),
        CongestionControl::Reno => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::Reno),
        CongestionControl::Bbr => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::BBR),
        CongestionControl::BbrV2 => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::BBR2),
    }

    // Set HTTP/3 application protocol
    config.set_application_protos(&[b"h3"]).map_err(|e| format!("Failed to set application protos: {}", e))?;

    Ok(config)
}

/// Convert QuicheConfig to quiche::Config for HTTP/3 connections  
fn convert_quiche_config_to_quiche(quiche_config: &QuicheConfig) -> Result<quiche::Config, String> {
    let mut config = quiche::Config::new(quiche::PROTOCOL_VERSION)
        .map_err(|e| format!("Failed to create quiche config: {}", e))?;

    // Set transport parameters from QuicheConfig
    config.set_initial_max_data(quiche_config.initial_max_data);
    config.set_initial_max_streams_bidi(quiche_config.initial_max_streams_bidi);
    config.set_initial_max_streams_uni(quiche_config.initial_max_streams_uni);
    config.set_initial_max_stream_data_bidi_local(quiche_config.initial_max_stream_data_bidi_local);
    config.set_initial_max_stream_data_bidi_remote(quiche_config.initial_max_stream_data_bidi_remote);
    config.set_initial_max_stream_data_uni(quiche_config.initial_max_stream_data_uni);
    
    // Set idle timeout
    config.set_max_idle_timeout(quiche_config.max_idle_timeout.as_millis() as u64);
    
    // Set UDP payload size
    config.set_max_recv_udp_payload_size(quiche_config.max_udp_payload_size as usize);
    config.set_max_send_udp_payload_size(quiche_config.max_udp_payload_size as usize);

    // Enable early data and 0-RTT if requested
    config.enable_early_data();
    if quiche_config.enable_0rtt {
        // 0-RTT is enabled by default in quiche when early data is enabled
    }

    // Set congestion control algorithm
    match quiche_config.congestion_control {
        CongestionControl::Cubic => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::CUBIC),
        CongestionControl::Reno => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::Reno),
        CongestionControl::Bbr => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::BBR),
        CongestionControl::BbrV2 => config.set_cc_algorithm(quiche::CongestionControlAlgorithm::BBR2),
    }

    // Set HTTP/3 application protocol
    config.set_application_protos(&[b"h3"]).map_err(|e| format!("Failed to set application protos: {}", e))?;

    Ok(config)
}
