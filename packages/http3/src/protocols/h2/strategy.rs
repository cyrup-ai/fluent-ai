//! H2 Protocol Strategy Implementation
//!
//! Provides HTTP/2 protocol implementation using h2 crate with AsyncStream patterns.

use std::sync::Arc;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::future::Future;

use fluent_ai_async::{AsyncStream, emit, spawn_task};
use h2::client;
use http::{Method, HeaderMap, HeaderName, HeaderValue, StatusCode, Version};
use bytes::Bytes;

use crate::prelude::*;
use crate::http::request::{HttpRequest, RequestBody};
use crate::http::response::{HttpResponse, HttpBodyChunk, HttpHeader};
use crate::protocols::strategy_trait::ProtocolStrategy;
use crate::connect::{Connector, TcpConnectionChunk};
use crossbeam_utils::Backoff;

/// H2 protocol strategy configuration
#[derive(Debug, Clone)]
pub struct H2Config {
    pub enable_push: bool,
    pub max_concurrent_streams: u32,
    pub initial_window_size: u32,
    pub max_frame_size: u32,
    pub max_header_list_size: u32,
}

impl Default for H2Config {
    fn default() -> Self {
        Self {
            enable_push: false,
            max_concurrent_streams: 100,
            initial_window_size: 65535,
            max_frame_size: 16384,
            max_header_list_size: 16384,
        }
    }
}

/// H2 protocol strategy
#[derive(Debug, Clone)]
pub struct H2Strategy {
    config: H2Config,
}

impl H2Strategy {
    pub fn new(config: H2Config) -> Self {
        Self { config }
    }
}

impl Default for H2Strategy {
    fn default() -> Self {
        Self::new(H2Config::default())
    }
}

impl ProtocolStrategy for H2Strategy {
    fn execute(&self, request: HttpRequest) -> HttpResponse {
        // Create response streams
        let (headers_tx, headers_internal) = AsyncStream::channel();
        let (body_tx, body_internal) = AsyncStream::channel();
        let (trailers_tx, trailers_internal) = AsyncStream::channel();
        
        // Extract request details
        let method = request.method().clone();
        let url = request.url().clone();
        let headers = request.headers().clone();
        let body_data = request.body().and_then(|body| {
            match body {
                RequestBody::Bytes(bytes) => Some(bytes.clone()),
                RequestBody::Text(text) => Some(Bytes::from(text.clone())),
                RequestBody::Json(json) => serde_json::to_vec(json).ok().map(Bytes::from),
                RequestBody::Form(form) => serde_urlencoded::to_string(form).ok().map(|s| Bytes::from(s)),
                _ => None,
            }
        });
        
        // Determine host and scheme from URL
        let host = url.host_str().unwrap_or("localhost").to_string();
        let scheme = url.scheme().to_string();
        let is_https = scheme == "https";
        let port = url.port().unwrap_or(if is_https { 443 } else { 80 });
        let authority = format!("{}:{}", host, port);
        
        // Clone for async task
        let config = self.config.clone();
        
        // Spawn task to handle H2 protocol
        spawn_task(move || {
            // Create a connector for TCP connection
            let mut connector = Connector::default();
            
            // Build the full URI for connection
            let connect_uri = match format!("{}://{}", scheme, authority).parse::<http::Uri>() {
                Ok(uri) => uri,
                Err(e) => {
                    emit!(headers_tx, HttpHeader::bad_chunk(format!("Invalid URI: {}", e)));
                    return;
                }
            };
            
            // Get TCP connection stream
            let tcp_stream = connector.connect(connect_uri);
            
            // Process the TCP connection
            for chunk in tcp_stream.collect() {
                match chunk {
                    TcpConnectionChunk::Connected { stream: Some(stream), .. } => {
                        // We have a connected TCP stream with TLS if needed
                        // Now perform H2 handshake using elite backoff - NO Futures
                        
                        // Create a no-op waker for manual polling
                        use futures::task::noop_waker_ref;
                        let waker = noop_waker_ref();
                        let mut context = Context::from_waker(waker);
                        
                        // Perform H2 handshake using elite backoff manual polling
                        let mut handshake_future = Box::pin(h2::client::handshake(stream));
                        let backoff = Backoff::new();
                        
                        let handshake_result = loop {
                            match handshake_future.as_mut().poll(&mut context) {
                                Poll::Ready(Ok(result)) => {
                                    break Some(result);
                                }
                                Poll::Ready(Err(e)) => {
                                    emit!(body_tx, HttpBodyChunk {
                                        data: Bytes::from(format!("H2 handshake failed: {}", e)),
                                        offset: 0,
                                        is_final: true,
                                        timestamp: std::time::Instant::now(),
                                    });
                                    break None;
                                }
                                Poll::Pending => {
                                    if backoff.is_completed() {
                                        std::thread::yield_now();
                                    } else {
                                        backoff.snooze();
                                    }
                                    continue;
                                }
                            }
                        };
                        
                        if let Some((mut send_request, connection)) = handshake_result {
                            // Spawn connection driver
                            spawn_task(move || {
                                let mut conn_future = connection;
                                let backoff = Backoff::new();
                                let waker = noop_waker_ref();
                                let mut context = Context::from_waker(waker);
                                
                                loop {
                                    match Pin::new(&mut conn_future).poll(&mut context) {
                                        Poll::Ready(Ok(())) => break,
                                        Poll::Ready(Err(e)) => {
                                            eprintln!("H2 connection error: {}", e);
                                            break;
                                        }
                                        Poll::Pending => {
                                            if backoff.is_completed() {
                                                std::thread::yield_now();
                                            } else {
                                                backoff.snooze();
                                            }
                                        }
                                    }
                                }
                            });
                            
                            // Build and send H2 request
                            let mut h2_request = http::Request::builder()
                                .method(method.clone())
                                .uri(url.path());
                            
                            // Add pseudo-headers for HTTP/2
                            h2_request = h2_request
                                .header(":method", method.as_str())
                                .header(":scheme", scheme)
                                .header(":authority", &authority)
                                .header(":path", if url.query().is_some() {
                                    format!("{}?{}", url.path(), url.query().unwrap())
                                } else {
                                    url.path().to_string()
                                });
                            
                            // Add regular headers
                            for (name, value) in headers.iter() {
                                if !name.as_str().starts_with(':') {
                                    h2_request = h2_request.header(name, value);
                                }
                            }
                            
                            // Create body
                            let body_bytes = body_data.unwrap_or_else(|| Bytes::new());
                            
                            // Build final request
                            let h2_request = match h2_request.body(()) {
                                Ok(req) => req,
                                Err(e) => {
                                    emit!(headers_tx, HttpHeader::bad_chunk(format!("Failed to build request: {}", e)));
                                    return;
                                }
                            };
                            
                            // Send request
                            match send_request.send_request(h2_request, false) {
                                Ok((response_future, mut send_stream)) => {
                                    // Send body if present
                                    if !body_bytes.is_empty() {
                                        if let Err(e) = send_stream.send_data(body_bytes, true) {
                                            emit!(body_tx, HttpBodyChunk::bad_chunk(format!("Failed to send body: {}", e)));
                                        }
                                    } else {
                                        if let Err(e) = send_stream.send_data(Bytes::new(), true) {
                                            emit!(body_tx, HttpBodyChunk::bad_chunk(format!("Failed to send empty body: {}", e)));
                                        }
                                    }
                                    
                                    // Get response
                                    let mut resp_future = response_future;
                                    let backoff = Backoff::new();
                                    let waker = noop_waker_ref();
                                    let mut context = Context::from_waker(waker);
                                    
                                    let response = loop {
                                        match Pin::new(&mut resp_future).poll(&mut context) {
                                            Poll::Ready(Ok(response)) => break response,
                                            Poll::Ready(Err(e)) => {
                                                emit!(body_tx, HttpBodyChunk {
                                                    data: Bytes::from(format!("H2 response error: {}", e)),
                                                    offset: 0,
                                                    is_final: true,
                                                    timestamp: std::time::Instant::now(),
                                                });
                                                return;
                                            }
                                            Poll::Pending => {
                                                if backoff.is_completed() {
                                                    std::thread::yield_now();
                                                } else {
                                                    backoff.snooze();
                                                }
                                            }
                                        }
                                    };
                                    
                                    // Process response headers
                                    for (name, value) in response.headers().iter() {
                                        emit!(headers_tx, HttpHeader {
                                            name: name.clone(),
                                            value: value.clone(),
                                            timestamp: std::time::Instant::now(),
                                        });
                                    }
                                    
                                    // Process response body
                                    let mut body_stream = response.into_body();
                                    let backoff = Backoff::new();
                                    
                                    loop {
                                        match body_stream.poll_data(&mut context) {
                                            Poll::Ready(Some(Ok(data))) => {
                                                emit!(body_tx, HttpBodyChunk {
                                                    data,
                                                    offset: 0,
                                                    is_final: false,
                                                    timestamp: std::time::Instant::now(),
                                                });
                                            }
                                            Poll::Ready(Some(Err(e))) => {
                                                emit!(body_tx, HttpBodyChunk {
                                                    data: Bytes::from(format!("H2 body error: {}", e)),
                                                    offset: 0,
                                                    is_final: true,
                                                    timestamp: std::time::Instant::now(),
                                                });
                                                break;
                                            }
                                            Poll::Ready(None) => {
                                                emit!(body_tx, HttpBodyChunk {
                                                    data: Bytes::new(),
                                                    offset: 0,
                                                    is_final: true,
                                                    timestamp: std::time::Instant::now(),
                                                });
                                                break;
                                            }
                                            Poll::Pending => {
                                                if backoff.is_completed() {
                                                    std::thread::yield_now();
                                                } else {
                                                    backoff.snooze();
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Check for trailers
                                    if let Poll::Ready(Ok(Some(trailers))) = body_stream.poll_trailers(&mut context) {
                                        for (name, value) in trailers.iter() {
                                            emit!(trailers_tx, HttpHeader {
                                                name: name.clone(),
                                                value: value.clone(),
                                                timestamp: std::time::Instant::now(),
                                            });
                                        }
                                    }
                                }
                                Err(e) => {
                                    emit!(body_tx, HttpBodyChunk {
                                        data: Bytes::from(format!("H2 send request error: {}", e)),
                                        offset: 0,
                                        is_final: true,
                                        timestamp: std::time::Instant::now(),
                                    });
                                }
                            }
                        }
                        break;
                    }
                    TcpConnectionChunk::Connected { stream: None, .. } => {
                        emit!(body_tx, HttpBodyChunk {
                            data: Bytes::from("TCP connected but no stream available"),
                            offset: 0,
                            is_final: true,
                            timestamp: std::time::Instant::now(),
                        });
                        break;
                    }
                    TcpConnectionChunk::Error { message, .. } => {
                        emit!(body_tx, HttpBodyChunk {
                            data: Bytes::from(format!("TCP connection error: {}", message)),
                            offset: 0,
                            is_final: true,
                            timestamp: std::time::Instant::now(),
                        });
                        break;
                    }
                    _ => continue,
                }
            }
        });
        
        // Create and return HttpResponse
        let mut response = HttpResponse::new(
            headers_internal,
            body_internal,
            trailers_internal,
            Version::HTTP_2,
            0,
        );
        
        response.set_status(StatusCode::OK);
        response
    }
    
    fn protocol_name(&self) -> &'static str {
        "HTTP/2"
    }
    
    fn supports_push(&self) -> bool {
        self.config.enable_push
    }
    
    fn max_concurrent_streams(&self) -> usize {
        self.config.max_concurrent_streams as usize
    }
}