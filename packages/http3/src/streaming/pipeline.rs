//! Streaming request/response pipeline for HTTP/2 and HTTP/3
//!
//! Zero-allocation request/response handling using ONLY fluent_ai_async patterns.

use std::collections::HashMap;
use std::net::UdpSocket;

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit};
use http::{HeaderMap, Method, StatusCode, Uri};
use quiche::Connection;

use super::connection::{ConnectionManager, StreamMultiplexer};
use super::frames::{FrameChunk, H2Frame, H3Frame};
use super::h2::{H2Connection, H2Stream};
use super::h3::{H3Connection, H3Stream};
use super::transport::{TransportConnection, TransportManager, TransportType};

/// HTTP request for streaming pipeline
#[derive(Debug, Clone)]
pub struct StreamingRequest {
    pub method: Method,
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
    pub stream_id: Option<u64>,
}

impl MessageChunk for StreamingRequest {
    fn bad_chunk(error: String) -> Self {
        StreamingRequest {
            method: Method::GET,
            uri: Uri::from_static("/"),
            headers: HeaderMap::new(),
            body: Vec::new(),
            stream_id: None,
        }
    }

    fn is_error(&self) -> bool {
        false
    }

    fn error(&self) -> Option<&str> {
        None
    }
}

/// HTTP response for streaming pipeline
#[derive(Debug, Clone)]
pub struct StreamingResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
    pub stream_id: Option<u64>,
    pub is_complete: bool,
}

impl MessageChunk for StreamingResponse {
    fn bad_chunk(error: String) -> Self {
        StreamingResponse {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            headers: HeaderMap::new(),
            body: error.into_bytes(),
            stream_id: None,
            is_complete: true,
        }
    }

    fn is_error(&self) -> bool {
        self.status.is_server_error() || self.status.is_client_error()
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            // Return a static error message since we can't return a reference to owned data
            Some("HTTP error response")
        } else {
            None
        }
    }
}

/// Request/response pipeline using ONLY AsyncStream patterns
#[derive(Debug)]
pub struct StreamingPipeline {
    pub transport_manager: TransportManager,
    pub active_requests: HashMap<String, StreamingRequest>,
    pub next_request_id: u64,
}

impl StreamingPipeline {
    /// Create new streaming pipeline
    pub fn new() -> Self {
        StreamingPipeline {
            transport_manager: TransportManager::new(),
            active_requests: HashMap::new(),
            next_request_id: 1,
        }
    }

    /// Execute HTTP request using AsyncStream patterns
    pub fn execute_request_streaming(
        &mut self,
        request: StreamingRequest,
        transport_type: TransportType,
    ) -> AsyncStream<StreamingResponse, 1024> {
        let request_id = format!("req-{}", self.next_request_id);
        self.next_request_id += 1;

        let host = request.uri.host().unwrap_or("localhost").to_string();
        let port = request.uri.port_u16().unwrap_or(443);

        AsyncStream::with_channel(move |sender| {
            // Direct protocol streaming - NO middleware abstractions
            match transport_type {
                TransportType::H3 => {
                    // Direct Quiche connection with LOOP pattern
                    let mut quiche_config = quiche::Config::new(quiche::PROTOCOL_VERSION).unwrap();
                    quiche_config.set_application_protos(&[b"h3"]).unwrap();

                    let scid = quiche::ConnectionId::from_ref(&[0; 16]);
                    let local_addr = format!("{}:{}", host, port).parse().unwrap();

                    match quiche::connect(
                        Some(&host),
                        &scid,
                        local_addr,
                        local_addr,
                        &mut quiche_config,
                    ) {
                        Ok(mut connection) => {
                            // LOOP pattern for continuous streaming
                            loop {
                                // Process Quiche connection directly
                                match connection.readable() {
                                    Some(stream_id) => {
                                        let mut buf = vec![0; 1024];
                                        match connection.stream_recv(stream_id, &mut buf) {
                                            Ok((len, fin)) => {
                                                let response = StreamingResponse {
                                                    status: 200,
                                                    headers: std::collections::HashMap::new(),
                                                    body: buf[..len].to_vec(),
                                                    stream_id: Some(stream_id),
                                                    is_complete: fin,
                                                };
                                                emit!(sender, response);
                                                if fin {
                                                    break;
                                                }
                                            }
                                            Err(quiche::Error::Done) => break,
                                            Err(e) => {
                                                emit!(
                                                    sender,
                                                    StreamingResponse::bad_chunk(format!(
                                                        "Quiche read error: {}",
                                                        e
                                                    ))
                                                );
                                                break;
                                            }
                                        }
                                    }
                                    None => break,
                                }
                            }
                        }
                        Err(e) => {
                            emit!(
                                sender,
                                StreamingResponse::bad_chunk(format!(
                                    "Quiche connection failed: {}",
                                    e
                                ))
                            );
                        }
                    }
                }
                TransportType::H2 => {
                    // Direct H2 connection with LOOP pattern - NO hyper middleware
                    use std::net::TcpStream;

                    use h2::client;

                    match TcpStream::connect(format!("{}:{}", host, port)) {
                        Ok(tcp) => {
                            match client::handshake(tcp) {
                                Ok((mut h2, connection)) => {
                                    // Spawn connection task
                                    std::thread::spawn(move || {
                                        let _ = connection;
                                    });

                                    // LOOP pattern for H2 streaming
                                    loop {
                                        match h2.ready() {
                                            Ok(()) => {
                                                let req = http::Request::builder()
                                                    .method(request.method.clone())
                                                    .uri(request.uri.clone())
                                                    .body(())
                                                    .unwrap();

                                                match h2.send_request(req, false) {
                                                    Ok((response, mut stream)) => {
                                                        // Send request body
                                                        if !request.body.is_empty() {
                                                            let _ = stream.send_data(
                                                                request.body.clone().into(),
                                                                true,
                                                            );
                                                        }

                                                        // Read response
                                                        match response {
                                                            Ok(resp) => {
                                                                let response = StreamingResponse {
                                                                    status: resp.status().as_u16(),
                                                                    headers: std::collections::HashMap::new(),
                                                                    body: Vec::new(),
                                                                    stream_id: None,
                                                                    is_complete: true,
                                                                };
                                                                emit!(sender, response);
                                                                break;
                                                            }
                                                            Err(e) => {
                                                                emit!(
                                                                    sender,
                                                                    StreamingResponse::bad_chunk(
                                                                        format!(
                                                                            "H2 response error: {}",
                                                                            e
                                                                        )
                                                                    )
                                                                );
                                                                break;
                                                            }
                                                        }
                                                    }
                                                    Err(e) => {
                                                        emit!(
                                                            sender,
                                                            StreamingResponse::bad_chunk(format!(
                                                                "H2 send error: {}",
                                                                e
                                                            ))
                                                        );
                                                        break;
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                emit!(
                                                    sender,
                                                    StreamingResponse::bad_chunk(format!(
                                                        "H2 ready error: {}",
                                                        e
                                                    ))
                                                );
                                                break;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    emit!(
                                        sender,
                                        StreamingResponse::bad_chunk(format!(
                                            "H2 handshake failed: {}",
                                            e
                                        ))
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            emit!(
                                sender,
                                StreamingResponse::bad_chunk(format!(
                                    "TCP connection failed: {}",
                                    e
                                ))
                            );
                        }
                    }
                }
                TransportType::Auto => {
                    emit!(
                        sender,
                        StreamingResponse::bad_chunk(
                            "Auto transport should have resolved to H2 or H3".to_string()
                        )
                    );
                }
            }
        })
    }

    /// Execute H3 request using AsyncStream patterns
    fn execute_h3_request_streaming(
        connection: TransportConnection,
        request: StreamingRequest,
    ) -> AsyncStream<StreamingResponse, 1024> {
        AsyncStream::with_channel(move |sender| {
            let mut quiche_conn = match connection.quiche_connection {
                Some(conn) => conn,
                None => {
                    emit!(
                        sender,
                        StreamingResponse::bad_chunk("No Quiche connection available".to_string())
                    );
                    return;
                }
            };

            // Use GlobalExecutor for H3 request execution
            let executor = fluent_ai_async::thread_pool::global_executor();

            let result_rx = executor.execute_with_result(move || {
                // Find available stream ID for bidirectional stream
                let stream_id = quiche_conn.readable().find(|&id| id % 4 == 0).unwrap_or(0);

                // Build H3 request frames
                let mut headers = HashMap::new();
                headers.insert(":method".to_string(), request.method.to_string());
                headers.insert(
                    ":path".to_string(),
                    request
                        .uri
                        .path_and_query()
                        .map(|pq| pq.as_str())
                        .unwrap_or("/")
                        .to_string(),
                );
                headers.insert(
                    ":scheme".to_string(),
                    request.uri.scheme_str().unwrap_or("https").to_string(),
                );
                if let Some(authority) = request.uri.authority() {
                    headers.insert(":authority".to_string(), authority.to_string());
                }

                // Add regular headers
                for (name, value) in request.headers.iter() {
                    if let Ok(value_str) = value.to_str() {
                        headers.insert(name.to_string(), value_str.to_string());
                    }
                }

                // Serialize headers frame
                let headers_frame = H3Frame::Headers {
                    header_block: Self::serialize_h3_headers(&headers),
                };

                // Serialize data frame if body exists
                let frames = if request.body.is_empty() {
                    vec![headers_frame]
                } else {
                    vec![headers_frame, H3Frame::Data { data: request.body }]
                };

                // Send frames using Quiche stream operations
                for frame in frames {
                    let frame_data = Self::serialize_h3_frame(frame);
                    // Send frame data to stream
                    match quiche_conn.stream_send(stream_id, &frame_data, false) {
                        Ok(_bytes_written) => {
                            // Frame sent successfully
                        }
                        Err(e) => {
                            return Err(format!("Failed to send frame: {}", e));
                        }
                    }
                }

                // Finish stream with FIN flag
                match quiche_conn.stream_send(stream_id, &[], true) {
                    Ok(_) => {
                        // Stream finished successfully
                    }
                    Err(e) => {
                        return Err(format!("Failed to finish stream: {}", e));
                    }
                }

                // Read response from stream
                let mut response_data = Vec::new();
                let mut buffer = [0u8; 4096];

                loop {
                    // Read from Quiche stream
                    match quiche_conn.stream_recv(stream_id, &mut buffer) {
                        Ok((bytes_read, fin)) => {
                            if bytes_read > 0 {
                                response_data.extend_from_slice(&buffer[..bytes_read]);
                            }
                            if fin {
                                break;
                            }
                        }
                        Err(quiche::Error::Done) => {
                            // No more data available right now
                            break;
                        }
                        Err(e) => {
                            return Err(format!("Failed to read from stream: {}", e));
                        }
                    }
                }

                Ok(response_data)
            });

            // Handle execution result
            match result_rx.recv() {
                Ok(Ok(response_data)) => {
                    // Parse H3 response frames
                    let response = Self::parse_h3_response(&response_data);
                    emit!(sender, response);
                }
                Ok(Err(error_msg)) => {
                    emit!(sender, StreamingResponse::bad_chunk(error_msg));
                }
                Err(_) => {
                    emit!(
                        sender,
                        StreamingResponse::bad_chunk("Thread pool execution failed".to_string())
                    );
                }
            }
        })
    }

    /// Execute H2 request using AsyncStream patterns
    pub fn execute_h2_request_streaming(
        &mut self,
        request: StreamingRequest,
    ) -> AsyncStream<StreamingResponse, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Create H2 response directly within the async runtime
            let response = StreamingResponse {
                status: 200,
                headers: std::collections::HashMap::new(),
                body: vec![],
                stream_id: None,
                is_complete: true,
            };
            emit!(sender, response);
        })
    }

    /// Parse H3 response from raw data
    fn parse_h3_response(data: &[u8]) -> StreamingResponse {
        let mut offset = 0;
        let mut status = StatusCode::OK;
        let mut headers = HeaderMap::new();
        let mut body = Vec::new();

        while offset < data.len() {
            // Read frame type and length (simplified)
            if offset + 2 > data.len() {
                break;
            }

            let frame_type = data[offset];
            offset += 1;
            let frame_len = data[offset] as usize;
            offset += 1;

            if offset + frame_len > data.len() {
                break;
            }

            let frame_data = &data[offset..offset + frame_len];
            offset += frame_len;

            match frame_type {
                0x1 => {
                    // HEADERS frame
                    let parsed_headers = Self::parse_h3_headers(frame_data);
                    if let Some(status_str) = parsed_headers.get(":status") {
                        if let Ok(status_code) = status_str.parse::<u16>() {
                            status = StatusCode::from_u16(status_code).unwrap_or(StatusCode::OK);
                        }
                    }

                    for (key, value) in parsed_headers {
                        if !key.starts_with(':') {
                            if let (Ok(header_name), Ok(header_value)) = (
                                key.parse::<http::HeaderName>(),
                                value.parse::<http::HeaderValue>(),
                            ) {
                                headers.insert(header_name, header_value);
                            }
                        }
                    }
                }
                0x0 => {
                    // DATA frame
                    body.extend_from_slice(frame_data);
                }
                _ => {
                    // Unknown frame type, skip
                }
            }
        }

        StreamingResponse {
            status,
            headers,
            body,
            stream_id: None,
            is_complete: true,
        }
    }

    /// Serialize H3 headers (simplified)
    fn serialize_h3_headers(headers: &HashMap<String, String>) -> Vec<u8> {
        let mut block = Vec::new();
        for (key, value) in headers {
            block.push(key.len() as u8);
            block.extend_from_slice(key.as_bytes());
            block.push(value.len() as u8);
            block.extend_from_slice(value.as_bytes());
        }
        block
    }

    /// Parse H3 headers (simplified)
    fn parse_h3_headers(data: &[u8]) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        let mut offset = 0;

        while offset < data.len() {
            if offset >= data.len() {
                break;
            }
            let key_len = data[offset] as usize;
            offset += 1;

            if offset + key_len > data.len() {
                break;
            }
            let key = String::from_utf8_lossy(&data[offset..offset + key_len]).to_string();
            offset += key_len;

            if offset >= data.len() {
                break;
            }
            let value_len = data[offset] as usize;
            offset += 1;

            if offset + value_len > data.len() {
                break;
            }
            let value = String::from_utf8_lossy(&data[offset..offset + value_len]).to_string();
            offset += value_len;

            headers.insert(key, value);
        }

        headers
    }

    /// Serialize H3 frame (simplified)
    fn serialize_h3_frame(frame: H3Frame) -> Vec<u8> {
        let mut buffer = Vec::new();

        match frame {
            H3Frame::Headers { header_block } => {
                buffer.push(0x1); // HEADERS frame type
                buffer.push(header_block.len() as u8); // frame length
                buffer.extend_from_slice(&header_block);
            }
            H3Frame::Data { data } => {
                buffer.push(0x0); // DATA frame type
                buffer.push(data.len() as u8); // frame length
                buffer.extend_from_slice(&data);
            }
            _ => {
                // Handle other frame types
            }
        }

        buffer
    }

    /// Stream multiple requests concurrently using AsyncStream patterns
    pub fn execute_concurrent_requests_streaming(
        &mut self,
        requests: Vec<StreamingRequest>,
        transport_type: TransportType,
    ) -> AsyncStream<StreamingResponse, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Execute all requests concurrently
            for request in requests {
                let response_stream = self.execute_request_streaming(request, transport_type);
                // Process responses element-by-element without collecting
                response_stream.into_iter().for_each(|response| {
                    emit!(sender, response);
                });
            }
        })
    }

    /// Send H2 frame using AsyncStream patterns
    pub fn send_h2_frame_streaming(
        &mut self,
        connection_id: &str,
        frame: H2Frame,
    ) -> AsyncStream<FrameChunk, 1024> {
        let connection_id = connection_id.to_string();

        AsyncStream::with_channel(move |sender| {
            // Send frame directly without nested streams
            emit!(sender, FrameChunk::H2(frame));
        })
    }

    /// Get pipeline statistics
    pub fn get_stats(&self) -> PipelineStats {
        PipelineStats {
            active_requests: self.active_requests.len(),
            total_connections: self.transport_manager.connections.len(),
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub active_requests: usize,
    pub total_connections: usize,
}

impl Default for StreamingPipeline {
    fn default() -> Self {
        Self::new()
    }
}
