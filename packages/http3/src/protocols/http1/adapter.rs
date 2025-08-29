//! HTTP/1.1 protocol adapter
//!
//! Simple HTTP/1.1 implementation for fallback scenarios.
//! Provides basic HTTP/1.1 request execution with streaming response support.

use std::time::Instant;
use std::io::{Read, Write, BufRead, BufReader};
use std::net::TcpStream;

use fluent_ai_async::prelude::*;

use crate::prelude::*;
use crate::http::response::{HttpResponse, HttpBodyChunk};

/// Execute HTTP/1.1 request and return canonical HttpResponse
///
/// This adapter provides basic HTTP/1.1 functionality for fallback scenarios
/// when HTTP/2 and HTTP/3 are not available.
///
/// # Arguments
/// * `request` - The HTTP request to execute
///
/// # Returns
/// * `Result<HttpResponse, HttpError>` - HttpResponse with streaming body or error
///
/// # Architecture
/// - Simple TCP connection with basic HTTP/1.1 protocol handling
/// - Converts raw HTTP response to HttpResponse with streaming body
/// - Maintains zero allocation patterns where possible
/// - Provides comprehensive error handling without panics
pub fn execute_http1_request(request: HttpRequest) -> Result<HttpResponse, HttpError> {
    // Extract host and port from request URI
    let (host, port) = extract_host_port(&request)?;
    
    // Establish TCP connection
    let mut stream = establish_connection(&host, port)?;
    
    // Send HTTP/1.1 request
    send_request(&mut stream, &request)?;
    
    // Read and parse response
    let (status_code, headers, body_stream) = read_response(stream)?;
    
    // Create canonical HttpResponse with streaming body
    let response = HttpResponse::from_http1_response(
        status_code,
        headers,
        body_stream,
        1, // HTTP/1.1 uses single stream ID
    );
    
    Ok(response)
}

/// Extract host and port from request URI
///
/// Parses the request URI to extract hostname and port for TCP connection.
fn extract_host_port(request: &HttpRequest) -> Result<(String, u16), HttpError> {
    let uri_str = request.uri();
    
    // Parse URI to extract host and port
    let uri = uri_str.parse::<http::Uri>()
        .map_err(|e| HttpError::new(crate::error::Kind::Request).with(e))?;
    
    let host = uri.host()
        .ok_or_else(|| HttpError::new(crate::error::Kind::Request))?
        .to_string();
    
    let port = match uri.port() {
        Some(port) => port.as_u16(),
        None => {
            // Default ports based on scheme
            match uri.scheme_str() {
                Some("https") => 443,
                Some("http") => 80,
                _ => 80,
            }
        }
    };
    
    Ok((host, port))
}

/// Establish TCP connection to the target host
///
/// Creates a TCP connection with appropriate timeout settings.
fn establish_connection(host: &str, port: u16) -> Result<TcpStream, HttpError> {
    let address = format!("{}:{}", host, port);
    
    let stream = TcpStream::connect(&address)
        .map_err(|e| HttpError::new(crate::error::Kind::Connection).with(e))?;
    
    // Set reasonable timeouts
    stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .map_err(|e| HttpError::new(crate::error::Kind::Connection).with(e))?;
    
    stream.set_write_timeout(Some(std::time::Duration::from_secs(30)))
        .map_err(|e| HttpError::new(crate::error::Kind::Connection).with(e))?;
    
    Ok(stream)
}

/// Send HTTP/1.1 request over TCP connection
///
/// Formats and sends the HTTP request according to HTTP/1.1 specification.
fn send_request(stream: &mut TcpStream, request: &HttpRequest) -> Result<(), HttpError> {
    let mut request_data = Vec::new();
    
    // Request line: METHOD /path HTTP/1.1
    let request_line = format!("{} {} HTTP/1.1\r\n", request.method(), request.uri());
    request_data.extend_from_slice(request_line.as_bytes());
    
    // Add Host header if not present
    let mut has_host_header = false;
    for (name, _) in request.headers().iter() {
        if name.as_str().to_lowercase() == "host" {
            has_host_header = true;
            break;
        }
    }
    
    if !has_host_header {
        if let Ok(uri) = request.uri().parse::<http::Uri>() {
            if let Some(host) = uri.host() {
                let host_header = format!("Host: {}\r\n", host);
                request_data.extend_from_slice(host_header.as_bytes());
            }
        }
    }
    
    // Add Connection: close for simple implementation
    request_data.extend_from_slice(b"Connection: close\r\n");
    
    // Add other headers
    for (name, value) in request.headers().iter() {
        let header_line = format!("{}: {}\r\n", name, value.to_str()
            .map_err(|e| HttpError::new(format!("Invalid header value: {}", e)))?);
        request_data.extend_from_slice(header_line.as_bytes());
    }
    
    // Add Content-Length if we have a body
    if let Some(body) = request.body() {
        let content_length = format!("Content-Length: {}\r\n", body.len());
        request_data.extend_from_slice(content_length.as_bytes());
    }
    
    // Header/body separator
    request_data.extend_from_slice(b"\r\n");
    
    // Add body if present
    if let Some(body) = request.body() {
        request_data.extend_from_slice(body);
    }
    
    // Send the complete request
    stream.write_all(&request_data)
        .map_err(|e| HttpError::new(format!("Failed to send request: {}", e)))?;
    
    stream.flush()
        .map_err(|e| HttpError::new(format!("Failed to flush request: {}", e)))?;
    
    Ok(())
}

/// Read and parse HTTP/1.1 response
///
/// Reads the response from the TCP stream and parses headers and body.
fn read_response(
    mut stream: TcpStream,
) -> Result<(http::StatusCode, http::HeaderMap, AsyncStream<HttpBodyChunk, 1024>), HttpError> {
    let mut reader = BufReader::new(&mut stream);
    
    // Read status line
    let mut status_line = String::new();
    reader.read_line(&mut status_line)
        .map_err(|e| HttpError::new(format!("Failed to read status line: {}", e)))?;
    
    let status_code = parse_status_line(&status_line)?;
    
    // Read headers
    let mut headers = http::HeaderMap::new();
    let mut content_length: Option<usize> = None;
    
    loop {
        let mut header_line = String::new();
        reader.read_line(&mut header_line)
            .map_err(|e| HttpError::new(format!("Failed to read header: {}", e)))?;
        
        // Empty line indicates end of headers
        if header_line == "\r\n" || header_line == "\n" {
            break;
        }
        
        // Parse header
        if let Some(colon_pos) = header_line.find(':') {
            let name = &header_line[..colon_pos].trim();
            let value = &header_line[colon_pos + 1..].trim();
            
            // Track content-length for body reading
            if name.to_lowercase() == "content-length" {
                content_length = value.parse().ok();
            }
            
            // Add to headers map
            if let (Ok(header_name), Ok(header_value)) = (
                http::HeaderName::from_bytes(name.as_bytes()),
                http::HeaderValue::from_str(value),
            ) {
                headers.insert(header_name, header_value);
            }
        }
    }
    
    // Create streaming body
    let body_stream = create_body_stream(reader, content_length);
    
    Ok((status_code, headers, body_stream))
}

/// Parse HTTP status line
///
/// Extracts status code from HTTP/1.1 status line.
fn parse_status_line(status_line: &str) -> Result<http::StatusCode, HttpError> {
    let parts: Vec<&str> = status_line.trim().split_whitespace().collect();
    
    if parts.len() < 2 {
        return Err(HttpError::new("Invalid status line format".to_string()));
    }
    
    let status_code_str = parts[1];
    let status_code_num: u16 = status_code_str.parse()
        .map_err(|e| HttpError::new(format!("Invalid status code: {}", e)))?;
    
    http::StatusCode::from_u16(status_code_num)
        .map_err(|e| HttpError::new(format!("Invalid status code value: {}", e)))
}

/// Create streaming body from buffered reader
///
/// Creates an AsyncStream that reads the response body in chunks.
fn create_body_stream(
    mut reader: BufReader<&mut TcpStream>,
    content_length: Option<usize>,
) -> AsyncStream<HttpBodyChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        spawn_task(move || {
            let mut buffer = [0u8; 8192];
            let mut total_read = 0;
            
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => {
                        // EOF reached
                        let final_chunk = HttpBodyChunk {
                            data: vec![],
                            offset: total_read,
                            is_final: true,
                            timestamp: Instant::now(),
                        };
                        emit!(sender, final_chunk);
                        break;
                    }
                    Ok(bytes_read) => {
                        total_read += bytes_read;
                        
                        let is_final = if let Some(expected_length) = content_length {
                            total_read >= expected_length
                        } else {
                            false
                        };
                        
                        let chunk = HttpBodyChunk {
                            data: buffer[..bytes_read].to_vec(),
                            offset: total_read - bytes_read,
                            is_final,
                            timestamp: Instant::now(),
                        };
                        
                        emit!(sender, chunk);
                        
                        if is_final {
                            break;
                        }
                    }
                    Err(e) => {
                        // Error occurred - emit empty final chunk
                        let error_chunk = HttpBodyChunk {
                            data: vec![],
                            offset: total_read,
                            is_final: true,
                            timestamp: Instant::now(),
                        };
                        emit!(sender, error_chunk);
                        break;
                    }
                }
            }
        });
    })
}