//! HTTP/2 protocol adapter - Pure Streams Architecture
//!
//! Provides HTTP/2 request execution using pure fluent_ai_async streaming patterns.
//! No blocking I/O, sync connections, or no-op wakers.

use fluent_ai_async::{AsyncStream, emit, spawn_task};

use crate::prelude::*;
use crate::protocols::h2::connection::H2Connection;
use crate::protocols::response_converter::convert_http_chunks_to_response;
use crate::protocols::strategy::H2Config;
use crate::http::response::HttpResponse;

/// Execute HTTP/2 request using pure streams-first architecture
///
/// Creates an AsyncStream-based HTTP/2 response without blocking I/O or sync connections.
/// Leverages existing H2Connection streaming infrastructure.
pub fn execute_h2_request(
    request: HttpRequest,
    config: H2Config,
) -> AsyncStream<HttpResponse, 1> {
    AsyncStream::with_channel(move |sender| {
        spawn_task(move || {
            // Use existing H2Connection for streams-first request handling
            match create_h2_connection_stream(request, config) {
                Ok(http_chunk_stream) => {
                    // Convert HttpChunk stream to HttpResponse using existing converter
                    let stream_id = 1; // HTTP/2 stream ID
                    let response = convert_http_chunks_to_response(http_chunk_stream, stream_id);
                    emit!(sender, response);
                }
                Err(e) => {
                    // Emit error response using HttpResponse::error
                    let error_response = HttpResponse::error(
                        http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!("H2 connection failed: {}", e)
                    );
                    emit!(sender, error_response);
                }
            }
        });
    })
}

/// Create H2 connection stream using proper strategy pattern
///
/// This delegates to the HttpProtocolStrategy which handles connection management,
/// context creation, and proper H2 execution following the established architecture.
fn create_h2_connection_stream(
    request: HttpRequest,
    config: H2Config,
) -> Result<AsyncStream<HttpChunk, 1024>, HttpError> {
    // Use the proper strategy pattern instead of direct connection creation
    // This follows the architecture established in strategy.rs
    let h2_strategy = crate::protocols::HttpProtocolStrategy::Http2(config);
    
    // Execute using the strategy, which handles connection management properly
    match h2_strategy.execute(request) {
        Ok(response) => {
            // Convert HttpResponse body_stream to HttpChunk stream
            Ok(AsyncStream::with_channel(move |sender| {
                for body_chunk in response.body_stream {
                    let http_chunk = HttpChunk::from(body_chunk);
                    fluent_ai_async::emit!(sender, http_chunk);
                }
            }))
        }
        Err(error_msg) => {
            // Return error stream following proper error handling patterns
            Ok(AsyncStream::with_channel(move |sender| {
                let error_chunk = HttpChunk::bad_chunk(error_msg);
                fluent_ai_async::emit!(sender, error_chunk);
            }))
        }
    }
}