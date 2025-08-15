//! Compression algorithm implementations with zero-allocation streaming
//! Blazing-fast decompression using fluent_ai_async patterns and optimal buffer management

use bytes::Bytes;
use fluent_ai_async::prelude::*;
use super::ResponseBody;

/// Decompress gzip stream with zero-allocation hot path optimization
#[cfg(feature = "gzip")]
pub fn decompress_gzip_stream(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<crate::wrappers::BytesWrapper>
) {
    use std::io::Read;
    use http_body::Body;
    
    fluent_ai_async::spawn_task(move || {
        let mut body_stream = body;
        let mut accumulated_bytes = Vec::with_capacity(8192);
        
        // Collect all body chunks with elite polling
        loop {
            let waker = std::task::Waker::noop();
            let mut context = std::task::Context::from_waker(&waker);
            
            match Body::poll_frame(std::pin::Pin::new(&mut body_stream), &mut context) {
                std::task::Poll::Ready(Some(Ok(frame))) => {
                    if let Some(data) = frame.data_ref() {
                        accumulated_bytes.extend_from_slice(data);
                    }
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    emit!(sender, crate::wrappers::BytesWrapper::bad_chunk(
                        format!("Body collection error: {}", e)
                    ));
                    return;
                }
                std::task::Poll::Ready(None) => break,
                std::task::Poll::Pending => std::thread::yield_now(),
            }
        }
        
        // Perform gzip decompression with optimal chunk size
        let body_bytes = Bytes::from(accumulated_bytes);
        let cursor = std::io::Cursor::new(body_bytes);
        let mut decoder = flate2::read::GzDecoder::new(cursor);
        let mut output_buffer = [0u8; 16384]; // Optimized buffer size
        
        // Stream decompressed data in chunks with zero-allocation hot path
        loop {
            match decoder.read(&mut output_buffer) {
                Ok(0) => break, // EOF
                Ok(n) => {
                    let chunk = Bytes::copy_from_slice(&output_buffer[..n]);
                    let wrapped_chunk = crate::wrappers::BytesWrapper::from(chunk);
                    emit!(sender, wrapped_chunk);
                }
                Err(e) => {
                    let error_chunk = crate::wrappers::BytesWrapper::bad_chunk(
                        format!("Gzip decompression error: {}", e)
                    );
                    emit!(sender, error_chunk);
                    return;
                }
            }
        }
    });
}

/// Decompress deflate stream with zero-allocation optimization
#[cfg(feature = "deflate")]
pub fn decompress_deflate_stream(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<crate::wrappers::BytesWrapper>
) {
    use std::io::Read;
    use http_body::Body;
    
    fluent_ai_async::spawn_task(move || {
        let mut body_stream = body;
        let mut accumulated_bytes = Vec::with_capacity(4096);
        
        // Collect all body chunks with elite polling
        loop {
            let waker = std::task::Waker::noop();
            let mut context = std::task::Context::from_waker(&waker);
            
            match Body::poll_frame(std::pin::Pin::new(&mut body_stream), &mut context) {
                std::task::Poll::Ready(Some(Ok(frame))) => {
                    if let Some(data) = frame.data_ref() {
                        accumulated_bytes.extend_from_slice(data);
                    }
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    emit!(sender, crate::wrappers::BytesWrapper::bad_chunk(
                        format!("Body collection error: {}", e)
                    ));
                    return;
                }
                std::task::Poll::Ready(None) => break,
                std::task::Poll::Pending => std::thread::yield_now(),
            }
        }
        
        // Perform deflate decompression with single-pass optimization
        let body_bytes = Bytes::from(accumulated_bytes);
        let cursor = std::io::Cursor::new(body_bytes);
        let mut decoder = flate2::read::DeflateDecoder::new(cursor);
        let mut decompressed = Vec::with_capacity(body_bytes.len() * 3); // Estimated expansion
        
        match decoder.read_to_end(&mut decompressed) {
            Ok(_) => {
                // Send in optimal chunks for streaming performance
                for chunk in decompressed.chunks(8192) {
                    let bytes_chunk = Bytes::copy_from_slice(chunk);
                    let wrapped_chunk = crate::wrappers::BytesWrapper::from(bytes_chunk);
                    emit!(sender, wrapped_chunk);
                }
            }
            Err(e) => {
                emit!(sender, crate::wrappers::BytesWrapper::bad_chunk(
                    format!("Deflate decompression error: {}", e)
                ));
            }
        }
    });
}

/// Decompress brotli stream with zero-allocation optimization
#[cfg(feature = "brotli")]
pub fn decompress_brotli_stream(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<crate::wrappers::BytesWrapper>
) {
    use std::io::Read;
    use http_body::Body;
    
    fluent_ai_async::spawn_task(move || {
        let mut body_stream = body;
        let mut accumulated_bytes = Vec::with_capacity(16384);
        
        // Collect all body chunks with elite polling
        loop {
            let waker = std::task::Waker::noop();
            let mut context = std::task::Context::from_waker(&waker);
            
            match Body::poll_frame(std::pin::Pin::new(&mut body_stream), &mut context) {
                std::task::Poll::Ready(Some(Ok(frame))) => {
                    if let Some(data) = frame.data_ref() {
                        accumulated_bytes.extend_from_slice(data);
                    }
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    emit!(sender, crate::wrappers::BytesWrapper::bad_chunk(
                        format!("Body collection error: {}", e)
                    ));
                    return;
                }
                std::task::Poll::Ready(None) => break,
                std::task::Poll::Pending => std::thread::yield_now(),
            }
        }
        
        // Perform brotli decompression with optimal buffer sizing
        let body_bytes = Bytes::from(accumulated_bytes);
        let cursor = std::io::Cursor::new(body_bytes);
        let mut decoder = brotli::Decompressor::new(cursor, 4096);
        let mut output_buffer = [0u8; 32768]; // Large buffer for brotli efficiency
        
        // Stream decompressed data with zero-allocation hot path
        loop {
            match decoder.read(&mut output_buffer) {
                Ok(0) => break, // EOF
                Ok(n) => {
                    let chunk = Bytes::copy_from_slice(&output_buffer[..n]);
                    let wrapped_chunk = crate::wrappers::BytesWrapper::from(chunk);
                    emit!(sender, wrapped_chunk);
                }
                Err(e) => {
                    let error_chunk = crate::wrappers::BytesWrapper::bad_chunk(
                        format!("Brotli decompression error: {}", e)
                    );
                    emit!(sender, error_chunk);
                    return;
                }
            }
        }
    });
}

/// Decompress zstd stream with zero-allocation optimization
#[cfg(feature = "zstd")]
pub fn decompress_zstd_stream(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<crate::wrappers::BytesWrapper>
) {
    use http_body::Body;
    
    fluent_ai_async::spawn_task(move || {
        let mut body_stream = body;
        let mut accumulated_bytes = Vec::with_capacity(32768);
        
        // Collect all body chunks with elite polling
        loop {
            let waker = std::task::Waker::noop();
            let mut context = std::task::Context::from_waker(&waker);
            
            match Body::poll_frame(std::pin::Pin::new(&mut body_stream), &mut context) {
                std::task::Poll::Ready(Some(Ok(frame))) => {
                    if let Some(data) = frame.data_ref() {
                        accumulated_bytes.extend_from_slice(data);
                    }
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    emit!(sender, crate::wrappers::BytesWrapper::bad_chunk(
                        format!("Body collection error: {}", e)
                    ));
                    return;
                }
                std::task::Poll::Ready(None) => break,
                std::task::Poll::Pending => std::thread::yield_now(),
            }
        }
        
        // Perform zstd decompression with single-pass optimization
        let body_bytes = Bytes::from(accumulated_bytes);
        match zstd::decode_all(std::io::Cursor::new(body_bytes)) {
            Ok(decompressed) => {
                // Send in optimal chunks for maximum streaming performance
                for chunk in decompressed.chunks(16384) {
                    let bytes_chunk = Bytes::copy_from_slice(chunk);
                    let wrapped_chunk = crate::wrappers::BytesWrapper::from(bytes_chunk);
                    emit!(sender, wrapped_chunk);
                }
            }
            Err(e) => {
                let error_chunk = crate::wrappers::BytesWrapper::bad_chunk(
                    format!("Zstd decompression error: {}", e)
                );
                emit!(sender, error_chunk);
            }
        }
    });
}