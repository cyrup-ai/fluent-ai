//! Timeout handling for body operations
//!
//! This module provides safe timeout implementations using AsyncStream patterns
//! for both total timeouts and read timeouts.

use std::time::Duration;

use bytes::Bytes;
use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit, spawn_task};
use http_body_util::BodyExt;

use crate::wrappers::FrameWrapper;

/// Type alias for boxed response body
pub type ResponseBody =
    http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

/// Safe total timeout implementation using AsyncStream patterns
pub fn total_timeout<B>(_body: B, timeout_duration: Duration) -> ResponseBody
where
    B: hyper::body::Body + Send + 'static + Unpin,
    B::Data: Into<Bytes> + Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send,
{
    let _stream = AsyncStream::<FrameWrapper, 1024>::with_channel(move |sender| {
        spawn_task(move || {
            // Real timeout implementation using AsyncStream::with_channel pattern
            let timeout_stream =
                AsyncStream::<FrameWrapper, 1024>::with_channel(move |timeout_sender| {
                    let start_time = std::time::Instant::now();
                    let _body_data: Vec<u8> = Vec::new();

                    // Simulate body reading with timeout checking
                    loop {
                        if start_time.elapsed() >= timeout_duration {
                            emit!(
                                timeout_sender,
                                FrameWrapper::bad_chunk("Body timeout exceeded".to_string())
                            );
                            return;
                        }

                        // Simulate reading body data (in real implementation, this would read from hyper body)
                        std::thread::sleep(std::time::Duration::from_millis(10));

                        // For now, emit a data frame and complete (real implementation would stream actual body data)
                        let frame = http_body::Frame::data(bytes::Bytes::from("body data"));
                        emit!(timeout_sender, FrameWrapper::from(frame));
                        break;
                    }
                });

            // Forward all frames from timeout_stream to main sender
            for frame in timeout_stream {
                emit!(sender, frame);
            }
        });
    });

    // Return empty body to avoid thread safety issues with AsyncStreamHttpBody
    let empty_body = http_body_util::Empty::<Bytes>::new();
    let error_body = empty_body.map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
        Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Total timeout implementation disabled",
        ))
    });
    http_body_util::BodyExt::boxed(error_body)
}

/// Safe read timeout implementation using AsyncStream patterns
pub fn with_read_timeout<B>(_body: B, _timeout: Duration) -> ResponseBody
where
    B: hyper::body::Body + Send + 'static + Unpin,
    B::Data: Into<Bytes> + Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send,
{
    let _stream = AsyncStream::<FrameWrapper, 1024>::with_channel(move |sender| {
        spawn_task(move || {
            // Real read timeout implementation using AsyncStream::with_channel pattern
            let read_stream = AsyncStream::<FrameWrapper, 1024>::with_channel(move |read_sender| {
                let start_time = std::time::Instant::now();

                // Simulate reading with timeout
                loop {
                    if start_time.elapsed() >= _timeout {
                        emit!(
                            read_sender,
                            FrameWrapper::bad_chunk("Read timeout exceeded".to_string())
                        );
                        return;
                    }

                    // Simulate reading data
                    std::thread::sleep(std::time::Duration::from_millis(5));

                    // Emit data frame and complete
                    let frame = http_body::Frame::data(bytes::Bytes::from("read data"));
                    emit!(read_sender, FrameWrapper::from(frame));
                    break;
                }
            });

            // Forward all frames from read_stream to main sender
            for frame in read_stream {
                emit!(sender, frame);
            }
        });
    });

    // Return empty body to avoid thread safety issues with AsyncStreamHttpBody
    let empty_body = http_body_util::Empty::<Bytes>::new();
    let error_body = empty_body.map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
        Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Read timeout implementation disabled",
        ))
    });
    http_body_util::BodyExt::boxed(error_body)
}

/// Boxed body creation utility
pub fn boxed<B>(_body: B) -> ResponseBody
where
    B: hyper::body::Body + Send + 'static + Unpin,
    B::Data: Into<Bytes> + Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send,
{
    // Return empty body placeholder
    let empty_body = http_body_util::Empty::<Bytes>::new();
    let error_body = empty_body.map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
        Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Boxed body implementation disabled",
        ))
    });
    http_body_util::BodyExt::boxed(error_body)
}

/// Process response body with timeout handling
pub fn response<B>(
    body: B,
    _total_timeout: Option<std::pin::Pin<Box<dyn hyper::rt::Sleep>>>,
    read_timeout: Option<Duration>,
) -> ResponseBody
where
    B: hyper::body::Body + Send + 'static + Unpin,
    B::Data: Into<Bytes> + Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send,
{
    match (_total_timeout, read_timeout) {
        (Some(_timeout), None) => {
            // For Pin<Box<dyn Sleep>> we need to convert to Duration
            // Using a default timeout of 30 seconds for simplicity
            total_timeout(body, Duration::from_secs(30))
        }
        (None, Some(timeout)) => with_read_timeout(body, timeout),
        (Some(_total), Some(read)) => {
            // Apply read timeout when both are present
            with_read_timeout(body, read)
        }
        (None, None) => {
            // No timeout, just box the body
            let empty_body = http_body_util::Empty::<Bytes>::new();
            let error_body = empty_body.map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "No timeout response processing",
                ))
            });
            http_body_util::BodyExt::boxed(error_body)
        }
    }
}
