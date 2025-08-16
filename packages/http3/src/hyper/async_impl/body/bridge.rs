//! AsyncStream to HttpBody bridging functionality
//!
//! This module provides the bridge between AsyncStream and HttpBody,
//! enabling zero-allocation streaming with safe patterns.

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use fluent_ai_async::AsyncStream;
use http_body::Body as HttpBody;

use crate::wrappers::FrameWrapper;

/// Real AsyncStream to HttpBody bridge implementation
pub struct AsyncStreamHttpBody {
    frame_stream: AsyncStream<FrameWrapper>,
}

impl AsyncStreamHttpBody {
    pub(super) fn new(frame_stream: AsyncStream<FrameWrapper>) -> Self {
        Self { frame_stream }
    }
}

impl HttpBody for AsyncStreamHttpBody {
    type Data = Bytes;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        // Use try_next() to get frames from AsyncStream - no futures needed
        if let Some(frame_wrapper) = self.frame_stream.try_next() {
            std::task::Poll::Ready(Some(Ok(frame_wrapper.0)))
        } else {
            // No more frames available right now
            std::task::Poll::Pending
        }
    }

    fn is_end_stream(&self) -> bool {
        // Check if the AsyncStream has no more data
        self.frame_stream.is_empty()
    }
}

/// Convert AsyncStream to HttpBody frame processing
pub(super) fn convert_body_to_frame_stream<B>(body: B) -> AsyncStream<FrameWrapper, 1024>
where
    B: HttpBody + Send + 'static + Unpin,
    B::Data: Into<Bytes> + Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send,
{
    AsyncStream::with_channel(move |sender| {
        fluent_ai_async::spawn_task(move || {
            // Convert body frames to FrameWrapper
            // This is a placeholder implementation - real implementation would
            // properly poll the HttpBody and convert frames
            let frame = http_body::Frame::data(Bytes::from("converted data"));
            fluent_ai_async::emit!(sender, FrameWrapper::from(frame));
        });
    })
}
