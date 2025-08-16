use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body::Body as HttpBody;
use http_body_util::combinators::BoxBody;
use pin_project_lite::pin_project;
use fluent_ai_async::AsyncStream;
use crate::wrappers::{BytesWrapper, FrameWrapper};

/// Real AsyncStream to HttpBody bridge implementation
pub(super) struct AsyncStreamHttpBody {
    pub(super) frame_stream: AsyncStream<FrameWrapper>,
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
        if let Some(frame_wrapper) = self.frame_stream.try_next() {
            std::task::Poll::Ready(Some(Ok(frame_wrapper.0)))
        } else {
            std::task::Poll::Pending
        }
    }

    fn is_end_stream(&self) -> bool {
        self.frame_stream.is_empty()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        http_body::SizeHint::default()
    }
}

/// An asynchronous request body.
pub struct Body {
    pub(super) inner: Inner,
}

pub(super) enum Inner {
    Reusable(Bytes),
    Streaming(BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>),
}

/// Converts any `impl Body` into a `impl Stream` of just its DATA frames.
#[cfg(any(feature = "stream", feature = "multipart",))]
pub(crate) struct DataStream<B>(pub(crate) B);

pub(crate) type ResponseBody =
    http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

pin_project! {
    pub(super) struct IntoBytesBody<B> {
        #[pin]
        pub(super) inner: B,
    }
}

impl<B> hyper::body::Body for IntoBytesBody<B>
where
    B: hyper::body::Body,
    B::Data: Into<Bytes>,
{
    type Data = Bytes;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context,
    ) -> Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        use std::task::ready;
        match ready!(self.project().inner.poll_frame(cx)) {
            Some(Ok(f)) => Poll::Ready(Some(Ok(f.map_data(Into::into)))),
            Some(Err(e)) => Poll::Ready(Some(Err(e))),
            None => Poll::Ready(None),
        }
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
}