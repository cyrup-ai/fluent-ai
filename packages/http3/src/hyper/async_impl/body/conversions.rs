//! Type conversions and From trait implementations for Body
//!
//! This module contains all the From implementations and conversion utilities
//! for transforming various types into Body instances.

use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use fluent_ai_async::{AsyncStream, emit};
use http_body::Body as HttpBody;
use pin_project_lite::pin_project;

use super::constructors::Body;
use crate::wrappers::BytesWrapper;

impl From<Bytes> for Body {
    #[inline]
    fn from(bytes: Bytes) -> Body {
        Body::reusable(bytes)
    }
}

impl From<Vec<u8>> for Body {
    #[inline]
    fn from(vec: Vec<u8>) -> Body {
        Body::reusable(vec.into())
    }
}

impl From<&'static [u8]> for Body {
    #[inline]
    fn from(s: &'static [u8]) -> Body {
        Body::reusable(Bytes::from_static(s))
    }
}

impl From<String> for Body {
    #[inline]
    fn from(s: String) -> Body {
        Body::reusable(s.into())
    }
}

impl From<&'static str> for Body {
    #[inline]
    fn from(s: &'static str) -> Body {
        s.as_bytes().into()
    }
}

#[cfg(feature = "stream")]
#[cfg_attr(docsrs, doc(cfg(feature = "stream")))]
impl From<std::fs::File> for Body {
    #[inline]
    fn from(file: std::fs::File) -> Body {
        // Convert file to AsyncStream without tokio dependencies
        let stream = AsyncStream::with_channel(move |sender| {
            let mut buffer = Vec::new();
            if let Ok(_) = std::io::copy(&mut std::io::BufReader::new(file), &mut buffer) {
                emit!(sender, BytesWrapper::from(Bytes::from(buffer)));
            }
        });
        Body::wrap_stream(stream)
    }
}

impl fmt::Debug for Body {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Body").finish()
    }
}

impl HttpBody for Body {
    type Data = Bytes;
    type Error = crate::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context,
    ) -> Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        use std::task::ready;

        use super::constructors::Inner;

        match self.inner {
            Inner::Reusable(ref mut bytes) => {
                let out = bytes.split_off(0);
                if out.is_empty() {
                    Poll::Ready(None)
                } else {
                    Poll::Ready(Some(Ok(hyper::body::Frame::data(out))))
                }
            }
            Inner::Streaming(ref mut body) => Poll::Ready(
                ready!(Pin::new(body).poll_frame(cx))
                    .map(|opt_chunk| opt_chunk.map_err(|e| crate::HttpError::body(e.to_string()))),
            ),
        }
    }

    fn size_hint(&self) -> http_body::SizeHint {
        use super::constructors::Inner;

        match self.inner {
            Inner::Reusable(ref bytes) => http_body::SizeHint::with_exact(bytes.len() as u64),
            Inner::Streaming(ref body) => body.size_hint(),
        }
    }

    fn is_end_stream(&self) -> bool {
        use super::constructors::Inner;

        match self.inner {
            Inner::Reusable(ref bytes) => bytes.is_empty(),
            Inner::Streaming(ref body) => body.is_end_stream(),
        }
    }
}

// ===== impl IntoBytesBody =====

pin_project! {
    pub struct IntoBytesBody<B> {
        #[pin]
        pub inner: B,
    }
}

// We can't use `map_frame()` because that loses the hint data (for good reason).
// But we aren't transforming the data.
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

    #[inline]
    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }

    #[inline]
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
}
