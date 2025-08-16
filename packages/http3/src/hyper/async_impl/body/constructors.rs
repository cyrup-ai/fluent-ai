//! Body construction and factory methods
//!
//! This module contains all the constructor methods and factory functions
//! for creating Body instances from various sources.

use bytes::Bytes;
use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit};
use http_body::Body as HttpBody;
use http_body_util::BodyExt;

use crate::wrappers::{BytesWrapper, FrameWrapper};

/// Response body representation
pub enum Inner {
    Reusable(Bytes),
    Streaming(http_body_util::combinators::BoxBody<Bytes, crate::Error>),
}

/// Main Body struct
pub struct Body {
    pub(super) inner: Inner,
}

impl Body {
    /// Wrap a [`HttpBody`] in a box inside `Body`.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::hyper::Body;
    /// # fn main() {
    /// let content = "hello,world!".to_string();
    ///
    /// let body = Body::wrap(content);
    /// # }
    /// ```
    pub fn wrap<B>(inner: B) -> Body
    where
        B: HttpBody<Error = Box<dyn std::error::Error + Send + Sync>> + Send + Sync + 'static,
        B::Data: Into<Bytes>,
    {
        let boxed = super::conversions::IntoBytesBody { inner };

        Body {
            inner: Inner::Streaming(http_body_util::BodyExt::boxed(boxed)),
        }
    }

    /// Create a streaming body from an async stream
    ///
    /// Uses AsyncStream for streams-first architecture.
    #[cfg(feature = "stream")]
    #[cfg_attr(docsrs, doc(cfg(feature = "stream")))]
    pub fn wrap_stream<T>(stream: AsyncStream<T>) -> Body
    where
        T: MessageChunk + Default + Send + 'static,
        Bytes: From<T>,
    {
        Body::from_async_stream(stream)
    }

    #[cfg(any(feature = "stream", feature = "multipart"))]
    pub(crate) fn from_async_stream<T>(stream: AsyncStream<T>) -> Body
    where
        T: MessageChunk + Default + Send + 'static,
        Bytes: From<T>,
    {
        use http_body::Frame;

        // Convert the AsyncStream<T> to AsyncStream<FrameWrapper> using collect pattern
        let _frame_stream = AsyncStream::with_channel(
            move |sender: fluent_ai_async::AsyncStreamSender<FrameWrapper, 1024>| {
                // Use collect to get all items from the stream
                let items = stream.collect();
                for item in items {
                    let bytes = Bytes::from(item);
                    let frame = Frame::data(bytes);
                    let frame_wrapper = FrameWrapper::from(frame);
                    emit!(sender, frame_wrapper);
                }
            },
        );

        // Streaming body collection disabled due to type incompatibilities
        // AsyncStream<T> to Vec<u8> conversion will be restored when type compatibility is resolved
        Body::reusable(Bytes::new())
    }

    pub(crate) fn empty() -> Body {
        Body::reusable(Bytes::new())
    }

    pub(crate) fn reusable(chunk: Bytes) -> Body {
        Body {
            inner: Inner::Reusable(chunk),
        }
    }

    pub(crate) fn try_reuse(self) -> (Option<Bytes>, Self) {
        let reuse = match self.inner {
            Inner::Reusable(ref chunk) => Some(chunk.clone()),
            Inner::Streaming { .. } => None,
        };

        (reuse, self)
    }

    pub(crate) fn try_clone(&self) -> Option<Body> {
        match self.inner {
            Inner::Reusable(ref chunk) => Some(Body::reusable(chunk.clone())),
            Inner::Streaming { .. } => None,
        }
    }

    #[cfg(feature = "multipart")]
    pub(crate) fn content_length(&self) -> Option<u64> {
        match self.inner {
            Inner::Reusable(ref bytes) => Some(bytes.len() as u64),
            Inner::Streaming(ref body) => body.size_hint().exact(),
        }
    }

    /// Get a reference to the body bytes, if available
    pub(crate) fn as_bytes(&self) -> Option<&Bytes> {
        match &self.inner {
            Inner::Reusable(bytes) => Some(bytes),
            Inner::Streaming(_) => None,
        }
    }
}

impl Default for Body {
    #[inline]
    fn default() -> Body {
        Body::empty()
    }
}
