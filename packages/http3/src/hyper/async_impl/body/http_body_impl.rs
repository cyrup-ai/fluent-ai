use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body::Body as HttpBody;

use super::types::{Body, Inner};

impl HttpBody for Body {
    type Data = Bytes;
    type Error = crate::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context,
    ) -> Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        match self.inner {
            Inner::Reusable(ref mut bytes) => {
                let out = bytes.split_off(0);
                if out.is_empty() {
                    Poll::Ready(None)
                } else {
                    Poll::Ready(Some(Ok(hyper::body::Frame::data(out))))
                }
            }
            Inner::Streaming(ref mut body) => match Pin::new(body).poll_frame(cx) {
                Poll::Ready(opt) => Poll::Ready(opt.map(|res| {
                    res.map_err(crate::Error::request)
                })),
                Poll::Pending => Poll::Pending,
            },
        }
    }

    fn size_hint(&self) -> http_body::SizeHint {
        match self.inner {
            Inner::Reusable(ref bytes) => {
                let exact = bytes.len() as u64;
                http_body::SizeHint::with_exact(exact)
            }
            Inner::Streaming(ref body) => body.size_hint(),
        }
    }

    fn is_end_stream(&self) -> bool {
        match self.inner {
            Inner::Reusable(ref bytes) => bytes.is_empty(),
            Inner::Streaming(ref body) => body.is_end_stream(),
        }
    }
}