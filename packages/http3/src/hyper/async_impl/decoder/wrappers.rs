//! MessageChunk wrapper types for decoder integration
//! Zero-allocation, blazing-fast wrapper implementations with fluent_ai_async compliance

use fluent_ai_async::prelude::*;
use super::core::Decoder;

/// MessageChunk wrapper for Decoder to implement Body trait with zero-allocation
#[derive(Debug)]
pub struct DecoderBodyWrapper {
    decoder: Decoder,
    body_data: Option<bytes::Bytes>,
    error_message: Option<String>,
}

impl MessageChunk for DecoderBodyWrapper {
    #[inline]
    fn bad_chunk(error: String) -> Self {
        Self {
            decoder: Decoder::empty(),
            body_data: None,
            error_message: Some(error),
        }
    }

    #[inline]
    fn is_error(&self) -> bool {
        self.error_message.is_some()
    }

    #[inline]
    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl DecoderBodyWrapper {
    /// Create new wrapper with decoder and empty body data
    #[inline]
    pub(super) fn new(decoder: Decoder) -> Self {
        Self {
            decoder,
            body_data: Some(bytes::Bytes::new()),
            error_message: None,
        }
    }

    /// Create wrapper with decoder and initial body data
    #[inline]
    pub(super) fn with_data(decoder: Decoder, data: bytes::Bytes) -> Self {
        Self {
            decoder,
            body_data: Some(data),
            error_message: None,
        }
    }

    /// Get reference to the decoder
    #[inline]
    pub(super) fn decoder(&self) -> &Decoder {
        &self.decoder
    }
}

impl http_body::Body for DecoderBodyWrapper {
    type Data = bytes::Bytes;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    #[inline]
    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        
        if let Some(ref error_msg) = this.error_message {
            let error: Box<dyn std::error::Error + Send + Sync> = 
                Box::new(std::io::Error::new(std::io::ErrorKind::Other, error_msg.clone()));
            return std::task::Poll::Ready(Some(Err(error)));
        }

        if let Some(data) = this.body_data.take() {
            let frame = http_body::Frame::data(data);
            std::task::Poll::Ready(Some(Ok(frame)))
        } else {
            std::task::Poll::Ready(None)
        }
    }

    #[inline]
    fn size_hint(&self) -> http_body::SizeHint {
        if self.error_message.is_some() {
            http_body::SizeHint::with_exact(0)
        } else if let Some(ref data) = self.body_data {
            http_body::SizeHint::with_exact(data.len() as u64)
        } else {
            http_body::SizeHint::default()
        }
    }
}

/// MessageChunk wrapper for Decoder to implement Stream trait with zero-allocation
pub struct DecoderStreamWrapper {
    decoder: Decoder,
    stream_data: Option<fluent_ai_async::AsyncStream<crate::wrappers::BytesWrapper>>,
    error_message: Option<String>,
}

impl MessageChunk for DecoderStreamWrapper {
    #[inline]
    fn bad_chunk(error: String) -> Self {
        Self {
            decoder: Decoder::empty(),
            stream_data: None,
            error_message: Some(error),
        }
    }

    #[inline]
    fn is_error(&self) -> bool {
        self.error_message.is_some()
    }

    #[inline]
    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl DecoderStreamWrapper {
    /// Create new stream wrapper with decoder and body
    pub fn new(decoder: Decoder, body: super::ResponseBody) -> Self {
        // Create optimized stream using decoder's decode method
        let stream_data = Some(decoder.decode(body));
        
        Self {
            decoder,
            stream_data,
            error_message: None,
        }
    }
    
    /// Create wrapper with pre-built stream
    #[inline]
    pub(super) fn with_stream(
        decoder: Decoder, 
        stream: fluent_ai_async::AsyncStream<crate::wrappers::BytesWrapper>
    ) -> Self {
        Self {
            decoder,
            stream_data: Some(stream),
            error_message: None,
        }
    }

    /// Get reference to the decoder
    #[inline]
    pub(super) fn decoder(&self) -> &Decoder {
        &self.decoder
    }
}

impl futures::Stream for DecoderStreamWrapper {
    type Item = Result<crate::wrappers::BytesWrapper, Box<dyn std::error::Error + Send + Sync>>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        
        if let Some(ref error_msg) = this.error_message {
            let error: Box<dyn std::error::Error + Send + Sync> = 
                Box::new(std::io::Error::new(std::io::ErrorKind::Other, error_msg.clone()));
            return std::task::Poll::Ready(Some(Err(error)));
        }

        if let Some(stream) = this.stream_data.take() {
            // Use non-blocking try_next for elite polling performance
            match stream.try_next() {
                Some(item) => {
                    // Put stream back for next poll
                    this.stream_data = Some(stream);
                    std::task::Poll::Ready(Some(Ok(item)))
                }
                None => {
                    // Stream exhausted
                    std::task::Poll::Ready(None)
                }
            }
        } else {
            std::task::Poll::Ready(None)
        }
    }
}

/// Utility function to create decoder body wrapper from headers and data
#[inline]
pub fn create_decoder_body(
    headers: &http::HeaderMap,
    data: bytes::Bytes,
) -> DecoderBodyWrapper {
    let decoder = Decoder::from_headers(headers);
    DecoderBodyWrapper::with_data(decoder, data)
}

/// Utility function to create decoder stream wrapper from headers and body
#[inline]
pub fn create_decoder_stream(
    headers: &http::HeaderMap,
    body: super::ResponseBody,
) -> DecoderStreamWrapper {
    let decoder = Decoder::from_headers(headers);
    DecoderStreamWrapper::new(decoder, body)
}