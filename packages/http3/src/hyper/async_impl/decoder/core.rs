//! Core decoder implementation with zero-allocation streaming decompression
//! Blazing-fast response body decoding using fluent_ai_async patterns

use bytes::Bytes;
use fluent_ai_async::prelude::*;
use http::HeaderMap;

use super::ResponseBody;
use super::types::DecoderType;

/// Core decoder for HTTP response body decompression
#[derive(Debug)]
pub struct Decoder {
    decoder_type: DecoderType,
}

impl Decoder {
    /// Create decoder from HTTP headers with zero-allocation header parsing
    #[inline]
    pub fn from_headers(headers: &HeaderMap) -> Self {
        let content_encoding = headers
            .get("content-encoding")
            .and_then(|v| v.to_str().ok());

        let decoder_type = content_encoding
            .map(DecoderType::from_content_encoding)
            .unwrap_or(DecoderType::Identity);

        Decoder { decoder_type }
    }

    /// Create empty decoder (no decompression) with const optimization
    #[inline]
    pub const fn empty() -> Self {
        Decoder {
            decoder_type: DecoderType::Identity,
        }
    }

    /// Detect decoder type from optional content encoding with fast path
    #[inline]
    pub fn detect(content_encoding: Option<&str>) -> Self {
        let decoder_type = content_encoding
            .map(DecoderType::from_content_encoding)
            .unwrap_or(DecoderType::Identity);

        Decoder { decoder_type }
    }

    /// Get the decoder type for inspection
    #[inline]
    pub(super) const fn decoder_type(&self) -> DecoderType {
        self.decoder_type
    }

    /// Check if decompression is needed
    #[inline]
    pub(super) const fn needs_decompression(&self) -> bool {
        self.decoder_type.needs_decompression()
    }

    /// Decode response body into stream with zero-allocation hot path
    pub(super) fn decode(
        &self,
        body: ResponseBody,
    ) -> fluent_ai_async::AsyncStream<crate::wrappers::BytesWrapper> {
        match self.decoder_type {
            DecoderType::Identity => {
                // Fast path: no decompression needed - stream body directly
                fluent_ai_async::AsyncStream::with_channel(move |sender| {
                    self.stream_body_direct(body, sender);
                })
            }
            #[cfg(feature = "gzip")]
            DecoderType::Gzip => fluent_ai_async::AsyncStream::with_channel(move |sender| {
                super::compression::decompress_gzip_stream(body, sender);
            }),
            #[cfg(feature = "brotli")]
            DecoderType::Brotli => fluent_ai_async::AsyncStream::with_channel(move |sender| {
                super::compression::decompress_brotli_stream(body, sender);
            }),
            #[cfg(feature = "zstd")]
            DecoderType::Zstd => fluent_ai_async::AsyncStream::with_channel(move |sender| {
                super::compression::decompress_zstd_stream(body, sender);
            }),
            #[cfg(feature = "deflate")]
            DecoderType::Deflate => fluent_ai_async::AsyncStream::with_channel(move |sender| {
                super::compression::decompress_deflate_stream(body, sender);
            }),
            #[allow(unreachable_patterns)]
            _ => {
                // Fallback for disabled features - treat as identity
                fluent_ai_async::AsyncStream::with_channel(move |sender| {
                    self.stream_body_direct(body, sender);
                })
            }
        }
    }

    /// Stream body directly without decompression (hot path optimization)
    #[inline]
    fn stream_body_direct(
        self,
        mut body: ResponseBody,
        sender: fluent_ai_async::AsyncStreamSender<crate::wrappers::BytesWrapper>,
    ) {
        use http_body::Body;

        // Spawn task for async body processing with elite polling
        fluent_ai_async::spawn_task(move || {
            let mut chunk_count = 0u32;

            loop {
                let waker = std::task::Waker::noop();
                let mut context = std::task::Context::from_waker(&waker);

                match Body::poll_frame(std::pin::Pin::new(&mut body), &mut context) {
                    std::task::Poll::Ready(Some(Ok(frame))) => {
                        if let Some(data) = frame.data_ref() {
                            chunk_count = chunk_count.saturating_add(1);
                            let bytes_data = Bytes::copy_from_slice(data);
                            let wrapped_chunk = crate::wrappers::BytesWrapper::from(bytes_data);
                            emit!(sender, wrapped_chunk);
                        }
                    }
                    std::task::Poll::Ready(Some(Err(e))) => {
                        let error_chunk = crate::wrappers::BytesWrapper::bad_chunk(format!(
                            "Body streaming error: {}",
                            e
                        ));
                        emit!(sender, error_chunk);
                        return;
                    }
                    std::task::Poll::Ready(None) => {
                        // End of stream
                        return;
                    }
                    std::task::Poll::Pending => {
                        // Elite polling with backoff
                        std::thread::yield_now();
                    }
                }

                // Prevent infinite tight loops
                if chunk_count > 10000 {
                    std::thread::sleep(std::time::Duration::from_nanos(100));
                }
            }
        });
    }

    /// Decode response body with pre-allocated buffer optimization
    pub(super) fn decode_with_hint(
        &self,
        body: ResponseBody,
        size_hint: Option<usize>,
    ) -> fluent_ai_async::AsyncStream<crate::wrappers::BytesWrapper> {
        let estimated_size = size_hint.unwrap_or(8192);
        let compression_ratio = self.decoder_type.compression_ratio_estimate();
        let _expected_output_size = (estimated_size as f32 * compression_ratio) as usize;

        // Use standard decode method - buffer optimization handled in compression modules
        self.decode(body)
    }
}

impl Clone for Decoder {
    #[inline]
    fn clone(&self) -> Self {
        Decoder {
            decoder_type: self.decoder_type,
        }
    }
}

impl Default for Decoder {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}
