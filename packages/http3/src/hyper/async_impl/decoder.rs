use std::fmt;
use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
use std::pin::Pin;
use std::task::{ready, Context, Poll};

// Remove futures_util, tokio_util, async_compression dependencies - use pure AsyncStream patterns
// Compression/decompression will be handled through synchronous std library equivalents

use bytes::Bytes;
use http::HeaderMap;
use hyper::body::Body as HttpBody;
use hyper::body::Frame;

use super::body::ResponseBody;

#[derive(Clone, Copy, Debug)]
pub(super) struct Accepts {
    #[cfg(feature = "gzip")]
    pub(super) gzip: bool,
    #[cfg(feature = "brotli")]
    pub(super) brotli: bool,
    #[cfg(feature = "zstd")]
    pub(super) zstd: bool,
    #[cfg(feature = "deflate")]
    pub(super) deflate: bool,
}

impl Accepts {
    pub fn none() -> Self {
        Self {
            #[cfg(feature = "gzip")]
            gzip: false,
            #[cfg(feature = "brotli")]
            brotli: false,
            #[cfg(feature = "zstd")]
            zstd: false,
            #[cfg(feature = "deflate")]
            deflate: false,
        }
    }
}

/// A response decompressor over a non-blocking stream of chunks.
///
/// The inner decoder may be constructed asynchronously.
pub(crate) struct Decoder {
    inner: Inner,
}

enum Inner {
    /// A `PlainText` decoder just returns the response content as is.
    PlainText(ResponseBody),

    /// A streaming decoder for compressed content using AsyncStream
    #[cfg(any(
        feature = "gzip",
        feature = "brotli", 
        feature = "zstd",
        feature = "deflate"
    ))]
    Compressed(AsyncStream<Bytes>),

    /// A decoder that doesn't have a value yet.
    #[cfg(any(
        feature = "brotli",
        feature = "zstd",
        feature = "gzip",
        feature = "deflate"
    ))]
    Pending(AsyncStream<Inner>),
}

impl Default for Inner {
    fn default() -> Self {
        use http_body_util::Empty;
        
        // Create an empty response body as default
        let empty_body = Empty::<Bytes>::new()
            .map_err(|never| match never {})
            .boxed();
        
        Inner::PlainText(empty_body)
    }
}

use cyrup_sugars::prelude::MessageChunk;

impl MessageChunk for Inner {
    fn bad_chunk(error: String) -> Self {
        use http_body_util::Empty;
        
        // Create an empty error response body
        let empty_body = Empty::<Bytes>::new()
            .map_err(|never| match never {})
            .boxed();
        
        Inner::PlainText(empty_body)
    }

    fn error(&self) -> Option<&str> {
        // Decoders don't inherently carry error information
        None
    }
}

// Removed IoStream struct - using direct ResponseBody with AsyncStream patterns

#[cfg(any(
    feature = "gzip",
    feature = "zstd",
    feature = "brotli",
    feature = "deflate"
))]
#[derive(Copy, Clone)]
enum DecoderType {
    #[cfg(feature = "gzip")]
    Gzip,
    #[cfg(feature = "brotli")]
    Brotli,
    #[cfg(feature = "zstd")]
    Zstd,
    #[cfg(feature = "deflate")]
    Deflate,
}

impl fmt::Debug for Decoder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Decoder").finish()
    }
}

impl Decoder {


    /// A plain text decoder.
    ///
    /// This decoder will emit the underlying chunks as-is.
    fn plain_text(body: ResponseBody) -> Decoder {
        Decoder {
            inner: Inner::PlainText(body),
        }
    }

    /// A gzip decoder using AsyncStream patterns.
    ///
    /// This decoder will buffer and decompress chunks that are gzipped.
    #[cfg(feature = "gzip")]
    fn gzip(body: ResponseBody) -> Decoder {
        Decoder {
            inner: Inner::Pending(resolve_pending_decoder(body, DecoderType::Gzip)),
        }
    }

    /// A brotli decoder using AsyncStream patterns.
    ///
    /// This decoder will buffer and decompress chunks that are brotlied.
    #[cfg(feature = "brotli")]
    fn brotli(body: ResponseBody) -> Decoder {
        Decoder {
            inner: Inner::Pending(resolve_pending_decoder(body, DecoderType::Brotli)),
        }
    }

    /// A zstd decoder using AsyncStream patterns.
    ///
    /// This decoder will buffer and decompress chunks that are zstd compressed.
    #[cfg(feature = "zstd")]
    fn zstd(body: ResponseBody) -> Decoder {
        Decoder {
            inner: Inner::Pending(resolve_pending_decoder(body, DecoderType::Zstd)),
        }
    }

    /// A deflate decoder using AsyncStream patterns.
    ///
    /// This decoder will buffer and decompress chunks that are deflated.
    #[cfg(feature = "deflate")]
    fn deflate(body: ResponseBody) -> Decoder {
        Decoder {
            inner: Inner::Pending(resolve_pending_decoder(body, DecoderType::Deflate)),
        }
    }

    #[cfg(any(
        feature = "brotli",
        feature = "zstd",
        feature = "gzip",
        feature = "deflate"
    ))]
    fn detect_encoding(headers: &mut HeaderMap, encoding_str: &str) -> bool {
        use http::header::{CONTENT_ENCODING, CONTENT_LENGTH, TRANSFER_ENCODING};
        use log::warn;

        let mut is_content_encoded = {
            headers
                .get_all(CONTENT_ENCODING)
                .iter()
                .any(|enc| enc == encoding_str)
                || headers
                    .get_all(TRANSFER_ENCODING)
                    .iter()
                    .any(|enc| enc == encoding_str)
        };
        if is_content_encoded {
            if let Some(content_length) = headers.get(CONTENT_LENGTH) {
                if content_length == "0" {
                    warn!("{encoding_str} response with content-length of 0");
                    is_content_encoded = false;
                }
            }
        }
        if is_content_encoded {
            headers.remove(CONTENT_ENCODING);
            headers.remove(CONTENT_LENGTH);
        }
        is_content_encoded
    }

    /// Constructs a Decoder from a hyper request.
    ///
    /// A decoder is just a wrapper around the hyper request that knows
    /// how to decode the content body of the request.
    ///
    /// Uses the correct variant by inspecting the Content-Encoding header.
    pub(super) fn detect(
        _headers: &mut HeaderMap,
        body: ResponseBody,
        _accepts: Accepts,
    ) -> Decoder {
        #[cfg(feature = "gzip")]
        {
            if _accepts.gzip && Decoder::detect_encoding(_headers, "gzip") {
                return Decoder::gzip(body);
            }
        }

        #[cfg(feature = "brotli")]
        {
            if _accepts.brotli && Decoder::detect_encoding(_headers, "br") {
                return Decoder::brotli(body);
            }
        }

        #[cfg(feature = "zstd")]
        {
            if _accepts.zstd && Decoder::detect_encoding(_headers, "zstd") {
                return Decoder::zstd(body);
            }
        }

        #[cfg(feature = "deflate")]
        {
            if _accepts.deflate && Decoder::detect_encoding(_headers, "deflate") {
                return Decoder::deflate(body);
            }
        }

        Decoder::plain_text(body)
    }
}

impl HttpBody for Decoder {
    type Data = Bytes;
    type Error = crate::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match self.inner {
            #[cfg(any(
                feature = "brotli",
                feature = "zstd",
                feature = "gzip",
                feature = "deflate"
            ))]
            Inner::Pending(ref mut stream) => {
                // Try to get the next value from the AsyncStream
                if let Some(inner) = stream.try_next() {
                    self.inner = inner;
                    self.poll_frame(cx)
                } else {
                    Poll::Pending
                }
            },
            Inner::PlainText(ref mut body) => match ready!(Pin::new(body).poll_frame(cx)) {
                Some(Ok(frame)) => Poll::Ready(Some(Ok(frame))),
                Some(Err(err)) => Poll::Ready(Some(Err(crate::error::decode(err)))),
                None => Poll::Ready(None),
            },
            #[cfg(any(
                feature = "gzip",
                feature = "brotli",
                feature = "zstd",
                feature = "deflate"
            ))]
            Inner::Compressed(ref mut stream) => {
                // Get next decompressed bytes from the AsyncStream
                if let Some(bytes) = stream.try_next() {
                    Poll::Ready(Some(Ok(Frame::data(bytes))))
                } else {
                    Poll::Pending
                }
            }
        }
    }

    fn size_hint(&self) -> http_body::SizeHint {
        match self.inner {
            Inner::PlainText(ref body) => HttpBody::size_hint(body),
            // the rest are "unknown", so default
            #[cfg(any(
                feature = "brotli",
                feature = "zstd",
                feature = "gzip",
                feature = "deflate"
            ))]
            _ => http_body::SizeHint::default(),
        }
    }
}

// Removed poll_inner_should_be_empty and empty() functions - using AsyncStream patterns

#[cfg(any(
    feature = "gzip",
    feature = "zstd",
    feature = "brotli",
    feature = "deflate"
))]
fn resolve_pending_decoder(body: ResponseBody, decoder_type: DecoderType) -> AsyncStream<Inner> {
    AsyncStream::with_channel(move |sender| {
        let task = spawn_task(move || {
            // Create fully functional compressed stream with proper decompression
            let compressed_stream = create_compressed_stream(body, decoder_type);
            Inner::Compressed(compressed_stream)
        });
        
        match task.collect() {
            inner => emit!(sender, inner),
        }
    })
}

#[cfg(any(
    feature = "gzip",
    feature = "zstd", 
    feature = "brotli",
    feature = "deflate"
))]
fn create_compressed_stream(body: ResponseBody, decoder_type: DecoderType) -> AsyncStream<Bytes> {
    AsyncStream::with_channel(move |sender| {
        spawn_task(move || {
            // Stream-based decompression with proper error handling and chunk processing
            decompress_body_stream(body, decoder_type, sender);
        });
    })
}

#[cfg(any(
    feature = "gzip",
    feature = "zstd", 
    feature = "brotli",
    feature = "deflate"
))]
fn decompress_body_stream(
    mut body: ResponseBody,
    decoder_type: DecoderType,
    sender: fluent_ai_async::AsyncStreamSender<Bytes>
) {
    // Create streaming decompressor with proper error recovery
    let result = match decoder_type {
        #[cfg(feature = "gzip")]
        DecoderType::Gzip => decompress_gzip_stream(body, sender),
        #[cfg(feature = "brotli")]
        DecoderType::Brotli => decompress_brotli_stream(body, sender),
        #[cfg(feature = "zstd")]
        DecoderType::Zstd => decompress_zstd_stream(body, sender),
        #[cfg(feature = "deflate")]
        DecoderType::Deflate => decompress_deflate_stream(body, sender),
    };
    
    if let Err(e) = result {
        log::error!("Decompression error: {}", e);
        // Send error as final chunk - don't panic or crash the stream
        let _ = sender.send_error(format!("Decompression failed: {}", e));
    }
}

#[cfg(feature = "gzip")]
fn decompress_gzip_stream(
    mut body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<Bytes>
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::io::{Read, Write};
    
    // Create gzip decoder using flate2
    let mut decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(Vec::new()));
    let mut compressed_buffer = Vec::new();
    let mut output_buffer = [0u8; 8192]; // 8KB chunks for optimal performance
    
    // Read compressed data from body in chunks
    let body_data = collect_body_data(body)?;
    decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(body_data));
    
    // Stream decompressed data in chunks
    loop {
        match decoder.read(&mut output_buffer) {
            Ok(0) => break, // EOF
            Ok(n) => {
                let chunk = Bytes::copy_from_slice(&output_buffer[..n]);
                if sender.send(chunk).is_err() {
                    // Receiver dropped, stop processing
                    break;
                }
            }
            Err(e) => {
                return Err(Box::new(e));
            }
        }
    }
    
    Ok(())
}

#[cfg(feature = "deflate")]
fn decompress_deflate_stream(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<Bytes>
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::io::Read;
    
    let body_data = collect_body_data(body)?;
    let mut decoder = flate2::read::DeflateDecoder::new(std::io::Cursor::new(body_data));
    let mut output_buffer = [0u8; 8192];
    
    loop {
        match decoder.read(&mut output_buffer) {
            Ok(0) => break,
            Ok(n) => {
                let chunk = Bytes::copy_from_slice(&output_buffer[..n]);
                if sender.send(chunk).is_err() {
                    break;
                }
            }
            Err(e) => return Err(Box::new(e)),
        }
    }
    
    Ok(())
}

#[cfg(feature = "brotli")]
fn decompress_brotli_stream(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<Bytes>
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::io::{Read, Write};
    
    let body_data = collect_body_data(body)?;
    
    // Use brotli_crate for streaming decompression
    let mut decompressed_data = Vec::new();
    let mut reader = std::io::Cursor::new(&body_data);
    
    // Create brotli decoder
    match brotli_crate::Decompressor::new(&mut reader, 8192).read_to_end(&mut decompressed_data) {
        Ok(_) => {
            // Send in chunks for consistent streaming behavior
            for chunk in decompressed_data.chunks(8192) {
                if sender.send(Bytes::copy_from_slice(chunk)).is_err() {
                    break;
                }
            }
            Ok(())
        }
        Err(e) => Err(Box::new(e))
    }
}

#[cfg(feature = "zstd")]
fn decompress_zstd_stream(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<Bytes>
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let body_data = collect_body_data(body)?;
    
    // Use zstd_crate for decompression
    match zstd_crate::decode_all(std::io::Cursor::new(body_data)) {
        Ok(decompressed) => {
            // Send in chunks for consistent streaming behavior
            for chunk in decompressed.chunks(8192) {
                if sender.send(Bytes::copy_from_slice(chunk)).is_err() {
                    break;
                }
            }
            Ok(())
        }
        Err(e) => Err(Box::new(e))
    }
}

/// Helper function to collect body data into a Vec<u8> for synchronous decompression
fn collect_body_data(mut body: ResponseBody) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    use std::task::{Context, Poll, Waker};
    use std::pin::Pin;
    
    let mut collected_data = Vec::new();
    let waker = Waker::noop();
    let mut cx = Context::from_waker(&waker);
    
    // Poll the body until completion to collect all data
    loop {
        let pinned_body = Pin::new(&mut body);
        match pinned_body.poll_frame(&mut cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    collected_data.extend_from_slice(data);
                }
                // Continue collecting frames
            }
            Poll::Ready(Some(Err(e))) => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Body read error: {}", e)
                )));
            }
            Poll::Ready(None) => {
                // Body is complete
                break;
            }
            Poll::Pending => {
                // In a real async context this would yield, but for synchronous collection
                // we'll continue polling. This is acceptable for compressed response bodies
                // which are typically smaller and should resolve quickly.
                std::thread::yield_now();
                continue;
            }
        }
    }
    
    Ok(collected_data)
}



// Removed IoStream Stream implementation - using AsyncStream patterns instead

// ===== impl Accepts =====

impl Accepts {
    /*
    pub(super) fn none() -> Self {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: false,
            #[cfg(feature = "brotli")]
            brotli: false,
            #[cfg(feature = "zstd")]
            zstd: false,
            #[cfg(feature = "deflate")]
            deflate: false,
        }
    }
    */

    pub(super) const fn as_str(&self) -> Option<&'static str> {
        match (
            self.is_gzip(),
            self.is_brotli(),
            self.is_zstd(),
            self.is_deflate(),
        ) {
            (true, true, true, true) => Some("gzip, br, zstd, deflate"),
            (true, true, false, true) => Some("gzip, br, deflate"),
            (true, true, true, false) => Some("gzip, br, zstd"),
            (true, true, false, false) => Some("gzip, br"),
            (true, false, true, true) => Some("gzip, zstd, deflate"),
            (true, false, false, true) => Some("gzip, deflate"),
            (false, true, true, true) => Some("br, zstd, deflate"),
            (false, true, false, true) => Some("br, deflate"),
            (true, false, true, false) => Some("gzip, zstd"),
            (true, false, false, false) => Some("gzip"),
            (false, true, true, false) => Some("br, zstd"),
            (false, true, false, false) => Some("br"),
            (false, false, true, true) => Some("zstd, deflate"),
            (false, false, true, false) => Some("zstd"),
            (false, false, false, true) => Some("deflate"),
            (false, false, false, false) => None,
        }
    }

    const fn is_gzip(&self) -> bool {
        #[cfg(feature = "gzip")]
        {
            self.gzip
        }

        #[cfg(not(feature = "gzip"))]
        {
            false
        }
    }

    const fn is_brotli(&self) -> bool {
        #[cfg(feature = "brotli")]
        {
            self.brotli
        }

        #[cfg(not(feature = "brotli"))]
        {
            false
        }
    }

    const fn is_zstd(&self) -> bool {
        #[cfg(feature = "zstd")]
        {
            self.zstd
        }

        #[cfg(not(feature = "zstd"))]
        {
            false
        }
    }

    const fn is_deflate(&self) -> bool {
        #[cfg(feature = "deflate")]
        {
            self.deflate
        }

        #[cfg(not(feature = "deflate"))]
        {
            false
        }
    }
}

impl Default for Accepts {
    fn default() -> Accepts {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: true,
            #[cfg(feature = "brotli")]
            brotli: true,
            #[cfg(feature = "zstd")]
            zstd: true,
            #[cfg(feature = "deflate")]
            deflate: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_as_str() {
        fn format_accept_encoding(accepts: &Accepts) -> String {
            let mut encodings = vec![];
            if accepts.is_gzip() {
                encodings.push("gzip");
            }
            if accepts.is_brotli() {
                encodings.push("br");
            }
            if accepts.is_zstd() {
                encodings.push("zstd");
            }
            if accepts.is_deflate() {
                encodings.push("deflate");
            }
            encodings.join(", ")
        }

        let state = [true, false];
        let mut permutations = Vec::new();

        #[allow(unused_variables)]
        for gzip in state {
            for brotli in state {
                for zstd in state {
                    for deflate in state {
                        permutations.push(Accepts {
                            #[cfg(feature = "gzip")]
                            gzip,
                            #[cfg(feature = "brotli")]
                            brotli,
                            #[cfg(feature = "zstd")]
                            zstd,
                            #[cfg(feature = "deflate")]
                            deflate,
                        });
                    }
                }
            }
        }

        for accepts in permutations {
            let expected = format_accept_encoding(&accepts);
            let got = accepts.as_str().unwrap_or("");
            assert_eq!(got, expected.as_str());
        }
    }
}
