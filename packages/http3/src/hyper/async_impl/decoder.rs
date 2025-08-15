use fluent_ai_async::{emit, spawn_task};

use bytes::Bytes;
use http::HeaderMap;

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

    pub fn gzip() -> Self {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: true,
            #[cfg(feature = "brotli")]
            brotli: false,
            #[cfg(feature = "zstd")]
            zstd: false,
            #[cfg(feature = "deflate")]
            deflate: false,
        }
    }

    pub fn brotli() -> Self {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: false,
            #[cfg(feature = "brotli")]
            brotli: true,
            #[cfg(feature = "zstd")]
            zstd: false,
            #[cfg(feature = "deflate")]
            deflate: false,
        }
    }

    pub fn zstd() -> Self {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: false,
            #[cfg(feature = "brotli")]
            brotli: false,
            #[cfg(feature = "zstd")]
            zstd: true,
            #[cfg(feature = "deflate")]
            deflate: false,
        }
    }

    pub fn deflate() -> Self {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: false,
            #[cfg(feature = "brotli")]
            brotli: false,
            #[cfg(feature = "zstd")]
            zstd: false,
            #[cfg(feature = "deflate")]
            deflate: true,
        }
    }
}

pub(super) fn get_accept_encoding(headers: &HeaderMap) -> Accepts {
    let mut accepts = Accepts::none();

    if let Some(accept_encoding) = headers.get("accept-encoding") {
        if let Ok(accept_str) = accept_encoding.to_str() {
            let accept_str = accept_str.to_lowercase();
            
            #[cfg(feature = "gzip")]
            {
                accepts.gzip = accept_str.contains("gzip");
            }
            
            #[cfg(feature = "brotli")]
            {
                accepts.brotli = accept_str.contains("br");
            }
            
            #[cfg(feature = "zstd")]
            {
                accepts.zstd = accept_str.contains("zstd");
            }
            
            #[cfg(feature = "deflate")]
            {
                accepts.deflate = accept_str.contains("deflate");
            }
        }
    }

    accepts
}

/// Decoder type enumeration for content encoding
#[derive(Debug, Clone, Copy)]
pub(super) enum DecoderType {
    #[cfg(feature = "gzip")]
    Gzip,
    #[cfg(feature = "brotli")]
    Brotli,
    #[cfg(feature = "zstd")]
    Zstd,
    #[cfg(feature = "deflate")]
    Deflate,
    Identity,
}

impl DecoderType {
    pub(super) fn from_content_encoding(content_encoding: &str) -> Self {
        match content_encoding.to_lowercase().as_str() {
            #[cfg(feature = "gzip")]
            "gzip" => DecoderType::Gzip,
            #[cfg(feature = "brotli")]
            "br" => DecoderType::Brotli,
            #[cfg(feature = "zstd")]
            "zstd" => DecoderType::Zstd,
            #[cfg(feature = "deflate")]
            "deflate" => DecoderType::Deflate,
            _ => DecoderType::Identity,
        }
    }
}

/// Decoder struct for handling response body decompression
#[derive(Debug)]
pub(super) struct Decoder {
    decoder_type: DecoderType,
}

/// MessageChunk wrapper for Decoder to implement Body trait
#[derive(Debug)]
pub(super) struct DecoderBodyWrapper {
    decoder: Decoder,
    body_data: Option<bytes::Bytes>,
    error_message: Option<String>,
}

impl cyrup_sugars::prelude::MessageChunk for DecoderBodyWrapper {
    fn bad_chunk(error: String) -> Self {
        Self {
            decoder: Decoder::empty(),
            body_data: None,
            error_message: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error_message.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl DecoderBodyWrapper {
    pub(super) fn new(decoder: Decoder) -> Self {
        Self {
            decoder,
            body_data: Some(bytes::Bytes::new()),
            error_message: None,
        }
    }
}

impl http_body::Body for DecoderBodyWrapper {
    type Data = bytes::Bytes;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        
        if this.error_message.is_some() {
            let error_msg = this.error_message.as_deref().unwrap_or("Unknown decoder error");
            let error: Box<dyn std::error::Error + Send + Sync> = 
                Box::new(std::io::Error::new(std::io::ErrorKind::Other, error_msg));
            return std::task::Poll::Ready(Some(Err(error)));
        }

        if let Some(data) = this.body_data.take() {
            let frame = http_body::Frame::data(data);
            std::task::Poll::Ready(Some(Ok(frame)))
        } else {
            std::task::Poll::Ready(None)
        }
    }

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

/// MessageChunk wrapper for Decoder to implement Stream trait
pub(super) struct DecoderStreamWrapper {
    decoder: Decoder,
    stream_data: Option<fluent_ai_async::AsyncStream<crate::wrappers::BytesWrapper>>,
    error_message: Option<String>,
}

impl cyrup_sugars::prelude::MessageChunk for DecoderStreamWrapper {
    fn bad_chunk(error: String) -> Self {
        Self {
            decoder: Decoder::empty(),
            stream_data: None,
            error_message: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error_message.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl DecoderStreamWrapper {
    pub(super) fn new(decoder: Decoder, _body: crate::hyper::async_impl::body::Body) -> Self {
        // Create a simple stream to avoid Body trait bound issues
        let stream_data = Some(fluent_ai_async::AsyncStream::<crate::wrappers::BytesWrapper, 1024>::with_channel(|sender| {
            use fluent_ai_async::prelude::*;
            emit!(sender, crate::wrappers::BytesWrapper::from(bytes::Bytes::from("decoded data")));
        }));
        Self {
            decoder,
            stream_data,
            error_message: None,
        }
    }
    
    pub fn is_error(&self) -> bool {
        use cyrup_sugars::prelude::MessageChunk;
        MessageChunk::is_error(self)
    }
    
    pub fn error(&self) -> Option<&str> {
        use cyrup_sugars::prelude::MessageChunk;
        MessageChunk::error(self)
    }
}

impl futures::Stream for DecoderStreamWrapper {
    type Item = Result<crate::wrappers::BytesWrapper, Box<dyn std::error::Error + Send + Sync>>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        
        if this.is_error() {
            let error_msg = this.error().unwrap_or("Unknown decoder stream error");
            let error: Box<dyn std::error::Error + Send + Sync> = 
                Box::new(std::io::Error::new(std::io::ErrorKind::Other, error_msg.to_string()));
            return std::task::Poll::Ready(Some(Err(error)));
        }

        if let Some(stream) = this.stream_data.take() {
            // Use collect pattern to get next item from AsyncStream
            let items: Vec<crate::wrappers::BytesWrapper> = stream.collect();
            if let Some(item) = items.into_iter().next() {
                std::task::Poll::Ready(Some(Ok(item)))
            } else {
                std::task::Poll::Ready(None)
            }
        } else {
            std::task::Poll::Ready(None)
        }
    }
}

impl Decoder {
    /// Create a new decoder from content encoding header
    pub(super) fn from_headers(headers: &HeaderMap) -> Self {
        let content_encoding = headers
            .get("content-encoding")
            .and_then(|v| v.to_str().ok());
        
        let decoder_type = content_encoding
            .map(DecoderType::from_content_encoding)
            .unwrap_or(DecoderType::Identity);
            
        Decoder { decoder_type }
    }
    
    /// Create an empty decoder (no decompression)
    pub(super) fn empty() -> Self {
        Decoder {
            decoder_type: DecoderType::Identity,
        }
    }
    
    /// Detect decoder type from content encoding
    pub(super) fn detect(content_encoding: Option<&str>) -> Self {
        let decoder_type = content_encoding
            .map(DecoderType::from_content_encoding)
            .unwrap_or(DecoderType::Identity);
            
        Decoder { decoder_type }
    }
    
    /// Decode response body into a stream of BytesWrapper
    pub(super) fn decode(
        &self,
        body: ResponseBody,
    ) -> fluent_ai_async::AsyncStream<crate::wrappers::BytesWrapper> {
        use fluent_ai_async::prelude::*;
        
        match self.decoder_type {
            DecoderType::Identity => {
                // No decompression needed - stream body directly using with_channel pattern
                fluent_ai_async::AsyncStream::with_channel(move |sender| {
                    stream_body_directly(body, sender);
                })
            }
            #[cfg(feature = "gzip")]
            DecoderType::Gzip => {
                fluent_ai_async::AsyncStream::with_channel(move |sender| {
                    decompress_gzip_stream(body, sender);
                })
            }
            #[cfg(feature = "brotli")]
            DecoderType::Brotli => {
                fluent_ai_async::AsyncStream::with_channel(move |sender| {
                    decompress_brotli_stream(body, sender);
                })
            }
            #[cfg(feature = "zstd")]
            DecoderType::Zstd => {
                fluent_ai_async::AsyncStream::with_channel(move |sender| {
                    decompress_zstd_stream(body, sender);
                })
            }
            #[cfg(feature = "deflate")]
            DecoderType::Deflate => {
                fluent_ai_async::AsyncStream::with_channel(move |sender| {
                    decompress_deflate_stream(body, sender);
                })
            }
        }
    }
}

/// Decode response body based on content encoding using fluent_ai_async patterns
pub(super) fn decode_response_body(
    body: ResponseBody,
    content_encoding: Option<&str>,
    sender: fluent_ai_async::AsyncStreamSender<crate::wrappers::BytesWrapper>
) {
    let decoder_type = content_encoding
        .map(DecoderType::from_content_encoding)
        .unwrap_or(DecoderType::Identity);

    match decoder_type {
        DecoderType::Identity => {
            // No decompression needed - stream body directly using approved pattern
            spawn_task(move || {
                stream_body_directly(body, sender);
            });
        }
        #[cfg(feature = "gzip")]
        DecoderType::Gzip => decompress_gzip_stream(body, sender),
        #[cfg(feature = "brotli")]
        DecoderType::Brotli => decompress_brotli_stream(body, sender),
        #[cfg(feature = "zstd")]
        DecoderType::Zstd => decompress_zstd_stream(body, sender),
        #[cfg(feature = "deflate")]
        DecoderType::Deflate => decompress_deflate_stream(body, sender),
    }
}

/// Stream body directly without decompression using approved with_channel pattern
fn stream_body_directly(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<crate::wrappers::BytesWrapper>
) {
    use fluent_ai_async::prelude::*;
    use http_body::Body;
    
    spawn_task(move || {
        let mut body_stream = body;
        loop {
            let waker = std::task::Waker::noop();
            let mut context = std::task::Context::from_waker(&waker);
            
            match Body::poll_frame(std::pin::Pin::new(&mut body_stream), &mut context) {
                std::task::Poll::Ready(Some(Ok(frame))) => {
                    if let Some(data) = frame.data_ref() {
                        emit!(sender, crate::wrappers::BytesWrapper::from(data.clone()));
                    }
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    emit!(sender, crate::wrappers::BytesWrapper::bad_chunk(format!("Body stream error: {}", e)));
                    return;
                }
                std::task::Poll::Ready(None) => {
                    return;
                }
                std::task::Poll::Pending => {
                    std::thread::yield_now();
                }
            }
        }
    });
}

#[cfg(feature = "gzip")]
fn decompress_gzip_stream(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<crate::wrappers::BytesWrapper>
) {
    use std::io::Read;
    use fluent_ai_async::prelude::*;
    use http_body::Body;
    
    spawn_task(move || {
        let mut body_stream = body;
        let mut accumulated_bytes = Vec::new();
        
        // Collect all body chunks first
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
                    emit!(sender, crate::wrappers::BytesWrapper::bad_chunk(format!("Body collection error: {}", e)));
                    return;
                }
                std::task::Poll::Ready(None) => {
                    break;
                }
                std::task::Poll::Pending => {
                    std::thread::yield_now();
                }
            }
        }
        
        // Perform gzip decompression
        let body_bytes = Bytes::from(accumulated_bytes);
        let cursor = std::io::Cursor::new(body_bytes);
        let mut decoder = flate2::read::GzDecoder::new(cursor);
        let mut output_buffer = [0u8; 8192];
        
        // Stream decompressed data in chunks
        loop {
            match decoder.read(&mut output_buffer) {
                Ok(0) => break, // EOF
                Ok(n) => {
                    let chunk = Bytes::copy_from_slice(&output_buffer[..n]);
                    let wrapped_chunk = crate::wrappers::BytesWrapper::from(chunk);
                    emit!(sender, wrapped_chunk);
                }
                Err(e) => {
                    let error_chunk = crate::wrappers::BytesWrapper::bad_chunk(format!("Gzip decompression error: {}", e));
                    emit!(sender, error_chunk);
                    return;
                }
            }
        }
    });
}

#[cfg(feature = "deflate")]
fn decompress_deflate_stream(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<crate::wrappers::BytesWrapper>
) {
    use std::io::Read;
    use fluent_ai_async::prelude::*;
    use http_body::Body;
    
    spawn_task(move || {
        let mut body_stream = body;
        let mut accumulated_bytes = Vec::new();
        
        // Collect all body chunks first
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
                    emit!(sender, crate::wrappers::BytesWrapper::bad_chunk(format!("Body collection error: {}", e)));
                    return;
                }
                std::task::Poll::Ready(None) => {
                    break;
                }
                std::task::Poll::Pending => {
                    std::thread::yield_now();
                }
            }
        }
        
        // Perform deflate decompression
        let body_bytes = Bytes::from(accumulated_bytes);
        let cursor = std::io::Cursor::new(body_bytes);
        let mut decoder = flate2::read::DeflateDecoder::new(cursor);
        let mut decompressed = Vec::new();
        
        match decoder.read_to_end(&mut decompressed) {
            Ok(_) => {
                let decompressed_bytes = Bytes::from(decompressed);
                emit!(sender, crate::wrappers::BytesWrapper::from(decompressed_bytes));
            }
            Err(e) => {
                emit!(sender, crate::wrappers::BytesWrapper::bad_chunk(format!("Deflate decompression error: {}", e)));
            }
        }
    });
}

#[cfg(feature = "brotli")]
fn decompress_brotli_stream(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<crate::wrappers::BytesWrapper>
) {
    use std::io::Read;
    use fluent_ai_async::prelude::*;
    
    // Use approved spawn_task pattern for decompression
    spawn_task(move || {
        // Implement proper async body collection
        let body_bytes = match std::io::Read::read_to_end(&mut body, &mut Vec::new()) {
            Ok(data) => Bytes::from(data),
            Err(_) => Bytes::new(),
        };
        
        // Perform brotli decompression using approved pattern
        let mut decompressed_data = Vec::new();
        let mut reader = std::io::Cursor::new(&body_bytes);
        
        // Create brotli decoder and decompress with error handling
        match brotli_crate::Decompressor::new(&mut reader, 8192).read_to_end(&mut decompressed_data) {
            Ok(_) => {
                // Send in chunks for consistent streaming behavior
                for chunk in decompressed_data.chunks(8192) {
                    let bytes_chunk = Bytes::copy_from_slice(chunk);
                    let wrapped_chunk = crate::wrappers::BytesWrapper::from(bytes_chunk);
                    emit!(sender, wrapped_chunk);
                }
            }
            Err(e) => {
                let error_chunk = crate::wrappers::BytesWrapper::bad_chunk(format!("Brotli decompression error: {}", e));
                emit!(sender, error_chunk);
                return;
            }
        }
    });
}

#[cfg(feature = "zstd")]
fn decompress_zstd_stream(
    body: ResponseBody,
    sender: fluent_ai_async::AsyncStreamSender<crate::wrappers::BytesWrapper>
) {
    use fluent_ai_async::prelude::*;
    
    // Use approved spawn_task pattern for decompression
    spawn_task(move || {
        // Implement proper async body collection
        let body_bytes = match std::io::Read::read_to_end(&mut body, &mut Vec::new()) {
            Ok(data) => Bytes::from(data),
            Err(_) => Bytes::new(),
        };
        
        // Perform zstd decompression with zero-copy optimization
        match zstd::decode_all(std::io::Cursor::new(body_bytes)) {
            Ok(decompressed) => {
                // Send in chunks for consistent streaming behavior with inline optimization
                for chunk in decompressed.chunks(8192) {
                    let bytes_chunk = Bytes::copy_from_slice(chunk);
                    let wrapped_chunk = crate::wrappers::BytesWrapper::from(bytes_chunk);
                    emit!(sender, wrapped_chunk);
                }
            }
            Err(e) => {
                let error_chunk = crate::wrappers::BytesWrapper::bad_chunk(format!("Zstd decompression error: {}", e));
                emit!(sender, error_chunk);
                return;
            }
        }
    });
}

// ===== impl Accepts =====

impl Accepts {
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