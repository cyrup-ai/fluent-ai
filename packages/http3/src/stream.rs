//! HTTP streaming utilities

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use pin_project_lite::pin_project;

use crate::{HttpError, HttpResult};
use std::path::Path;

pin_project! {
    /// HTTP response stream wrapper that provides zero-allocation streaming
    pub struct HttpStream {
        #[pin]
        inner: Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin>,
        buffer: Vec<u8>,
        chunk_size: usize,
    }
}

impl HttpStream {
    /// Create a new HTTP stream from a reqwest Response with automatic decompression
    pub fn new(response: reqwest::Response) -> Self {
        // reqwest automatically decompresses the bytes_stream() when compression is detected
        // The response headers are already parsed by reqwest's decompression layer
        Self {
            inner: Box::new(response.bytes_stream()),
            buffer: Vec::new(),
            chunk_size: 8192, // 8KB chunks
        }
    }

    /// Create a new HTTP stream with custom chunk size
    pub fn with_chunk_size(response: reqwest::Response, chunk_size: usize) -> Self {
        Self {
            inner: Box::new(response.bytes_stream()),
            buffer: Vec::new(),
            chunk_size,
        }
    }

    /// Get the chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Set the chunk size
    pub fn set_chunk_size(&mut self, chunk_size: usize) {
        self.chunk_size = chunk_size;
    }

    /// Get the current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer
    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
    }

    /// Read the entire stream into a vector
    pub async fn collect(self) -> HttpResult<Vec<u8>> {
        let mut result = Vec::new();
        let mut stream = self;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            result.extend_from_slice(&chunk);
        }

        Ok(result)
    }

    /// Read the entire stream into a string
    pub async fn collect_string(self) -> HttpResult<String> {
        let bytes = self.collect().await?;
        String::from_utf8(bytes).map_err(|e| HttpError::DeserializationError {
            message: format!("Invalid UTF-8 in stream: {}", e),
        })
    }

    /// Read the entire stream and parse as JSON
    pub async fn collect_json<T: serde::de::DeserializeOwned>(self) -> HttpResult<T> {
        let bytes = self.collect().await?;
        serde_json::from_slice(&bytes).map_err(|e| HttpError::DeserializationError {
            message: format!("Failed to parse JSON from stream: {}", e),
        })
    }

    /// Convert to a lines stream
    pub fn lines(self) -> LinesStream {
        LinesStream::new(self)
    }

    /// Convert to a Server-Sent Events stream
    pub fn sse(self) -> SseStream {
        SseStream::new(self)
    }

    /// Convert to a JSON lines stream
    pub fn json_lines<T: serde::de::DeserializeOwned>(self) -> JsonLinesStream<T> {
        JsonLinesStream::new(self)
    }
}

impl Stream for HttpStream {
    type Item = HttpResult<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        // Poll the underlying bytes stream
        match this.inner.poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                // Convert reqwest::Bytes to bytes::Bytes and return
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                // Convert reqwest error to HttpError
                Poll::Ready(Some(Err(HttpError::NetworkError {
                    message: e.to_string(),
                })))
            }
            Poll::Ready(None) => {
                // End of stream
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

pin_project! {
    /// Lines stream that splits HTTP stream by newlines
    pub struct LinesStream {
        #[pin]
        inner: HttpStream,
        buffer: String,
    }
}

impl LinesStream {
    /// Create a new lines stream
    pub fn new(stream: HttpStream) -> Self {
        Self {
            inner: stream,
            buffer: String::new(),
        }
    }
}

impl Stream for LinesStream {
    type Item = HttpResult<String>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            // Check if we have a complete line in the buffer
            if let Some(newline_pos) = this.buffer.find('\n') {
                let line = this.buffer.drain(..=newline_pos).collect::<String>();
                let line = line.trim_end_matches('\n').trim_end_matches('\r');
                return Poll::Ready(Some(Ok(line.to_string())));
            }

            // Read more data
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    this.buffer.push_str(&chunk_str);
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    // End of stream, return remaining buffer if any
                    if !this.buffer.is_empty() {
                        let line = this.buffer.drain(..).collect();
                        return Poll::Ready(Some(Ok(line)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

pin_project! {
    /// Server-Sent Events stream
    pub struct SseStream {
        #[pin]
        inner: LinesStream,
        current_event: SseEvent,
    }
}

/// Server-Sent Event structure
#[derive(Debug, Clone, Default)]
pub struct SseEvent {
    /// Event type (optional)
    pub event_type: Option<String>,
    /// Event data lines
    pub data: Vec<String>,
    /// Event ID (optional)
    pub id: Option<String>,
    /// Retry timeout in milliseconds (optional)
    pub retry: Option<u64>,
}

/// Download chunk containing data and metadata for file downloads
#[derive(Debug, Clone)]
pub struct DownloadChunk {
    /// The actual data bytes (zero-copy using Bytes)
    pub data: bytes::Bytes,
    /// Sequential chunk number starting from 0
    pub chunk_number: u64,
    /// Total file size if known from Content-Length header
    pub total_size: Option<u64>,
    /// Total bytes downloaded so far (cumulative)
    pub bytes_downloaded: u64,
    /// Timestamp when this chunk was received
    pub timestamp: std::time::Instant,
    /// Download speed in bytes per second (calculated)
    pub download_speed: Option<f64>,
}

impl DownloadChunk {
    /// Create a new download chunk
    pub fn new(
        data: bytes::Bytes,
        chunk_number: u64,
        total_size: Option<u64>,
        bytes_downloaded: u64,
        download_speed: Option<f64>,
    ) -> Self {
        Self {
            data,
            chunk_number,
            total_size,
            bytes_downloaded,
            timestamp: std::time::Instant::now(),
            download_speed,
        }
    }

    /// Get the size of this chunk in bytes
    pub fn chunk_size(&self) -> usize {
        self.data.len()
    }

    /// Calculate download progress as a percentage (0.0 to 100.0)
    pub fn progress_percentage(&self) -> Option<f64> {
        self.total_size.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.bytes_downloaded as f64 / total as f64) * 100.0
            }
        })
    }

    /// Check if this is the final chunk (when total size is known)
    pub fn is_final_chunk(&self) -> Option<bool> {
        self.total_size.map(|total| self.bytes_downloaded >= total)
    }

    /// Get estimated time remaining based on current download speed
    pub fn estimated_time_remaining(&self) -> Option<std::time::Duration> {
        match (self.total_size, self.download_speed) {
            (Some(total), Some(speed)) if speed > 0.0 => {
                let remaining_bytes = total.saturating_sub(self.bytes_downloaded) as f64;
                let remaining_seconds = remaining_bytes / speed;
                Some(std::time::Duration::from_secs_f64(remaining_seconds))
            }
            _ => None,
        }
    }
}

pin_project! {
    /// Download stream for file downloads with progress tracking
    pub struct DownloadStream {
        #[pin]
        stream: futures::stream::BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
        chunk_number: u64,
        bytes_downloaded: u64,
        total_size: Option<u64>,
        start_time: std::time::Instant,
        last_chunk_time: std::time::Instant,
        on_chunk_handler: Option<Box<dyn Fn(&DownloadChunk) -> crate::HttpResult<()> + Send + Sync>>,
    }
}

impl DownloadStream {
    /// Create a new download stream from a reqwest Response
    pub fn new(response: reqwest::Response) -> Self {
        use futures::StreamExt;

        let total_size = response.content_length().filter(|&size| size > 0);

        // Convert the response into a stream of bytes
        let stream = response.bytes_stream().boxed();

        let now = std::time::Instant::now();
        Self {
            stream,
            chunk_number: 0,
            bytes_downloaded: 0,
            total_size,
            start_time: now,
            last_chunk_time: now,
            on_chunk_handler: None,
        }
    }

    /// Set an on_chunk handler for processing chunks as they arrive
    pub fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(&DownloadChunk) -> crate::HttpResult<()> + Send + Sync + 'static,
    {
        self.on_chunk_handler = Some(Box::new(handler));
        self
    }
}

impl Stream for DownloadStream {
    type Item = crate::HttpResult<DownloadChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().project();

        // Poll the underlying bytes stream
        match this.stream.poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                let chunk_size = chunk.len();
                *this.bytes_downloaded += chunk_size as u64;
                let chunk_number = *this.chunk_number;
                *this.chunk_number += 1;
                *this.last_chunk_time = std::time::Instant::now();

                // Calculate download speed using projected fields
                let download_speed = {
                    let elapsed = this.start_time.elapsed();
                    if elapsed.as_secs_f64() > 0.0 && *this.bytes_downloaded > 0 {
                        Some(*this.bytes_downloaded as f64 / elapsed.as_secs_f64())
                    } else {
                        None
                    }
                };

                let download_chunk = DownloadChunk::new(
                    chunk,
                    chunk_number,
                    *this.total_size,
                    *this.bytes_downloaded,
                    download_speed,
                );

                // Call the on_chunk handler if set
                if let Some(handler) = this.on_chunk_handler.as_ref() {
                    if let Err(e) = handler(&download_chunk) {
                        return Poll::Ready(Some(Err(e)));
                    }
                }

                Poll::Ready(Some(Ok(download_chunk)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(crate::HttpError::NetworkError {
                message: format!("Download stream error: {}", e),
            }))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl SseStream {
    /// Create a new SSE stream
    pub fn new(stream: HttpStream) -> Self {
        Self {
            inner: stream.lines(),
            current_event: SseEvent::default(),
        }
    }
}

impl Stream for SseStream {
    type Item = HttpResult<SseEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(line))) => {
                    if line.is_empty() {
                        // Empty line signals end of event
                        if !this.current_event.data.is_empty()
                            || this.current_event.event_type.is_some()
                            || this.current_event.id.is_some()
                        {
                            let event = std::mem::take(this.current_event);
                            return Poll::Ready(Some(Ok(event)));
                        }
                        continue;
                    }

                    if line.starts_with(':') {
                        // Comment line, ignore
                        continue;
                    }

                    if let Some(colon_pos) = line.find(':') {
                        let field = &line[..colon_pos];
                        let value = line[colon_pos + 1..].trim_start();

                        match field {
                            "event" => this.current_event.event_type = Some(value.to_string()),
                            "data" => this.current_event.data.push(value.to_string()),
                            "id" => this.current_event.id = Some(value.to_string()),
                            "retry" => {
                                if let Ok(retry) = value.parse::<u64>() {
                                    this.current_event.retry = Some(retry);
                                }
                            }
                            _ => {} // Unknown field, ignore
                        }
                    } else {
                        // Line without colon is treated as data
                        this.current_event.data.push(line);
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    // End of stream, return current event if any
                    if !this.current_event.data.is_empty()
                        || this.current_event.event_type.is_some()
                        || this.current_event.id.is_some()
                    {
                        let event = std::mem::take(this.current_event);
                        return Poll::Ready(Some(Ok(event)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

pin_project! {
    /// JSON lines stream that parses each line as JSON
    pub struct JsonLinesStream<T> {
        #[pin]
        inner: LinesStream,
        _phantom: std::marker::PhantomData<T>,
    }
}

impl<T> JsonLinesStream<T> {
    /// Create a new JSON lines stream
    pub fn new(stream: HttpStream) -> Self {
        Self {
            inner: stream.lines(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: serde::de::DeserializeOwned> Stream for JsonLinesStream<T> {
    type Item = HttpResult<T>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(line))) => {
                if line.trim().is_empty() {
                    // Skip empty lines - continue polling
                    return Poll::Pending;
                }

                match serde_json::from_str::<T>(&line) {
                    Ok(value) => Poll::Ready(Some(Ok(value))),
                    Err(e) => Poll::Ready(Some(Err(HttpError::DeserializationError {
                        message: format!("Failed to parse JSON line: {}", e),
                    }))),
                }
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl SseEvent {
    /// Get the event data as a single string
    pub fn data_string(&self) -> String {
        self.data.join("\n")
    }

    /// Check if this is a specific event type
    pub fn is_event_type(&self, event_type: &str) -> bool {
        self.event_type.as_ref().map_or(false, |t| t == event_type)
    }

    /// Check if this is a "done" event (common in AI streaming)
    pub fn is_done(&self) -> bool {
        self.data_string().trim() == "[DONE]"
    }

    /// Parse the data as JSON
    pub fn parse_json<T: serde::de::DeserializeOwned>(&self) -> HttpResult<T> {
        let data = self.data_string();
        serde_json::from_str(&data).map_err(|e| HttpError::DeserializationError {
            message: format!("Failed to parse SSE data as JSON: {}", e),
        })
    }
}

pin_project! {
    /// Cached download stream that reads from a local file and emits DownloadChunk items
    /// This provides the same interface as DownloadStream but sources data from cache
    pub struct CachedDownloadStream {
        file_path: std::path::PathBuf,
        #[pin]
        file_stream: Option<futures::stream::BoxStream<'static, std::io::Result<Bytes>>>,
        chunk_number: u64,
        bytes_downloaded: u64,
        total_size: Option<u64>,
        start_time: std::time::Instant,
        last_chunk_time: std::time::Instant,
        chunk_size: usize,
        etag: Option<String>,
        computed_expires: Option<u64>,
    }
}

impl CachedDownloadStream {
    /// Create a new cached download stream from a file path
    pub async fn from_file<P: AsRef<Path>>(
        path: P,
        etag: Option<String>,
        computed_expires: Option<u64>,
    ) -> HttpResult<Self> {
        let file_path = path.as_ref().to_path_buf();
        
        // Get file size
        let total_size = std::fs::metadata(&file_path)
            .map_err(|e| HttpError::IoError {
                message: format!("Failed to read file metadata: {}", e),
            })?
            .len();

        let now = std::time::Instant::now();

        Ok(Self {
            file_path,
            file_stream: None,
            chunk_number: 0,
            bytes_downloaded: 0,
            total_size: Some(total_size),
            start_time: now,
            last_chunk_time: now,
            chunk_size: 8192, // 8KB chunks
            etag,
            computed_expires,
        })
    }

    /// Get the ETag for this cached file
    pub fn etag(&self) -> Option<&String> {
        self.etag.as_ref()
    }

    /// Get the computed expires timestamp
    pub fn computed_expires(&self) -> Option<u64> {
        self.computed_expires
    }

    /// Check if the cached file has expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires) = self.computed_expires {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            
            now >= expires
        } else {
            false
        }
    }

    /// Set chunk size for streaming
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

}

impl Stream for CachedDownloadStream {
    type Item = HttpResult<DownloadChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // Initialize file stream if needed
        if this.file_stream.is_none() {
            use futures::StreamExt;
            use tokio::io::AsyncReadExt;

            let file_path = this.file_path.clone();
            let chunk_size = *this.chunk_size;

            // Create async file reader stream
            let stream = async_stream::stream! {
                let mut file = match tokio::fs::File::open(&file_path).await {
                    Ok(f) => f,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };

                let mut buffer = vec![0u8; chunk_size];
                loop {
                    match file.read(&mut buffer).await {
                        Ok(0) => break, // EOF
                        Ok(n) => {
                            let chunk = Bytes::copy_from_slice(&buffer[..n]);
                            yield Ok(chunk);
                        }
                        Err(e) => {
                            yield Err(e);
                            break;
                        }
                    }
                }
            };

            this.file_stream.set(Some(stream.boxed()));
        }

        match this.file_stream.as_mut().as_pin_mut() {
            Some(stream) => match stream.poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    let now = std::time::Instant::now();
                    let chunk_size = bytes.len() as u64;
                    *this.bytes_downloaded += chunk_size;
                    let chunk_number = *this.chunk_number;
                    *this.chunk_number += 1;

                    // Calculate download speed (from cache is very fast)
                    let elapsed = now.duration_since(*this.last_chunk_time).as_secs_f64();
                    let speed = if elapsed > 0.0 {
                        Some(chunk_size as f64 / elapsed)
                    } else {
                        None
                    };

                    *this.last_chunk_time = now;

                    let chunk = DownloadChunk {
                        data: bytes,
                        chunk_number,
                        total_size: *this.total_size,
                        bytes_downloaded: *this.bytes_downloaded,
                        timestamp: now,
                        download_speed: speed,
                    };

                    Poll::Ready(Some(Ok(chunk)))
                }
                Poll::Ready(Some(Err(e))) => {
                    Poll::Ready(Some(Err(HttpError::IoError {
                        message: format!("Failed to read cached file: {}", e),
                    })))
                }
                Poll::Ready(None) => Poll::Ready(None),
                Poll::Pending => Poll::Pending,
            },
            None => Poll::Ready(Some(Err(HttpError::IoError {
                message: "File stream not initialized".to_string(),
            }))),
        }
    }
}

impl DownloadStream {
    /// Create a download stream from a cached file
    pub async fn from_file<P: AsRef<Path>>(path: P) -> HttpResult<CachedDownloadStream> {
        CachedDownloadStream::from_file(path, None, None).await
    }

    /// Create a download stream from a cached file with metadata
    pub async fn from_file_with_metadata<P: AsRef<Path>>(
        path: P,
        etag: Option<String>,
        computed_expires: Option<u64>,
    ) -> HttpResult<CachedDownloadStream> {
        CachedDownloadStream::from_file(path, etag, computed_expires).await
    }
}
