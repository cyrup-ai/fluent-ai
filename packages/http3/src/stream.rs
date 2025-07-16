//! HTTP streaming utilities

use crate::{HttpError, HttpResult};
use futures::{Stream, StreamExt};
use pin_project_lite::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};
use bytes::Bytes;

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
    /// Create a new HTTP stream from a reqwest Response
    pub fn new(response: reqwest::Response) -> Self {
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
        String::from_utf8(bytes).map_err(|e| {
            HttpError::DeserializationError {
                message: format!("Invalid UTF-8 in stream: {}", e),
            }
        })
    }
    
    /// Read the entire stream and parse as JSON
    pub async fn collect_json<T: serde::de::DeserializeOwned>(self) -> HttpResult<T> {
        let bytes = self.collect().await?;
        serde_json::from_slice(&bytes).map_err(|e| {
            HttpError::DeserializationError {
                message: format!("Failed to parse JSON from stream: {}", e),
            }
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
                        if !this.current_event.data.is_empty() || 
                           this.current_event.event_type.is_some() ||
                           this.current_event.id.is_some() {
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
                    if !this.current_event.data.is_empty() || 
                       this.current_event.event_type.is_some() ||
                       this.current_event.id.is_some() {
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
            Poll::Ready(Some(Err(e))) => {
                Poll::Ready(Some(Err(e)))
            }
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
        serde_json::from_str(&data).map_err(|e| {
            HttpError::DeserializationError {
                message: format!("Failed to parse SSE data as JSON: {}", e),
            }
        })
    }
}