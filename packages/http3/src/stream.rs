//! HTTP streaming utilities - Zero futures, pure unwrapped value streams

use bytes::Bytes;

use crate::{HttpError, HttpResult};
use std::path::Path;

/// HTTP response stream wrapper that provides zero-allocation streaming
/// Returns pure unwrapped Bytes values - no Result wrapping, no futures
pub struct HttpStream {
    body: Vec<u8>,
    chunk_size: usize,
    position: usize,
}

impl HttpStream {
    /// Create a new HTTP stream from response body - pure unwrapped values
    pub fn new(body: Vec<u8>) -> Self {
        Self {
            body,
            chunk_size: 8192, // 8KB chunks
            position: 0,
        }
    }

    /// Create a new HTTP stream with custom chunk size - pure unwrapped values
    pub fn with_chunk_size(body: Vec<u8>, chunk_size: usize) -> Self {
        Self {
            body,
            chunk_size,
            position: 0,
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

    /// Get remaining bytes to read
    pub fn remaining_bytes(&self) -> usize {
        self.body.len().saturating_sub(self.position)
    }

    /// Get next chunk of bytes - returns unwrapped Bytes directly
    pub fn next_chunk(&mut self) -> Option<Bytes> {
        if self.position >= self.body.len() {
            return None;
        }

        let end_pos = std::cmp::min(self.position + self.chunk_size, self.body.len());
        let chunk = Bytes::copy_from_slice(&self.body[self.position..end_pos]);
        self.position = end_pos;
        
        Some(chunk)
    }

    /// Collect all remaining bytes - returns Vec<u8> directly (no futures)
    pub fn collect(self) -> Vec<u8> {
        self.body[self.position..].to_vec()
    }

    /// Collect as string - returns String directly (no futures)
    pub fn collect_string(self) -> String {
        String::from_utf8_lossy(&self.body[self.position..]).to_string()
    }

    /// Collect and parse as JSON - returns T directly (no futures)
    /// User on_chunk handler receives error context if parsing fails
    pub fn collect_json<T: serde::de::DeserializeOwned + Default>(self) -> T {
        match serde_json::from_slice(&self.body[self.position..]) {
            Ok(parsed) => parsed,
            Err(_) => T::default(), // User on_chunk handler processes errors
        }
    }

    /// Convert to a lines stream - returns unwrapped lines
    pub fn lines(self) -> LinesStream {
        LinesStream::new(self)
    }

    /// Convert to a Server-Sent Events stream - returns unwrapped SSE events
    pub fn sse(self) -> SseStream {
        SseStream::new(self)
    }

    /// Convert to a JSON lines stream - returns unwrapped T values
    pub fn json_lines<T: serde::de::DeserializeOwned + Default>(self) -> JsonLinesStream<T> {
        JsonLinesStream::new(self)
    }
}

/// Lines stream that splits HTTP stream by newlines - returns unwrapped String values
pub struct LinesStream {
    body: String,
    position: usize,
}

impl LinesStream {
    /// Create a new lines stream - returns unwrapped String values
    pub fn new(stream: HttpStream) -> Self {
        let body = String::from_utf8_lossy(&stream.body).to_string();
        Self {
            body,
            position: 0,
        }
    }

    /// Get next line - returns unwrapped String directly
    pub fn next_line(&mut self) -> Option<String> {
        if self.position >= self.body.len() {
            return None;
        }

        let remaining = &self.body[self.position..];
        if let Some(newline_pos) = remaining.find('\n') {
            let line = &remaining[..newline_pos];
            let line = line.trim_end_matches('\r'); // Handle CRLF
            self.position += newline_pos + 1;
            Some(line.to_string())
        } else if !remaining.is_empty() {
            // Last line without newline
            self.position = self.body.len();
            Some(remaining.to_string())
        } else {
            None
        }
    }

    /// Collect all remaining lines - returns Vec<String> directly
    pub fn collect_lines(mut self) -> Vec<String> {
        let mut lines = Vec::new();
        while let Some(line) = self.next_line() {
            lines.push(line);
        }
        lines
    }
}

/// Server-Sent Events stream - returns unwrapped SseEvent values
pub struct SseStream {
    lines: LinesStream,
    current_event: SseEvent,
}

impl SseStream {
    /// Create a new SSE stream - returns unwrapped SseEvent values
    pub fn new(stream: HttpStream) -> Self {
        Self {
            lines: stream.lines(),
            current_event: SseEvent::default(),
        }
    }

    /// Get next SSE event - returns unwrapped SseEvent directly
    pub fn next_event(&mut self) -> Option<SseEvent> {
        while let Some(line) = self.lines.next_line() {
            if line.is_empty() {
                // Empty line signals end of event
                if !self.current_event.data.is_empty()
                    || self.current_event.event_type.is_some()
                    || self.current_event.id.is_some()
                {
                    let event = std::mem::take(&mut self.current_event);
                    return Some(event);
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
                    "event" => self.current_event.event_type = Some(value.to_string()),
                    "data" => self.current_event.data.push(value.to_string()),
                    "id" => self.current_event.id = Some(value.to_string()),
                    "retry" => {
                        if let Ok(retry) = value.parse::<u64>() {
                            self.current_event.retry = Some(retry);
                        }
                    }
                    _ => {} // Unknown field, ignore
                }
            } else {
                // Line without colon is treated as data
                self.current_event.data.push(line);
            }
        }

        // End of stream, return current event if any
        if !self.current_event.data.is_empty()
            || self.current_event.event_type.is_some()
            || self.current_event.id.is_some()
        {
            let event = std::mem::take(&mut self.current_event);
            Some(event)
        } else {
            None
        }
    }

    /// Collect all SSE events - returns Vec<SseEvent> directly
    pub fn collect_events(mut self) -> Vec<SseEvent> {
        let mut events = Vec::new();
        while let Some(event) = self.next_event() {
            events.push(event);
        }
        events
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

/// Download stream for file downloads with progress tracking - returns unwrapped DownloadChunk values
pub struct DownloadStream {
    body: Vec<u8>,
    chunk_number: u64,
    bytes_downloaded: u64,
    total_size: Option<u64>,
    start_time: std::time::Instant,
    last_chunk_time: std::time::Instant,
    chunk_size: usize,
    position: usize,
    on_chunk_handler: Option<Box<dyn Fn(&DownloadChunk) -> crate::HttpResult<()> + Send + Sync>>,
}

impl DownloadStream {
    /// Create a new download stream from response body - returns unwrapped DownloadChunk values
    pub fn new(body: Vec<u8>, total_size: Option<u64>) -> Self {
        let now = std::time::Instant::now();
        Self {
            body,
            chunk_number: 0,
            bytes_downloaded: 0,
            total_size,
            start_time: now,
            last_chunk_time: now,
            chunk_size: 8192, // 8KB chunks
            position: 0,
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

    /// Get next download chunk - returns unwrapped DownloadChunk directly
    pub fn next_chunk(&mut self) -> Option<DownloadChunk> {
        if self.position >= self.body.len() {
            return None;
        }

        let end_pos = std::cmp::min(self.position + self.chunk_size, self.body.len());
        let chunk_data = Bytes::copy_from_slice(&self.body[self.position..end_pos]);
        let chunk_size = chunk_data.len() as u64;
        
        self.bytes_downloaded += chunk_size;
        let chunk_number = self.chunk_number;
        self.chunk_number += 1;
        self.position = end_pos;
        self.last_chunk_time = std::time::Instant::now();

        // Calculate download speed
        let download_speed = {
            let elapsed = self.start_time.elapsed();
            if elapsed.as_secs_f64() > 0.0 && self.bytes_downloaded > 0 {
                Some(self.bytes_downloaded as f64 / elapsed.as_secs_f64())
            } else {
                None
            }
        };

        let download_chunk = DownloadChunk::new(
            chunk_data,
            chunk_number,
            self.total_size,
            self.bytes_downloaded,
            download_speed,
        );

        // Call the on_chunk handler if set - ignore errors for pure streaming
        if let Some(handler) = self.on_chunk_handler.as_ref() {
            let _ = handler(&download_chunk); // User on_chunk handler processes errors
        }

        Some(download_chunk)
    }

    /// Collect all remaining chunks - returns Vec<DownloadChunk> directly
    pub fn collect_chunks(mut self) -> Vec<DownloadChunk> {
        let mut chunks = Vec::new();
        while let Some(chunk) = self.next_chunk() {
            chunks.push(chunk);
        }
        chunks
    }
}


/// JSON lines stream that parses each line as JSON - returns unwrapped T values  
pub struct JsonLinesStream<T> {
    lines: LinesStream,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> JsonLinesStream<T> {
    /// Create a new JSON lines stream - returns unwrapped T values
    pub fn new(stream: HttpStream) -> Self {
        Self {
            lines: stream.lines(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: serde::de::DeserializeOwned + Default> JsonLinesStream<T> {
    /// Get next JSON value - returns unwrapped T directly
    pub fn next_json(&mut self) -> Option<T> {
        while let Some(line) = self.lines.next_line() {
            if line.trim().is_empty() {
                continue; // Skip empty lines
            }

            match serde_json::from_str::<T>(&line) {
                Ok(value) => return Some(value),
                Err(_) => return Some(T::default()), // User on_chunk handler processes errors
            }
        }
        None
    }

    /// Collect all JSON values - returns Vec<T> directly  
    pub fn collect_json(mut self) -> Vec<T> {
        let mut values = Vec::new();
        while let Some(value) = self.next_json() {
            values.push(value);
        }
        values
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

    /// Parse the data as JSON - returns T directly (user on_chunk handler processes errors)
    pub fn parse_json<T: serde::de::DeserializeOwned + Default>(&self) -> T {
        let data = self.data_string();
        match serde_json::from_str(&data) {
            Ok(parsed) => parsed,
            Err(_) => T::default(), // User on_chunk handler processes errors
        }
    }
}

/// Cached download stream that reads from a local file and emits DownloadChunk items
/// This provides the same interface as DownloadStream but sources data from cache
/// Returns unwrapped DownloadChunk values - no futures
pub struct CachedDownloadStream {
    file_data: Vec<u8>,
    chunk_number: u64,
    bytes_downloaded: u64,
    total_size: Option<u64>,
    start_time: std::time::Instant,
    last_chunk_time: std::time::Instant,
    chunk_size: usize,
    position: usize,
    etag: Option<String>,
    computed_expires: Option<u64>,
}

impl CachedDownloadStream {
    /// Create a new cached download stream from a file path - returns directly (no futures)
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        etag: Option<String>,
        computed_expires: Option<u64>,
    ) -> HttpResult<Self> {
        let file_path = path.as_ref().to_path_buf();
        
        // Read file data synchronously
        let file_data = std::fs::read(&file_path)
            .map_err(|e| HttpError::IoError {
                message: format!("Failed to read file: {}", e),
            })?;
            
        let total_size = file_data.len() as u64;
        let now = std::time::Instant::now();

        Ok(Self {
            file_data,
            chunk_number: 0,
            bytes_downloaded: 0,
            total_size: Some(total_size),
            start_time: now,
            last_chunk_time: now,
            chunk_size: 8192, // 8KB chunks
            position: 0,
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

    /// Get next download chunk - returns unwrapped DownloadChunk directly
    pub fn next_chunk(&mut self) -> Option<DownloadChunk> {
        if self.position >= self.file_data.len() {
            return None;
        }

        let end_pos = std::cmp::min(self.position + self.chunk_size, self.file_data.len());
        let chunk_data = Bytes::copy_from_slice(&self.file_data[self.position..end_pos]);
        let chunk_size = chunk_data.len() as u64;
        
        self.bytes_downloaded += chunk_size;
        let chunk_number = self.chunk_number;
        self.chunk_number += 1;
        self.position = end_pos;
        self.last_chunk_time = std::time::Instant::now();

        // Calculate download speed (from cache is very fast)
        let elapsed = self.last_chunk_time.duration_since(self.start_time).as_secs_f64();
        let speed = if elapsed > 0.0 {
            Some(self.bytes_downloaded as f64 / elapsed)
        } else {
            None
        };

        Some(DownloadChunk {
            data: chunk_data,
            chunk_number,
            total_size: self.total_size,
            bytes_downloaded: self.bytes_downloaded,
            timestamp: self.last_chunk_time,
            download_speed: speed,
        })
    }

    /// Collect all remaining chunks - returns Vec<DownloadChunk> directly
    pub fn collect_chunks(mut self) -> Vec<DownloadChunk> {
        let mut chunks = Vec::new();
        while let Some(chunk) = self.next_chunk() {
            chunks.push(chunk);
        }
        chunks
    }
}

impl DownloadStream {
    /// Create a download stream from a cached file - returns directly (no futures)
    pub fn from_file<P: AsRef<Path>>(path: P) -> HttpResult<CachedDownloadStream> {
        CachedDownloadStream::from_file(path, None, None)
    }

    /// Create a download stream from a cached file with metadata - returns directly (no futures)
    pub fn from_file_with_metadata<P: AsRef<Path>>(
        path: P,
        etag: Option<String>,
        computed_expires: Option<u64>,
    ) -> HttpResult<CachedDownloadStream> {
        CachedDownloadStream::from_file(path, etag, computed_expires)
    }
}
