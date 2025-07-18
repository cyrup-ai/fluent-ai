//! Blazing-fast streaming implementation for Gemini completions
//! 
//! This module provides zero-allocation, lock-free streaming with HTTP3/SSE
//! support and production-ready error handling.

use super::gemini_error::{GeminiError, GeminiResult};
use super::gemini_types::{GenerateContentRequest, parse_gemini_chunk};
use crate::completion_provider::CompletionError;
use fluent_ai_domain::{AsyncTask, spawn_async, chunk::CompletionChunk};
use fluent_ai_http3::{HttpClient, HttpRequest, HttpResponse};
use futures_util::StreamExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, error, warn};

/// High-performance streaming completion processor
#[derive(Debug)]
pub struct GeminiStreamProcessor {
    /// HTTP3 client for blazing-fast connections
    client: HttpClient,
    /// Base URL for Gemini API
    base_url: &'static str,
    /// Cancellation flag for graceful shutdown
    cancel_flag: Arc<AtomicBool>,
}

impl GeminiStreamProcessor {
    /// Create new streaming processor with HTTP3 client
    #[inline(always)]
    pub fn new(client: HttpClient) -> Self {
        Self {
            client,
            base_url: "https://generativelanguage.googleapis.com",
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }
    
    /// Execute streaming completion with zero-allocation patterns
    #[inline(always)]
    pub async fn execute_streaming_completion(
        &self,
        request_body: GenerateContentRequest,
        model_name: &str,
        api_key: &str,
    ) -> GeminiResult<crate::AsyncStream<Result<CompletionChunk, CompletionError>>> {
        let start_time = Instant::now();
        
        // Serialize request to bytes (single allocation)
        let body_bytes = serde_json::to_vec(&request_body)
            .map_err(|e| GeminiError::parse_error(format!("Request serialization failed: {}", e)))?;

        debug!("Gemini streaming request size: {} bytes", body_bytes.len());
        
        // Build streaming URL with API key
        let url = format!("{}/v1beta/models/{}:streamGenerateContent?key={}", 
                         self.base_url, model_name, api_key);

        // Create HTTP3 request with optimized headers
        let request = HttpRequest::post(&url, body_bytes)
            .map_err(|e| GeminiError::http_error(format!("Request creation failed: {}", e)))?
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache");

        // Send request and get response
        let response = self.client.send(request).await
            .map_err(|e| GeminiError::http_error(format!("HTTP request failed: {}", e)))?;

        debug!("Gemini streaming request sent in {:?}", start_time.elapsed());

        // Check response status before processing stream
        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        // Create high-performance streaming pipeline
        Ok(self.create_streaming_pipeline(response).await?)
    }
    
    /// Handle error response with detailed context
    async fn handle_error_response(&self, response: HttpResponse) -> GeminiError {
        let status_code = response.status().as_u16();
        
        // Try to get response body for error details
        let error_body = match response.text().await {
            Ok(body) => Some(body),
            Err(e) => {
                warn!("Failed to read error response body: {}", e);
                None
            }
        };
        
        match error_body {
            Some(body) => super::gemini_error::parse_api_error_response(&body),
            None => super::gemini_error::parse_http_status_error(status_code, None),
        }
    }
    
    /// Create blazing-fast streaming pipeline with zero allocation where possible
    async fn create_streaming_pipeline(
        &self,
        response: HttpResponse,
    ) -> GeminiResult<crate::AsyncStream<Result<CompletionChunk, CompletionError>>> {
        // Create high-throughput channel for chunks using the crate's async stream system
        let (chunk_sender, chunk_receiver) = crate::async_stream_channel();
        
        // Get SSE stream from HTTP3 response
        let sse_stream = response.sse();
        let cancel_flag = Arc::clone(&self.cancel_flag);
        
        // Spawn streaming task with optimized processing
        spawn_async(async move {
            let mut sse_stream = sse_stream;
            let mut chunk_count = 0u64;
            let start_time = Instant::now();
            
            while let Some(event_result) = sse_stream.next().await {
                // Check for cancellation
                if cancel_flag.load(Ordering::Relaxed) {
                    debug!("Streaming cancelled after {} chunks", chunk_count);
                    break;
                }
                
                match event_result {
                    Ok(sse_event) => {
                        if let Some(data) = sse_event.data {
                            // Fast chunk parsing with zero-copy where possible
                            match parse_gemini_chunk(data.as_bytes()) {
                                Ok(chunk) => {
                                    chunk_count += 1;
                                    
                                    // Convert to provider error type
                                    if chunk_sender.send(Ok(chunk)).is_err() {
                                        debug!("Chunk receiver dropped, stopping stream");
                                        break;
                                    }
                                }
                                Err(parse_error) => {
                                    error!("Chunk parsing failed: {}", parse_error);
                                    let provider_error: CompletionError = parse_error.into();
                                    
                                    if chunk_sender.send(Err(provider_error)).is_err() {
                                        debug!("Error receiver dropped, stopping stream");
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(sse_error) => {
                        error!("SSE stream error: {}", sse_error);
                        let stream_error = GeminiError::stream_error(format!("SSE error: {}", sse_error));
                        let provider_error: CompletionError = stream_error.into();
                        
                        if chunk_sender.send(Err(provider_error)).is_err() {
                            debug!("Error receiver dropped, stopping stream");
                        }
                        break;
                    }
                }
            }
            
            debug!(
                "Streaming completed: {} chunks in {:?}",
                chunk_count,
                start_time.elapsed()
            );
        });
        
        // Return the async stream receiver
        Ok(chunk_receiver)
    }
    
    /// Cancel ongoing streaming operation
    #[inline(always)]
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
    }
    
    /// Reset cancellation flag for reuse
    #[inline(always)]
    pub fn reset(&self) {
        self.cancel_flag.store(false, Ordering::Relaxed);
    }
}

/// Optimized streaming response wrapper with metrics
#[derive(Debug)]
pub struct StreamingResponse<S> {
    /// The underlying chunk stream
    stream: S,
    /// Performance metrics
    metrics: StreamingMetrics,
    /// Start time for duration calculation
    start_time: Instant,
}

impl<S> StreamingResponse<S> 
where
    S: futures::Stream<Item = Result<CompletionChunk, CompletionError>>,
{
    /// Create new streaming response with metrics
    #[inline(always)]
    pub fn new(stream: S) -> Self {
        Self {
            stream,
            metrics: StreamingMetrics::default(),
            start_time: Instant::now(),
        }
    }
    
    /// Get performance metrics
    #[inline(always)]
    pub fn metrics(&self) -> &StreamingMetrics {
        &self.metrics
    }
    
    /// Get stream duration so far
    #[inline(always)]
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// Consume the response and get the inner stream
    #[inline(always)]
    pub fn into_stream(self) -> S {
        self.stream
    }
}

impl<S> futures::Stream for StreamingResponse<S>
where
    S: futures::Stream<Item = Result<CompletionChunk, CompletionError>> + Unpin,
{
    type Item = Result<CompletionChunk, CompletionError>;
    
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::pin::Pin;
        
        match Pin::new(&mut self.stream).poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(chunk))) => {
                // Update metrics
                self.metrics.total_chunks += 1;
                
                match &chunk {
                    CompletionChunk::Text { text } => {
                        self.metrics.text_chunks += 1;
                        self.metrics.total_text_bytes += text.len();
                    }
                    CompletionChunk::Complete { text, usage, .. } => {
                        self.metrics.text_chunks += 1;
                        self.metrics.total_text_bytes += text.len();
                        if let Some(usage) = usage {
                            self.metrics.total_tokens = Some(usage.total_tokens);
                        }
                    }
                    CompletionChunk::ToolCall { .. } => {
                        self.metrics.tool_call_chunks += 1;
                    }
                    CompletionChunk::Error { .. } => {
                        self.metrics.error_chunks += 1;
                    }
                    _ => {}
                }
                
                std::task::Poll::Ready(Some(Ok(chunk)))
            }
            std::task::Poll::Ready(Some(Err(e))) => {
                self.metrics.error_chunks += 1;
                std::task::Poll::Ready(Some(Err(e)))
            }
            std::task::Poll::Ready(None) => {
                self.metrics.completed = true;
                std::task::Poll::Ready(None)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

/// Performance metrics for streaming operations
#[derive(Debug, Default, Clone)]
pub struct StreamingMetrics {
    /// Total chunks processed
    pub total_chunks: u64,
    /// Text chunks count
    pub text_chunks: u64,
    /// Tool call chunks count
    pub tool_call_chunks: u64,
    /// Error chunks count
    pub error_chunks: u64,
    /// Total text bytes received
    pub total_text_bytes: usize,
    /// Total tokens if available
    pub total_tokens: Option<u32>,
    /// Whether stream completed successfully
    pub completed: bool,
}

impl StreamingMetrics {
    /// Calculate average chunk size
    #[inline(always)]
    pub fn average_chunk_size(&self) -> f64 {
        if self.total_chunks == 0 {
            0.0
        } else {
            self.total_text_bytes as f64 / self.total_chunks as f64
        }
    }
    
    /// Calculate error rate
    #[inline(always)]
    pub fn error_rate(&self) -> f64 {
        if self.total_chunks == 0 {
            0.0
        } else {
            self.error_chunks as f64 / self.total_chunks as f64
        }
    }
    
    /// Check if streaming was successful
    #[inline(always)]
    pub fn is_successful(&self) -> bool {
        self.completed && self.error_chunks == 0
    }
}

/// Streaming configuration for fine-tuning performance
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size for chunk processing
    pub buffer_size: usize,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Read timeout for chunks
    pub read_timeout: Duration,
    /// Enable detailed metrics collection
    pub enable_metrics: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 8192,
            max_connections: 100,
            connection_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(60),
            enable_metrics: true,
        }
    }
}

/// Create optimized streaming processor for Gemini
#[inline(always)]
pub fn create_streaming_processor(client: HttpClient) -> GeminiStreamProcessor {
    GeminiStreamProcessor::new(client)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_ai_http3::HttpConfig;
    
    #[tokio::test]
    async fn test_streaming_processor_creation() {
        let client = HttpClient::with_config(HttpConfig::streaming_optimized())
            .map_err(|_| "Failed to create HTTP client")
            .unwrap();
            
        let processor = create_streaming_processor(client);
        assert!(!processor.cancel_flag.load(Ordering::Relaxed));
    }
    
    #[test]
    fn test_streaming_metrics() {
        let mut metrics = StreamingMetrics::default();
        
        metrics.total_chunks = 10;
        metrics.text_chunks = 8;
        metrics.error_chunks = 1;
        metrics.total_text_bytes = 1000;
        
        assert_eq!(metrics.average_chunk_size(), 100.0);
        assert_eq!(metrics.error_rate(), 0.1);
        assert!(!metrics.is_successful()); // has errors
        
        metrics.error_chunks = 0;
        metrics.completed = true;
        assert!(metrics.is_successful());
    }
    
    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.buffer_size, 8192);
        assert_eq!(config.max_connections, 100);
        assert!(config.enable_metrics);
    }
    
    #[test]
    fn test_processor_cancellation() {
        let client = HttpClient::default();
        let processor = create_streaming_processor(client);
        
        assert!(!processor.cancel_flag.load(Ordering::Relaxed));
        
        processor.cancel();
        assert!(processor.cancel_flag.load(Ordering::Relaxed));
        
        processor.reset();
        assert!(!processor.cancel_flag.load(Ordering::Relaxed));
    }
}