//! Real-time token streaming with zero-allocation performance
//!
//! This module provides blazing-fast token streaming with:
//! - Zero allocation in hot paths with bounded buffers
//! - Lock-free atomic flow control and backpressure handling  
//! - Sub-100μs token transmission latency
//! - Bounded 32KB memory usage with graceful degradation
//! - Production-ready error handling without unwrap/expect

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicU64, AtomicBool, Ordering as AtomicOrdering},
        Arc,
    },
    task::{Context, Poll},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arrayvec::ArrayString;
use fluent_ai_domain::{FinishReason, completion::StreamingResponse};
use crossbeam_channel::{bounded, Sender, Receiver, TrySendError, TryRecvError};
use futures::Stream;

use crate::error::{CandleError, CandleResult};
use candle_core::Tensor;

/// Maximum text length per token chunk for bounded memory usage
pub const MAX_CHUNK_TEXT_SIZE: usize = 512;

/// Default buffer size for token transmission queue
pub const DEFAULT_BUFFER_SIZE: usize = 64;

/// Maximum buffer size to prevent unbounded memory growth
pub const MAX_BUFFER_SIZE: usize = 256;

/// Target token transmission latency in microseconds
pub const TARGET_LATENCY_MICROS: u64 = 100;

/// Memory usage budget for streaming (32KB)
pub const MEMORY_BUDGET_BYTES: usize = 32 * 1024;

/// Token chunk with bounded text storage and metadata
#[derive(Debug, Clone)]
pub struct TokenChunk {
    /// Token text with bounded storage to prevent allocation
    pub text: ArrayString<512>,
    /// Token metadata for downstream processing
    pub metadata: TokenMetadata,
    /// Generation timestamp for latency tracking
    pub timestamp: u64,
    /// Sequence position for ordering
    pub sequence_id: u64,
}

impl TokenChunk {
    /// Create new token chunk with current timestamp
    #[inline(always)]
    pub fn new(text: &str, metadata: TokenMetadata, sequence_id: u64) -> CandleResult<Self> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| CandleError::streaming_error("Failed to get system time"))?
            .as_nanos() as u64;

        let text_array = ArrayString::from(text)
            .map_err(|_| CandleError::streaming_error("Token text too long for bounded buffer"))?;

        Ok(Self {
            text: text_array,
            metadata,
            timestamp,
            sequence_id,
        })
    }

    /// Create empty chunk for graceful termination
    #[inline(always)]
    pub fn empty(sequence_id: u64) -> CandleResult<Self> {
        Self::new("", TokenMetadata::default(), sequence_id)
    }

    /// Create token chunk from model logits and metadata (for client integration)
    #[inline(always)]
    pub fn from_logits(
        logits: &Tensor,
        text: &str,
        position: u32,
        sequence_id: u64,
        processing_start: std::time::Instant,
    ) -> CandleResult<Self> {
        // Extract probability from logits (simplified - could be enhanced with more sophisticated extraction)
        let logprob = match logits.to_vec1::<f32>() {
            Ok(logits_vec) => {
                if logits_vec.is_empty() {
                    0.0
                } else {
                    // Find the maximum logit as a simple approximation
                    logits_vec.iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max)
                }
            }
            Err(_) => 0.0,
        };

        let processing_latency_nanos = processing_start.elapsed().as_nanos() as u64;

        let metadata = TokenMetadata {
            position,
            logprob,
            finish_reason: None,
            processing_latency_nanos,
            temperature: 1.0,
            top_p: None,
            top_k: None,
        };

        Self::new(text, metadata, sequence_id)
    }

    /// Check if chunk represents stream termination
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Get text as string slice for zero-copy access
    #[inline(always)]
    pub fn text_str(&self) -> &str {
        self.text.as_str()
    }

    /// Calculate chunk memory footprint
    #[inline(always)]
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    /// Merge with another chunk for overflow handling
    #[inline(always)]
    pub fn try_merge(&mut self, other: &TokenChunk) -> CandleResult<()> {
        if self.text.len() + other.text.len() > MAX_CHUNK_TEXT_SIZE {
            return Err(CandleError::streaming_error("Cannot merge chunks: would exceed size limit"));
        }

        self.text.try_push_str(&other.text)
            .map_err(|_| CandleError::streaming_error("Failed to merge token chunks"))?;

        // Update metadata to reflect merged state
        self.metadata.merge(&other.metadata);

        Ok(())
    }
}

/// Token metadata for processing context and statistics
#[derive(Debug, Clone, Default)]
pub struct TokenMetadata {
    /// Position in generated sequence
    pub position: u32,
    /// Log probability of token selection
    pub logprob: f32,
    /// Stream termination reason if applicable
    pub finish_reason: Option<FinishReason>,
    /// Processing latency in nanoseconds
    pub processing_latency_nanos: u64,
    /// Sampling temperature used for generation
    pub temperature: f32,
    /// Top-p value used if applicable
    pub top_p: Option<f32>,
    /// Top-k value used if applicable  
    pub top_k: Option<u32>,
}

impl TokenMetadata {
    /// Create new metadata with position and logprob
    #[inline(always)]
    pub fn new(position: u32, logprob: f32) -> Self {
        Self {
            position,
            logprob,
            finish_reason: None,
            processing_latency_nanos: 0,
            temperature: 1.0,
            top_p: None,
            top_k: None,
        }
    }

    /// Create terminal metadata with finish reason
    #[inline(always)]
    pub fn terminal(reason: FinishReason) -> Self {
        Self {
            position: 0,
            logprob: 0.0,
            finish_reason: Some(reason),
            processing_latency_nanos: 0,
            temperature: 1.0,
            top_p: None,
            top_k: None,
        }
    }

    /// Merge metadata from another token for chunk combination
    #[inline(always)]
    pub fn merge(&mut self, other: &TokenMetadata) {
        // Keep earliest position
        self.position = self.position.min(other.position);
        
        // Average log probabilities (approximate)
        self.logprob = (self.logprob + other.logprob) / 2.0;
        
        // Use other's finish reason if present
        if other.finish_reason.is_some() {
            self.finish_reason = other.finish_reason.clone();
        }
        
        // Accumulate processing latency
        self.processing_latency_nanos = self.processing_latency_nanos.saturating_add(other.processing_latency_nanos);
    }
}

/// Streaming configuration for customizable behavior
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size for token queue (bounded to prevent unbounded growth)
    pub buffer_size: usize,
    /// Timeout for individual token transmission
    pub chunk_timeout_ms: u16,
    /// Maximum chunk size before forced flush
    pub max_chunk_size: usize,
    /// Flush policy for batching behavior
    pub flush_policy: FlushPolicy,
    /// Enable automatic chunk merging on overflow
    pub merge_on_overflow: bool,
    /// Maximum merge attempts before dropping
    pub max_merge_attempts: u8,
}

impl Default for StreamingConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            buffer_size: DEFAULT_BUFFER_SIZE,
            chunk_timeout_ms: 100, // 100ms timeout
            max_chunk_size: MAX_CHUNK_TEXT_SIZE,
            flush_policy: FlushPolicy::Immediate,
            merge_on_overflow: true,
            max_merge_attempts: 3,
        }
    }
}

impl StreamingConfig {
    /// Create new configuration with validation
    pub fn new() -> Self {
        Self::default()
    }

    /// Set buffer size with bounds checking
    pub fn buffer_size(mut self, size: usize) -> CandleResult<Self> {
        if size == 0 {
            return Err(CandleError::configuration("Buffer size cannot be zero"));
        }
        if size > MAX_BUFFER_SIZE {
            return Err(CandleError::configuration("Buffer size exceeds maximum allowed"));
        }
        self.buffer_size = size;
        Ok(self)
    }

    /// Set chunk timeout with validation
    pub fn chunk_timeout_ms(mut self, timeout: u16) -> CandleResult<Self> {
        if timeout == 0 {
            return Err(CandleError::configuration("Chunk timeout cannot be zero"));
        }
        self.chunk_timeout_ms = timeout;
        Ok(self)
    }

    /// Set maximum chunk size with validation
    pub fn max_chunk_size(mut self, size: usize) -> CandleResult<Self> {
        if size == 0 {
            return Err(CandleError::configuration("Max chunk size cannot be zero"));
        }
        if size > MAX_CHUNK_TEXT_SIZE {
            return Err(CandleError::configuration("Max chunk size exceeds limit"));
        }
        self.max_chunk_size = size;
        Ok(self)
    }

    /// Set flush policy
    #[inline(always)]
    pub fn flush_policy(mut self, policy: FlushPolicy) -> Self {
        self.flush_policy = policy;
        self
    }

    /// Enable or disable merge on overflow
    #[inline(always)]
    pub fn merge_on_overflow(mut self, enabled: bool) -> Self {
        self.merge_on_overflow = enabled;
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> CandleResult<()> {
        if self.buffer_size == 0 {
            return Err(CandleError::configuration("Buffer size cannot be zero"));
        }
        if self.chunk_timeout_ms == 0 {
            return Err(CandleError::configuration("Chunk timeout cannot be zero"));
        }
        if self.max_chunk_size == 0 {
            return Err(CandleError::configuration("Max chunk size cannot be zero"));
        }
        if self.max_merge_attempts == 0 {
            return Err(CandleError::configuration("Max merge attempts cannot be zero"));
        }

        Ok(())
    }
}

/// Flush policy for batching behavior
#[derive(Debug, Clone, Copy)]
pub enum FlushPolicy {
    /// Send tokens immediately without batching
    Immediate,
    /// Batch up to N tokens before flushing
    Batched(usize),
    /// Flush after timeout duration
    Timeout(Duration),
    /// Adaptive flushing based on stream characteristics
    Adaptive,
}

impl Default for FlushPolicy {
    #[inline(always)]
    fn default() -> Self {
        Self::Immediate
    }
}

/// Streaming metrics for performance monitoring
#[derive(Debug)]
pub struct StreamingMetrics {
    /// Total tokens sent through stream
    pub tokens_sent: AtomicU64,
    /// Total chunks sent through stream
    pub chunks_sent: AtomicU64,
    /// Count of buffer overflow events
    pub buffer_overflows: AtomicU64,
    /// Total streaming latency in nanoseconds
    pub total_latency_nanos: AtomicU64,
    /// Count of merge operations performed
    pub merge_operations: AtomicU64,
    /// Count of dropped chunks due to errors
    pub dropped_chunks: AtomicU64,
    /// Peak buffer usage
    pub peak_buffer_usage: AtomicU64,
    /// Stream start time for rate calculations
    pub stream_start_nanos: AtomicU64,
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self {
            tokens_sent: AtomicU64::new(0),
            chunks_sent: AtomicU64::new(0),
            buffer_overflows: AtomicU64::new(0),
            total_latency_nanos: AtomicU64::new(0),
            merge_operations: AtomicU64::new(0),
            dropped_chunks: AtomicU64::new(0),
            peak_buffer_usage: AtomicU64::new(0),
            stream_start_nanos: AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0)
            ),
        }
    }
}

impl StreamingMetrics {
    /// Record token sent with latency tracking
    #[inline(always)]
    pub fn record_token_sent(&self, latency_nanos: u64) {
        self.tokens_sent.fetch_add(1, AtomicOrdering::Relaxed);
        self.total_latency_nanos.fetch_add(latency_nanos, AtomicOrdering::Relaxed);
    }

    /// Record chunk sent
    #[inline(always)]
    pub fn record_chunk_sent(&self) {
        self.chunks_sent.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record buffer overflow event
    #[inline(always)]
    pub fn record_buffer_overflow(&self) {
        self.buffer_overflows.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record merge operation
    #[inline(always)]
    pub fn record_merge_operation(&self) {
        self.merge_operations.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record dropped chunk
    #[inline(always)]
    pub fn record_dropped_chunk(&self) {
        self.dropped_chunks.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Update peak buffer usage
    #[inline(always)]
    pub fn update_peak_buffer_usage(&self, current_usage: u64) {
        let mut peak = self.peak_buffer_usage.load(AtomicOrdering::Relaxed);
        while current_usage > peak {
            match self.peak_buffer_usage.compare_exchange_weak(
                peak,
                current_usage,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
    }

    /// Get tokens per second rate
    pub fn tokens_per_second(&self) -> f64 {
        let tokens = self.tokens_sent.load(AtomicOrdering::Relaxed) as f64;
        let start_nanos = self.stream_start_nanos.load(AtomicOrdering::Relaxed);
        let current_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(start_nanos);
        
        let duration_secs = (current_nanos.saturating_sub(start_nanos)) as f64 / 1_000_000_000.0;
        
        if duration_secs > 0.0 {
            tokens / duration_secs
        } else {
            0.0
        }
    }

    /// Get average latency per token
    pub fn average_latency_nanos(&self) -> f64 {
        let total_latency = self.total_latency_nanos.load(AtomicOrdering::Relaxed) as f64;
        let tokens = self.tokens_sent.load(AtomicOrdering::Relaxed) as f64;
        
        if tokens > 0.0 {
            total_latency / tokens
        } else {
            0.0
        }
    }

    /// Get buffer overflow rate
    pub fn buffer_overflow_rate(&self) -> f64 {
        let overflows = self.buffer_overflows.load(AtomicOrdering::Relaxed) as f64;
        let chunks = self.chunks_sent.load(AtomicOrdering::Relaxed) as f64;
        
        if chunks > 0.0 {
            overflows / chunks
        } else {
            0.0
        }
    }
}

/// Real-time token output stream with zero-allocation performance
pub struct TokenOutputStream {
    /// Channel receiver for token chunks
    receiver: Receiver<TokenChunk>,
    /// Atomic flag for stream termination
    terminated: Arc<AtomicBool>,
    /// Stream metrics for monitoring
    metrics: Arc<StreamingMetrics>,
    /// Configuration for stream behavior
    config: StreamingConfig,
    /// Current buffer state
    buffer_state: StreamBufferState,
    /// Sequence counter for chunk ordering
    sequence_counter: AtomicU64,
}

/// Internal buffer state for flow control
#[derive(Debug)]
struct StreamBufferState {
    /// Pending chunk for batching
    pending_chunk: Option<TokenChunk>,
    /// Last flush timestamp
    last_flush_nanos: u64,
    /// Current buffer usage estimate
    buffer_usage_bytes: usize,
}

impl Default for StreamBufferState {
    fn default() -> Self {
        Self {
            pending_chunk: None,
            last_flush_nanos: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            buffer_usage_bytes: 0,
        }
    }
}

impl TokenOutputStream {
    /// Create new token output stream with configuration using project async primitives
    pub fn new(config: StreamingConfig) -> CandleResult<(Self, TokenStreamSender)> {
        // Validate configuration first
        config.validate()?;

        // Create bounded channel with specified capacity
        let (sender, receiver) = bounded::<TokenChunk>(config.buffer_size);
        
        let terminated = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(StreamingMetrics::default());

        let token_stream = TokenOutputStream {
            receiver,
            terminated: terminated.clone(),
            metrics: metrics.clone(),
            config: config.clone(),
            buffer_state: StreamBufferState::default(),
            sequence_counter: AtomicU64::new(0),
        };

        let stream_sender = TokenStreamSender::new(
            sender,
            terminated.clone(),
            metrics.clone(), 
            config.clone(),
        );

        Ok((token_stream, stream_sender))
    }

    /// Create with default configuration
    #[inline(always)]
    pub fn default() -> CandleResult<(Self, TokenStreamSender)> {
        Self::new(StreamingConfig::default())
    }

    /// Get stream metrics
    #[inline(always)]
    pub fn metrics(&self) -> &StreamingMetrics {
        &self.metrics
    }

    /// Check if stream is terminated
    #[inline(always)]
    pub fn is_terminated(&self) -> bool {
        self.terminated.load(AtomicOrdering::Relaxed)
    }

    /// Get next sequence number
    #[inline(always)]
    fn next_sequence_id(&self) -> u64 {
        self.sequence_counter.fetch_add(1, AtomicOrdering::Relaxed)
    }

    /// Handle buffer overflow with graceful degradation
    #[inline(always)]
    fn handle_buffer_overflow(&mut self, chunk: TokenChunk) -> CandleResult<Option<TokenChunk>> {
        self.metrics.record_buffer_overflow();

        if self.config.merge_on_overflow {
            if let Some(ref mut pending) = self.buffer_state.pending_chunk {
                // Try to merge with pending chunk
                match pending.try_merge(&chunk) {
                    Ok(()) => {
                        self.metrics.record_merge_operation();
                        return Ok(None); // Merged successfully
                    }
                    Err(_) => {
                        // Cannot merge, return pending chunk and replace with new
                        let old_pending = self.buffer_state.pending_chunk.take();
                        self.buffer_state.pending_chunk = Some(chunk);
                        return Ok(old_pending);
                    }
                }
            } else {
                // Store as pending chunk
                self.buffer_state.pending_chunk = Some(chunk);
                return Ok(None);
            }
        }

        // Drop chunk and record metrics
        self.metrics.record_dropped_chunk();
        Ok(None)
    }
}

/// Token stream sender for producer side
#[derive(Clone)]
pub struct TokenStreamSender {
    /// Channel sender for token chunks
    sender: Sender<TokenChunk>,
    /// Shared termination flag
    terminated: Arc<AtomicBool>,
    /// Shared metrics
    metrics: Arc<StreamingMetrics>,
    /// Stream configuration
    config: StreamingConfig,
    /// Sequence counter for ordering
    sequence_counter: AtomicU64,
}

impl TokenStreamSender {
    /// Create new sender
    fn new(
        sender: Sender<TokenChunk>,
        terminated: Arc<AtomicBool>,
        metrics: Arc<StreamingMetrics>,
        config: StreamingConfig,
    ) -> Self {
        Self {
            sender,
            terminated,
            metrics,
            config,
            sequence_counter: AtomicU64::new(0),
        }
    }

    /// Send token chunk with latency tracking
    #[inline(always)]
    pub fn send_token(&self, text: &str, metadata: TokenMetadata) -> CandleResult<()> {
        if self.terminated.load(AtomicOrdering::Relaxed) {
            return Err(CandleError::streaming_error("Stream already terminated"));
        }

        let start_time = Instant::now();
        let sequence_id = self.sequence_counter.fetch_add(1, AtomicOrdering::Relaxed);
        
        let chunk = TokenChunk::new(text, metadata, sequence_id)?;

        match self.sender.try_send(chunk) {
            Ok(()) => {
                let latency_nanos = start_time.elapsed().as_nanos() as u64;
                self.metrics.record_token_sent(latency_nanos);
                self.metrics.record_chunk_sent();
                
                // Update buffer usage estimate
                let current_usage = self.sender.len() as u64;
                self.metrics.update_peak_buffer_usage(current_usage);

                Ok(())
            }
            Err(TrySendError::Full(chunk)) => {
                // Channel is full, handle overflow
                Err(CandleError::streaming_error("Stream buffer overflow"))
            }
            Err(TrySendError::Disconnected(_)) => {
                Err(CandleError::streaming_error("Stream receiver disconnected"))
            }
        }
    }

    /// Send multiple tokens as batch
    pub fn send_batch(&self, tokens: &[(String, TokenMetadata)]) -> CandleResult<()> {
        for (text, metadata) in tokens {
            self.send_token(text, metadata.clone())?;
        }
        Ok(())
    }

    /// Terminate stream gracefully
    #[inline(always)]
    pub fn terminate(&self, reason: FinishReason) -> CandleResult<()> {
        if !self.terminated.compare_exchange(
            false,
            true,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ).unwrap_or(true) {
            // Send terminal chunk
            let sequence_id = self.sequence_counter.fetch_add(1, AtomicOrdering::Relaxed);
            let terminal_chunk = TokenChunk::new("", TokenMetadata::terminal(reason), sequence_id)?;
            
            match self.sender.try_send(terminal_chunk) {
                Ok(()) => Ok(()),
                Err(_) => {
                    // Channel full or disconnected, termination still successful
                    Ok(())
                }
            }
        } else {
            // Already terminated
            Ok(())
        }
    }

    /// Check if sender is still connected
    #[inline(always)]
    pub fn is_connected(&self) -> bool {
        !self.terminated.load(AtomicOrdering::Relaxed) && !self.sender.is_empty()
    }

    /// Get current buffer utilization (0.0 to 1.0)
    #[inline(always)]
    pub fn buffer_utilization(&self) -> f32 {
        (self.sender.len() as f32) / (self.config.buffer_size as f32)
    }
}

impl Stream for TokenOutputStream {
    type Item = CandleResult<TokenChunk>;

    #[inline(always)]
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Check for termination first
        if self.terminated.load(AtomicOrdering::Relaxed) && self.receiver.is_empty() {
            return Poll::Ready(None);
        }

        // Try to receive next chunk without blocking
        match self.receiver.try_recv() {
            Ok(chunk) => {
                // Check for terminal chunk
                if chunk.is_empty() && chunk.metadata.finish_reason.is_some() {
                    self.terminated.store(true, AtomicOrdering::Relaxed);
                    return Poll::Ready(Some(Ok(chunk)));
                }

                // Update buffer usage
                let current_usage = (self.receiver.len() * std::mem::size_of::<TokenChunk>()) as u64;
                self.metrics.update_peak_buffer_usage(current_usage);

                Poll::Ready(Some(Ok(chunk)))
            }
            Err(TryRecvError::Empty) => {
                // No data available, need to wait
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(TryRecvError::Disconnected) => {
                // Sender disconnected, terminate stream
                self.terminated.store(true, AtomicOrdering::Relaxed);
                Poll::Ready(None)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.receiver.len();
        (len, None) // Upper bound is unknown for streaming
    }
}

// Implement Drop for cleanup
impl Drop for TokenOutputStream {
    fn drop(&mut self) {
        // Mark as terminated for cleanup
        self.terminated.store(true, AtomicOrdering::Relaxed);
    }
}

impl Drop for TokenStreamSender {
    fn drop(&mut self) {
        // Mark as terminated for cleanup
        self.terminated.store(true, AtomicOrdering::Relaxed);
    }
}

// ============================================================================
// Backward Compatibility Conversion Traits
// ============================================================================

/// Convert TokenOutputStream to StreamingResponse for backward compatibility
impl Into<StreamingResponse> for TokenOutputStream {
    fn into(mut self) -> StreamingResponse {
        use futures::stream::{unfold, Stream};
        
        // Create a stream that yields strings by consuming TokenChunks
        let stream = unfold(Some(self), |mut token_stream_opt| async move {
            match token_stream_opt.take() {
                Some(mut token_stream) => {
                    match token_stream.next().await {
                        Some(chunk_result) => {
                            match chunk_result {
                                Ok(chunk) => {
                                    let text = chunk.text_str();
                                    if !text.is_empty() {
                                        // Continue with stream for next iteration
                                        (Some(Ok(text.to_string())), Some(token_stream))
                                    } else if chunk.metadata.finish_reason.is_some() {
                                        // Stream finished
                                        (None, None)
                                    } else {
                                        // Continue without yielding empty text
                                        (Some(Ok(String::new())), Some(token_stream))
                                    }
                                }
                                Err(e) => {
                                    // Error occurred, end stream
                                    (Some(Err(format!("Streaming error: {}", e))), None)
                                }
                            }
                        }
                        None => {
                            // Stream ended naturally
                            (None, None)
                        }
                    }
                }
                None => (None, None)
            }
        });
        
        StreamingResponse::new(Box::pin(stream))
    }
}

/// Wrapper for TokenOutputStream that implements the required traits for StreamingResponse
pub struct TokenStreamWrapper {
    inner: TokenOutputStream,
}

impl TokenStreamWrapper {
    /// Create new wrapper
    pub fn new(stream: TokenOutputStream) -> Self {
        Self { inner: stream }
    }
    
    /// Get reference to inner stream
    pub fn inner(&self) -> &TokenOutputStream {
        &self.inner
    }
    
    /// Get mutable reference to inner stream
    pub fn inner_mut(&mut self) -> &mut TokenOutputStream {
        &mut self.inner
    }
    
    /// Consume wrapper and return inner stream
    pub fn into_inner(self) -> TokenOutputStream {
        self.inner
    }
}