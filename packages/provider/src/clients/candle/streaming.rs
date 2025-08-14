//! Zero-allocation streaming coordinator for real-time text generation
//!
//! This module provides high-performance streaming with backpressure handling,
//! lock-free coordination, and comprehensive flow control for live inference.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use crossbeam::atomic::AtomicCell;
use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded, unbounded};
use smallvec::SmallVec;

use super::error::{CandleError, CandleResult};

/// Maximum number of chunks in flight for backpressure control
const MAX_CHUNKS_IN_FLIGHT: usize = 128;
/// Default buffer size for streaming chunks
const DEFAULT_CHUNK_BUFFER_SIZE: usize = 1024;
/// Maximum token text length for efficient storage
const MAX_TOKEN_TEXT_LEN: usize = 256;

/// Finish reason for generation completion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Generation completed successfully
    Completed,
    /// Hit maximum token limit
    MaxLength,
    /// Encountered stop token
    StopToken,
    /// User requested stop
    UserStopped,
    /// Error occurred during generation
    Error,
    /// Timeout reached
    Timeout,
}

impl FinishReason {
    /// Check if this represents a successful completion
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            FinishReason::Completed | FinishReason::StopToken | FinishReason::MaxLength
        )
    }

    /// Check if this represents an error condition
    pub fn is_error(&self) -> bool {
        matches!(self, FinishReason::Error | FinishReason::Timeout)
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Completed => "completed",
            FinishReason::MaxLength => "max_length",
            FinishReason::StopToken => "stop_token",
            FinishReason::UserStopped => "user_stopped",
            FinishReason::Error => "error",
            FinishReason::Timeout => "timeout",
        }
    }
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Zero-allocation streaming chunk for token delivery
#[derive(Debug, Clone)]
pub struct StreamingChunk {
    /// Token ID that was generated
    pub token_id: u32,
    /// Decoded text bytes (UTF-8)
    pub text_bytes: SmallVec<u8, MAX_TOKEN_TEXT_LEN>,
    /// Token position in sequence
    pub token_position: u32,
    /// Timestamp when chunk was created
    pub timestamp: Instant,
    /// Chunk sequence number for ordering
    pub sequence_number: u64,
}

impl StreamingChunk {
    /// Create a new streaming chunk
    pub fn new(token_id: u32, text_bytes: &[u8], token_position: u32) -> Self {
        let mut text_vec = SmallVec::new();
        text_vec.extend_from_slice(text_bytes);

        Self {
            token_id,
            text_bytes: text_vec,
            token_position,
            timestamp: Instant::now(),
            sequence_number: 0, // Will be set by streaming coordinator
        }
    }

    /// Get text as string slice
    pub fn text(&self) -> CandleResult<&str> {
        std::str::from_utf8(&self.text_bytes).map_err(|e| {
            CandleError::streaming(
                &format!("Invalid UTF-8 in chunk: {}", e),
                "text",
                "valid UTF-8 bytes",
            )
        })
    }

    /// Get text bytes
    pub fn text_bytes(&self) -> &[u8] {
        &self.text_bytes
    }

    /// Get chunk size in bytes
    pub fn size_bytes(&self) -> usize {
        self.text_bytes.len()
    }

    /// Check if chunk is empty
    pub fn is_empty(&self) -> bool {
        self.text_bytes.is_empty()
    }

    /// Get age of this chunk
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

/// Configuration for streaming behavior
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum chunks to buffer before backpressure
    pub max_buffer_size: usize,
    /// Enable backpressure handling
    pub enable_backpressure: bool,
    /// Timeout for chunk delivery
    pub chunk_timeout: Duration,
    /// Enable chunk ordering guarantees
    pub preserve_order: bool,
    /// Maximum latency before forced flush
    pub max_latency: Duration,
    /// Minimum batch size for efficiency
    pub min_batch_size: usize,
    /// Maximum batch size for responsiveness
    pub max_batch_size: usize,
    /// Enable streaming statistics collection
    pub enable_statistics: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: MAX_CHUNKS_IN_FLIGHT,
            enable_backpressure: true,
            chunk_timeout: Duration::from_millis(100),
            preserve_order: true,
            max_latency: Duration::from_millis(50),
            min_batch_size: 1,
            max_batch_size: 16,
            enable_statistics: true,
        }
    }
}

impl StreamingConfig {
    /// Create configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            max_buffer_size: 32,
            enable_backpressure: true,
            chunk_timeout: Duration::from_millis(10),
            preserve_order: true,
            max_latency: Duration::from_millis(5),
            min_batch_size: 1,
            max_batch_size: 4,
            enable_statistics: true,
        }
    }

    /// Create configuration optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            max_buffer_size: 512,
            enable_backpressure: false,
            chunk_timeout: Duration::from_millis(500),
            preserve_order: false,
            max_latency: Duration::from_millis(200),
            min_batch_size: 8,
            max_batch_size: 64,
            enable_statistics: true,
        }
    }

    /// Create configuration for balanced performance
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> CandleResult<()> {
        if self.max_buffer_size == 0 {
            return Err(CandleError::config(
                "Buffer size must be positive",
                "max_buffer_size",
                "> 0",
            ));
        }

        if self.max_buffer_size > 10000 {
            return Err(CandleError::config(
                "Buffer size too large",
                "max_buffer_size",
                "<= 10000",
            ));
        }

        if self.min_batch_size == 0 {
            return Err(CandleError::config(
                "Minimum batch size must be positive",
                "min_batch_size",
                "> 0",
            ));
        }

        if self.max_batch_size < self.min_batch_size {
            return Err(CandleError::config(
                "Maximum batch size must be >= minimum batch size",
                "max_batch_size",
                ">= min_batch_size",
            ));
        }

        if self.chunk_timeout.is_zero() {
            return Err(CandleError::config(
                "Chunk timeout must be positive",
                "chunk_timeout",
                "> 0",
            ));
        }

        if self.max_latency.is_zero() {
            return Err(CandleError::config(
                "Maximum latency must be positive",
                "max_latency",
                "> 0",
            ));
        }

        Ok(())
    }
}

/// Streaming session state for coordination
#[derive(Debug, Clone)]
struct StreamingSession {
    /// Session ID for tracking
    session_id: u64,
    /// Start timestamp
    start_time: Instant,
    /// Current sequence number
    sequence_number: u64,
    /// Number of chunks sent
    chunks_sent: u64,
    /// Total bytes sent
    bytes_sent: u64,
    /// Whether session is active
    is_active: bool,
    /// Session finish reason (if completed)
    finish_reason: Option<FinishReason>,
}

impl Default for StreamingSession {
    fn default() -> Self {
        Self {
            session_id: 0,
            start_time: Instant::now(),
            sequence_number: 0,
            chunks_sent: 0,
            bytes_sent: 0,
            is_active: false,
            finish_reason: None,
        }
    }
}

impl StreamingSession {
    /// Create new streaming session
    fn new(session_id: u64) -> Self {
        Self {
            session_id,
            start_time: Instant::now(),
            sequence_number: 0,
            chunks_sent: 0,
            bytes_sent: 0,
            is_active: true,
            finish_reason: None,
        }
    }

    /// Get session duration
    fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Calculate throughput in chunks per second
    fn chunks_per_second(&self) -> f32 {
        let duration_secs = self.duration().as_secs_f32();
        if duration_secs > 0.0 {
            self.chunks_sent as f32 / duration_secs
        } else {
            0.0
        }
    }

    /// Calculate bandwidth in bytes per second
    fn bytes_per_second(&self) -> f32 {
        let duration_secs = self.duration().as_secs_f32();
        if duration_secs > 0.0 {
            self.bytes_sent as f32 / duration_secs
        } else {
            0.0
        }
    }
}

/// High-performance streaming coordinator with backpressure handling
#[derive(Debug)]
pub struct StreamingCoordinator {
    /// Streaming configuration
    config: StreamingConfig,
    /// Current streaming session
    session: ArcSwap<StreamingSession>,
    /// Chunk sender for producer
    chunk_sender: Sender<StreamingChunk>,
    /// Chunk receiver for consumer
    chunk_receiver: Receiver<StreamingChunk>,
    /// Streaming statistics
    stats: StreamingStatistics,
    /// Flow control state
    flow_control: FlowControl,
}

/// Flow control state for backpressure management
#[derive(Debug)]
struct FlowControl {
    /// Current buffer level
    buffer_level: AtomicCell<usize>,
    /// Backpressure active flag
    backpressure_active: AtomicCell<bool>,
    /// Total chunks dropped due to backpressure
    chunks_dropped: AtomicCell<u64>,
    /// Last backpressure event time
    last_backpressure: AtomicCell<Option<Instant>>,
}

impl Default for FlowControl {
    fn default() -> Self {
        Self {
            buffer_level: AtomicCell::new(0),
            backpressure_active: AtomicCell::new(false),
            chunks_dropped: AtomicCell::new(0),
            last_backpressure: AtomicCell::new(None),
        }
    }
}

impl StreamingCoordinator {
    /// Create a new streaming coordinator
    pub fn new(config: StreamingConfig) -> Self {
        let (sender, receiver) = if config.enable_backpressure {
            bounded(config.max_buffer_size)
        } else {
            let (s, r) = unbounded();
            (s, r)
        };

        Self {
            config,
            session: ArcSwap::from_pointee(StreamingSession::default()),
            chunk_sender: sender,
            chunk_receiver: receiver,
            stats: StreamingStatistics::default(),
            flow_control: FlowControl::default(),
        }
    }

    /// Start a new streaming session
    pub fn start_streaming(&self) -> CandleResult<u64> {
        let session_id = self.stats.total_sessions.load() + 1;
        let new_session = StreamingSession::new(session_id);

        self.session.store(Arc::new(new_session));
        self.stats.total_sessions.store(session_id);
        self.stats.active_sessions.store(1);

        // Reset flow control
        self.flow_control.buffer_level.store(0);
        self.flow_control.backpressure_active.store(false);

        Ok(session_id)
    }

    /// Send a streaming chunk
    pub fn send_chunk(&self, mut chunk: StreamingChunk) -> CandleResult<()> {
        let session = self.session.load();

        if !session.is_active {
            return Err(CandleError::streaming(
                "No active streaming session",
                "send_chunk",
                "active session",
            ));
        }

        // Set sequence number
        chunk.sequence_number = session.sequence_number + 1;

        // Check for backpressure
        if self.config.enable_backpressure {
            match self.chunk_sender.try_send(chunk.clone()) {
                Ok(_) => {
                    // Success - update flow control
                    self.flow_control
                        .buffer_level
                        .store(self.chunk_sender.len());
                    self.flow_control.backpressure_active.store(false);
                }
                Err(TrySendError::Full(_)) => {
                    // Backpressure - handle according to configuration
                    self.flow_control.backpressure_active.store(true);
                    self.flow_control
                        .last_backpressure
                        .store(Some(Instant::now()));
                    self.flow_control
                        .chunks_dropped
                        .store(self.flow_control.chunks_dropped.load() + 1);

                    return Err(CandleError::streaming(
                        "Streaming buffer full (backpressure)",
                        "send_chunk",
                        "consumer to process chunks",
                    ));
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(CandleError::streaming(
                        "Streaming channel disconnected",
                        "send_chunk",
                        "connected receiver",
                    ));
                }
            }
        } else {
            // No backpressure - send with blocking
            self.chunk_sender.send(chunk.clone()).map_err(|_| {
                CandleError::streaming("Failed to send chunk", "send_chunk", "connected receiver")
            })?;
        }

        // Update session state
        let mut new_session = (**session).clone();
        new_session.sequence_number += 1;
        new_session.chunks_sent += 1;
        new_session.bytes_sent += chunk.size_bytes() as u64;
        self.session.store(Arc::new(new_session));

        // Update statistics
        self.stats
            .total_chunks_sent
            .store(self.stats.total_chunks_sent.load() + 1);
        self.stats
            .total_bytes_sent
            .store(self.stats.total_bytes_sent.load() + chunk.size_bytes() as u64);

        Ok(())
    }

    /// End the current streaming session
    pub fn end_streaming(&self, finish_reason: FinishReason) -> CandleResult<()> {
        let current_session = self.session.load();

        if !current_session.is_active {
            return Err(CandleError::streaming(
                "No active streaming session to end",
                "end_streaming",
                "active session",
            ));
        }

        // Update session with finish reason
        let mut final_session = (**current_session).clone();
        final_session.is_active = false;
        final_session.finish_reason = Some(finish_reason);
        self.session.store(Arc::new(final_session));

        // Update statistics
        self.stats.active_sessions.store(0);

        if finish_reason.is_success() {
            self.stats
                .successful_sessions
                .store(self.stats.successful_sessions.load() + 1);
        } else {
            self.stats
                .failed_sessions
                .store(self.stats.failed_sessions.load() + 1);
        }

        Ok(())
    }

    /// Get the next chunk (consumer side)
    pub fn receive_chunk(&self) -> Result<StreamingChunk, TryRecvError> {
        let result = self.chunk_receiver.try_recv();

        // Update flow control buffer level
        self.flow_control
            .buffer_level
            .store(self.chunk_receiver.len());

        if result.is_ok() {
            self.stats
                .total_chunks_received
                .store(self.stats.total_chunks_received.load() + 1);
        }

        result
    }

    /// Check if streaming session is active
    pub fn is_streaming(&self) -> bool {
        self.session.load().is_active
    }

    /// Get current session information
    pub fn current_session(&self) -> Option<StreamingSessionInfo> {
        let session = self.session.load();

        if session.session_id > 0 {
            Some(StreamingSessionInfo {
                session_id: session.session_id,
                is_active: session.is_active,
                duration: session.duration(),
                chunks_sent: session.chunks_sent,
                bytes_sent: session.bytes_sent,
                chunks_per_second: session.chunks_per_second(),
                bytes_per_second: session.bytes_per_second(),
                finish_reason: session.finish_reason,
            })
        } else {
            None
        }
    }

    /// Get current buffer level
    pub fn buffer_level(&self) -> usize {
        self.flow_control.buffer_level.load()
    }

    /// Check if backpressure is active
    pub fn is_backpressure_active(&self) -> bool {
        self.flow_control.backpressure_active.load()
    }

    /// Get streaming configuration
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Get comprehensive streaming statistics
    pub fn statistics(&self) -> StreamingStatistics {
        StreamingStatistics {
            total_sessions: self.stats.total_sessions.load(),
            active_sessions: self.stats.active_sessions.load(),
            successful_sessions: self.stats.successful_sessions.load(),
            failed_sessions: self.stats.failed_sessions.load(),
            total_chunks_sent: self.stats.total_chunks_sent.load(),
            total_chunks_received: self.stats.total_chunks_received.load(),
            total_bytes_sent: self.stats.total_bytes_sent.load(),
            chunks_dropped: self.flow_control.chunks_dropped.load(),
            backpressure_events: if self.flow_control.last_backpressure.load().is_some() {
                1
            } else {
                0
            },
            current_buffer_level: self.buffer_level(),
            is_backpressure_active: self.is_backpressure_active(),
        }
    }
}

/// Session information for monitoring
#[derive(Debug, Clone)]
pub struct StreamingSessionInfo {
    /// Session ID
    pub session_id: u64,
    /// Whether session is active
    pub is_active: bool,
    /// Session duration
    pub duration: Duration,
    /// Chunks sent in this session
    pub chunks_sent: u64,
    /// Bytes sent in this session
    pub bytes_sent: u64,
    /// Throughput in chunks per second
    pub chunks_per_second: f32,
    /// Bandwidth in bytes per second
    pub bytes_per_second: f32,
    /// Finish reason (if completed)
    pub finish_reason: Option<FinishReason>,
}

/// Comprehensive streaming statistics
#[derive(Debug, Clone)]
pub struct StreamingStatistics {
    /// Total sessions started
    total_sessions: AtomicCell<u64>,
    /// Currently active sessions
    active_sessions: AtomicCell<u64>,
    /// Successfully completed sessions
    successful_sessions: AtomicCell<u64>,
    /// Failed sessions
    failed_sessions: AtomicCell<u64>,
    /// Total chunks sent
    total_chunks_sent: AtomicCell<u64>,
    /// Total chunks received
    total_chunks_received: AtomicCell<u64>,
    /// Total bytes sent
    total_bytes_sent: AtomicCell<u64>,
    /// Chunks dropped due to backpressure
    chunks_dropped: AtomicCell<u64>,
    /// Number of backpressure events
    backpressure_events: AtomicCell<u64>,
    /// Current buffer level
    current_buffer_level: AtomicCell<usize>,
    /// Whether backpressure is currently active
    is_backpressure_active: AtomicCell<bool>,
}

impl Default for StreamingStatistics {
    fn default() -> Self {
        Self {
            total_sessions: AtomicCell::new(0),
            active_sessions: AtomicCell::new(0),
            successful_sessions: AtomicCell::new(0),
            failed_sessions: AtomicCell::new(0),
            total_chunks_sent: AtomicCell::new(0),
            total_chunks_received: AtomicCell::new(0),
            total_bytes_sent: AtomicCell::new(0),
            chunks_dropped: AtomicCell::new(0),
            backpressure_events: AtomicCell::new(0),
            current_buffer_level: AtomicCell::new(0),
            is_backpressure_active: AtomicCell::new(false),
        }
    }
}

impl StreamingStatistics {
    /// Calculate success rate
    pub fn success_rate(&self) -> f32 {
        let total = self.total_sessions.load();
        if total > 0 {
            self.successful_sessions.load() as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Calculate chunk delivery rate
    pub fn delivery_rate(&self) -> f32 {
        let sent = self.total_chunks_sent.load();
        if sent > 0 {
            self.total_chunks_received.load() as f32 / sent as f32
        } else {
            0.0
        }
    }

    /// Calculate drop rate due to backpressure
    pub fn drop_rate(&self) -> f32 {
        let total_attempted = self.total_chunks_sent.load() + self.chunks_dropped.load();
        if total_attempted > 0 {
            self.chunks_dropped.load() as f32 / total_attempted as f32
        } else {
            0.0
        }
    }

    /// Get average bytes per chunk
    pub fn avg_bytes_per_chunk(&self) -> f32 {
        let chunks = self.total_chunks_sent.load();
        if chunks > 0 {
            self.total_bytes_sent.load() as f32 / chunks as f32
        } else {
            0.0
        }
    }

    /// Check if streaming is performing well
    pub fn is_healthy(&self) -> bool {
        let success_rate = self.success_rate();
        let delivery_rate = self.delivery_rate();
        let drop_rate = self.drop_rate();

        success_rate >= 0.95 && delivery_rate >= 0.98 && drop_rate <= 0.02
    }
}

/// Token streamer interface for consumer implementations
pub trait TokenStreamer {
    /// Handle a streaming chunk
    fn handle_chunk(&mut self, chunk: StreamingChunk) -> CandleResult<()>;

    /// Handle session completion
    fn handle_completion(&mut self, session_info: StreamingSessionInfo) -> CandleResult<()>;

    /// Handle streaming error
    fn handle_error(&mut self, error: CandleError) -> CandleResult<()>;
}

/// Simple text accumulator implementation of TokenStreamer
#[derive(Debug, Default)]
pub struct TextAccumulator {
    /// Accumulated text
    pub text: String,
    /// Chunk count
    pub chunk_count: usize,
    /// Total bytes received
    pub bytes_received: usize,
}

impl TokenStreamer for TextAccumulator {
    fn handle_chunk(&mut self, chunk: StreamingChunk) -> CandleResult<()> {
        let text = chunk.text()?;
        self.text.push_str(text);
        self.chunk_count += 1;
        self.bytes_received += chunk.size_bytes();
        Ok(())
    }

    fn handle_completion(&mut self, _session_info: StreamingSessionInfo) -> CandleResult<()> {
        // Default implementation does nothing
        Ok(())
    }

    fn handle_error(&mut self, _error: CandleError) -> CandleResult<()> {
        // Default implementation does nothing
        Ok(())
    }
}

impl TextAccumulator {
    /// Create new text accumulator
    pub fn new() -> Self {
        Self::default()
    }

    /// Get accumulated text
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get chunk count
    pub fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    /// Get bytes received
    pub fn bytes_received(&self) -> usize {
        self.bytes_received
    }

    /// Clear accumulated data
    pub fn clear(&mut self) {
        self.text.clear();
        self.chunk_count = 0;
        self.bytes_received = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_chunk_creation() {
        let chunk = StreamingChunk::new(123, b"hello", 5);
        assert_eq!(chunk.token_id, 123);
        assert_eq!(chunk.text().unwrap(), "hello");
        assert_eq!(chunk.token_position, 5);
        assert_eq!(chunk.size_bytes(), 5);
    }

    #[test]
    fn test_streaming_config_validation() {
        let config = StreamingConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.max_buffer_size = 0;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_finish_reason_properties() {
        assert!(FinishReason::Completed.is_success());
        assert!(!FinishReason::Completed.is_error());

        assert!(!FinishReason::Error.is_success());
        assert!(FinishReason::Error.is_error());

        assert_eq!(FinishReason::StopToken.as_str(), "stop_token");
    }

    #[test]
    fn test_streaming_coordinator_creation() {
        let config = StreamingConfig::balanced();
        let coordinator = StreamingCoordinator::new(config);

        assert!(!coordinator.is_streaming());
        assert_eq!(coordinator.buffer_level(), 0);
        assert!(!coordinator.is_backpressure_active());
    }

    #[test]
    fn test_streaming_session_lifecycle() {
        let config = StreamingConfig::balanced();
        let coordinator = StreamingCoordinator::new(config);

        // Start session
        let session_id = coordinator.start_streaming().unwrap();
        assert!(session_id > 0);
        assert!(coordinator.is_streaming());

        // End session
        coordinator.end_streaming(FinishReason::Completed).unwrap();
        assert!(!coordinator.is_streaming());

        let stats = coordinator.statistics();
        assert_eq!(stats.successful_sessions.load(), 1);
    }

    #[test]
    fn test_text_accumulator() {
        let mut accumulator = TextAccumulator::new();

        let chunk1 = StreamingChunk::new(1, b"Hello", 0);
        let chunk2 = StreamingChunk::new(2, b" world", 1);

        accumulator.handle_chunk(chunk1).unwrap();
        accumulator.handle_chunk(chunk2).unwrap();

        assert_eq!(accumulator.text(), "Hello world");
        assert_eq!(accumulator.chunk_count(), 2);
        assert_eq!(accumulator.bytes_received(), 11);
    }
}
