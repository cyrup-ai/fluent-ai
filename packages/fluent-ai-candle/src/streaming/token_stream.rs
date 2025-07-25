//! Token stream implementation
//!
//! Provides token streaming functionality with zero-allocation patterns,
//! lock-free operations, and blazing-fast token transmission.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering as AtomicOrdering}};

use crossbeam_channel::{Receiver, bounded};

use super::{
    streaming_config::StreamingConfig, streaming_metrics::StreamingMetrics,
    token_chunk::TokenChunk, token_sender::TokenStreamSender};
use crate::error::CandleResult;

/// Create new token stream with configuration
pub fn create_token_stream(
    config: StreamingConfig,
) -> CandleResult<(TokenOutputStream, TokenStreamSender)> {
    config.validate()?;

    let (sender, receiver) = bounded(config.buffer_size);
    let terminated = Arc::new(AtomicBool::new(false));
    let metrics = Arc::new(StreamingMetrics::new());

    let output_stream = TokenOutputStream::new(
        receiver,
        terminated.clone(),
        metrics.clone(),
        config.clone(),
    );
    let stream_sender = TokenStreamSender::new(sender, terminated, metrics, config);

    Ok((output_stream, stream_sender))
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
    config: StreamingConfig}

impl TokenOutputStream {
    /// Create new output stream
    fn new(
        receiver: Receiver<TokenChunk>,
        terminated: Arc<AtomicBool>,
        metrics: Arc<StreamingMetrics>,
        config: StreamingConfig,
    ) -> Self {
        Self {
            receiver,
            terminated,
            metrics,
            config}
    }

    /// Get stream metrics for monitoring
    #[inline(always)]
    pub fn metrics(&self) -> &StreamingMetrics {
        &self.metrics
    }

    /// Check if stream is terminated
    #[inline(always)]
    pub fn is_terminated(&self) -> bool {
        self.terminated.load(AtomicOrdering::Relaxed)
    }

    /// Get configuration
    #[inline(always)]
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Try to receive the next token chunk (non-blocking, uses receiver field)
    pub fn try_recv(&self) -> Option<TokenChunk> {
        self.receiver.try_recv().ok()
    }

    /// Check if stream has pending tokens (uses receiver field)
    pub fn has_pending_tokens(&self) -> bool {
        !self.receiver.is_empty()
    }

    /// Get channel capacity info (uses receiver field)
    pub fn channel_info(&self) -> (usize, usize) {
        // Returns (pending messages, capacity)
        (self.receiver.len(), self.receiver.capacity().unwrap_or(0))
    }
}
