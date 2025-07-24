//! Token stream implementation
//!
//! Provides token streaming functionality with zero-allocation patterns,
//! lock-free operations, and blazing-fast token transmission.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering as AtomicOrdering},
        Arc,
    },
};

use crossbeam_channel::{bounded, Receiver};
use crate::error::CandleResult;
use super::{
    streaming_config::StreamingConfig,
    streaming_metrics::StreamingMetrics,
    token_chunk::TokenChunk,
    token_sender::TokenStreamSender,
};

/// Create new token stream with configuration
pub fn create_token_stream(config: StreamingConfig) -> CandleResult<(TokenOutputStream, TokenStreamSender)> {
    config.validate()?;
    
    let (sender, receiver) = bounded(config.buffer_size);
    let terminated = Arc::new(AtomicBool::new(false));
    let metrics = Arc::new(StreamingMetrics::new());

    let output_stream = TokenOutputStream::new(receiver, terminated.clone(), metrics.clone(), config.clone());
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
    config: StreamingConfig,
}

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
            config,
        }
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
}