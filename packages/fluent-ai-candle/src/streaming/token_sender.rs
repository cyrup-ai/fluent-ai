//! Token stream sender for producer side
//!
//! Provides TokenStreamSender for producing tokens with zero-allocation
//! patterns and blazing-fast transmission with flow control.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering}},
    time::Instant};

use crossbeam_channel::{Sender, TrySendError};

use super::{
    streaming_config::StreamingConfig, streaming_metrics::StreamingMetrics,
    token_chunk::TokenChunk, token_metadata::TokenMetadata};
use crate::{
    error::{CandleError, CandleResult},
    types::FinishReason};

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
    sequence_counter: Arc<AtomicU64>}

impl TokenStreamSender {
    /// Create new sender
    pub(super) fn new(
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
            sequence_counter: Arc::new(AtomicU64::new(0))}
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
                // Channel is full, handle overflow with chunk information
                self.metrics.record_buffer_overflow();
                let chunk_size = std::mem::size_of_val(&chunk);
                if chunk_size > 1024 {
                    Err(CandleError::streaming_error(
                        "Stream buffer overflow - chunk too large",
                    ))
                } else {
                    Err(CandleError::streaming_error(
                        "Stream buffer overflow - failed to send chunk",
                    ))
                }
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
        if !self
            .terminated
            .compare_exchange(
                false,
                true,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            )
            .is_ok()
        {
            return Ok(()); // Already terminated
        }

        // Send termination chunk
        let sequence_id = self.sequence_counter.fetch_add(1, AtomicOrdering::Relaxed);
        let mut metadata = TokenMetadata::default();
        metadata.finish_reason = Some(reason);

        let termination_chunk = TokenChunk::empty(sequence_id)?;

        match self.sender.try_send(termination_chunk) {
            Ok(()) => {
                self.metrics.record_chunk_sent();
                Ok(())
            }
            Err(_) => {
                // If we can't send termination chunk, still mark as terminated
                Ok(())
            }
        }
    }

    /// Check if sender is terminated
    #[inline(always)]
    pub fn is_terminated(&self) -> bool {
        self.terminated.load(AtomicOrdering::Relaxed)
    }

    /// Get current buffer length for flow control
    #[inline(always)]
    pub fn buffer_len(&self) -> usize {
        self.sender.len()
    }

    /// Get stream metrics
    #[inline(always)]
    pub fn metrics(&self) -> &StreamingMetrics {
        &self.metrics
    }
    /// Get streaming configuration (uses the config field)
    pub fn get_config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Check if buffering is enabled according to config
    pub fn is_buffering_enabled(&self) -> bool {
        self.config.buffer_size > 0
    }
}

impl Drop for TokenStreamSender {
    fn drop(&mut self) {
        // Mark as terminated for cleanup
        let _ = self.terminated.store(true, AtomicOrdering::Relaxed);
    }
}
