//! Token chunk with bounded text storage and metadata
//!
//! Provides TokenChunk structure with zero-allocation design,
//! bounded memory usage, and blazing-fast token processing.

use std::time::{SystemTime, UNIX_EPOCH};

use arrayvec::ArrayString;
use candle_core::Tensor;

use super::{constants::MAX_CHUNK_TEXT_SIZE, token_metadata::TokenMetadata};
use crate::error::{CandleError, CandleResult};

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
    pub sequence_id: u64}

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
            sequence_id})
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
                    logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                }
            }
            Err(_) => 0.0};

        let processing_latency_nanos = processing_start.elapsed().as_nanos() as u64;

        let metadata = TokenMetadata {
            position,
            logprob,
            finish_reason: None,
            processing_latency_nanos,
            temperature: 1.0,
            top_p: None,
            top_k: None};

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
            return Err(CandleError::streaming_error(
                "Cannot merge chunks: would exceed size limit",
            ));
        }

        self.text
            .try_push_str(&other.text)
            .map_err(|_| CandleError::streaming_error("Failed to merge token chunks"))?;

        // Update metadata to reflect merged state
        self.metadata.merge(&other.metadata);

        Ok(())
    }

    /// Check if chunk indicates stream should terminate
    #[inline(always)]
    pub fn is_termination_chunk(&self) -> bool {
        self.is_empty() && self.metadata.finish_reason.is_some()
    }

    /// Get sequence position with zero-cost access
    #[inline(always)]
    pub fn sequence_id(&self) -> u64 {
        self.sequence_id
    }

    /// Get timestamp with zero-cost access
    #[inline(always)]
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }
}
