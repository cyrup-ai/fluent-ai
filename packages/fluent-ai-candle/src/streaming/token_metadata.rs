//! Token metadata for streaming operations
//!
//! Provides TokenMetadata structure with zero-allocation design
//! for efficient token processing and statistics tracking.

use crate::types::FinishReason;

/// Token metadata for streaming operations with zero-allocation design
#[derive(Debug, Clone, Default)]
pub struct TokenMetadata {
    /// Position in generated sequence
    pub position: u32,
    /// Log probability of token selection
    pub logprob: f32,
    /// Stream finish reason if applicable
    pub finish_reason: Option<FinishReason>,
    /// Processing latency in nanoseconds
    pub processing_latency_nanos: u64,
    /// Temperature parameter used for generation
    pub temperature: f32,
    /// Top-p parameter if used
    pub top_p: Option<f32>,
    /// Top-k parameter if used
    pub top_k: Option<u32>}

impl TokenMetadata {
    /// Create new metadata with blazing-fast initialization
    #[inline(always)]
    pub fn new(position: u32, logprob: f32) -> Self {
        Self {
            position,
            logprob,
            finish_reason: None,
            processing_latency_nanos: 0,
            temperature: 1.0,
            top_p: None,
            top_k: None}
    }

    /// Create metadata with finish reason for stream termination
    #[inline(always)]
    pub fn with_finish_reason(position: u32, logprob: f32, reason: FinishReason) -> Self {
        Self {
            position,
            logprob,
            finish_reason: Some(reason),
            processing_latency_nanos: 0,
            temperature: 1.0,
            top_p: None,
            top_k: None}
    }

    /// Merge metadata from another instance for chunk aggregation
    #[inline(always)]
    pub fn merge(&mut self, other: &TokenMetadata) {
        // Keep the later position for sequence ordering
        if other.position > self.position {
            self.position = other.position;
        }

        // Use higher log probability (more confident prediction)
        if other.logprob > self.logprob {
            self.logprob = other.logprob;
        }

        // Use other's finish reason if present
        if other.finish_reason.is_some() {
            self.finish_reason = other.finish_reason.clone();
        }

        // Accumulate processing latency
        self.processing_latency_nanos = self
            .processing_latency_nanos
            .saturating_add(other.processing_latency_nanos);
    }

    /// Check if metadata indicates stream termination
    #[inline(always)]
    pub fn is_stream_end(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// Set temperature parameter with inline optimization
    #[inline(always)]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-p parameter with inline optimization
    #[inline(always)]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k parameter with inline optimization
    #[inline(always)]
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }
}
