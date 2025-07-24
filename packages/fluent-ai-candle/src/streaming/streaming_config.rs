//! Streaming configuration for customizable behavior
//!
//! Provides StreamingConfig structure with validation and zero-allocation
//! patterns for configuring streaming performance and behavior.

use super::constants::{DEFAULT_BUFFER_SIZE, MAX_BUFFER_SIZE, MAX_CHUNK_TEXT_SIZE};
use crate::error::{CandleError, CandleResult};

/// Flush policy for batching behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushPolicy {
    /// Immediate flush on each token
    Immediate,
    /// Batch tokens until buffer is full
    Batched,
    /// Flush on timeout or buffer full
    Adaptive,
}

impl Default for FlushPolicy {
    #[inline(always)]
    fn default() -> Self {
        Self::Immediate
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
            return Err(CandleError::configuration(
                "Buffer size exceeds maximum allowed",
            ));
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

    /// Set flush policy with blazing-fast inline configuration
    #[inline(always)]
    pub fn flush_policy(mut self, policy: FlushPolicy) -> Self {
        self.flush_policy = policy;
        self
    }

    /// Enable or disable merge on overflow with zero-cost configuration
    #[inline(always)]
    pub fn merge_on_overflow(mut self, enable: bool) -> Self {
        self.merge_on_overflow = enable;
        self
    }

    /// Set maximum merge attempts with validation
    pub fn max_merge_attempts(mut self, attempts: u8) -> CandleResult<Self> {
        if attempts == 0 {
            return Err(CandleError::configuration(
                "Max merge attempts cannot be zero",
            ));
        }
        self.max_merge_attempts = attempts;
        Ok(self)
    }

    /// Validate configuration for consistency
    pub fn validate(&self) -> CandleResult<()> {
        if self.buffer_size == 0 {
            return Err(CandleError::configuration("Buffer size cannot be zero"));
        }
        if self.buffer_size > MAX_BUFFER_SIZE {
            return Err(CandleError::configuration("Buffer size exceeds maximum"));
        }
        if self.chunk_timeout_ms == 0 {
            return Err(CandleError::configuration("Chunk timeout cannot be zero"));
        }
        if self.max_chunk_size == 0 {
            return Err(CandleError::configuration("Max chunk size cannot be zero"));
        }
        if self.max_chunk_size > MAX_CHUNK_TEXT_SIZE {
            return Err(CandleError::configuration("Max chunk size exceeds limit"));
        }
        if self.max_merge_attempts == 0 {
            return Err(CandleError::configuration(
                "Max merge attempts cannot be zero",
            ));
        }
        Ok(())
    }

    /// Get estimated memory usage for this configuration
    #[inline(always)]
    pub fn estimated_memory_bytes(&self) -> usize {
        // Buffer size * chunk size + overhead
        self.buffer_size * self.max_chunk_size + 1024 // 1KB overhead
    }
}
