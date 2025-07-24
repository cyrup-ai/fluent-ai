//! Streaming metrics and performance tracking
//!
//! Provides StreamingMetrics structure with atomic operations for
//! zero-allocation performance monitoring and statistics collection.

use std::{
    sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
    time::{SystemTime, UNIX_EPOCH},
};

/// Streaming performance metrics with atomic operations
#[derive(Debug)]
pub struct StreamingMetrics {
    /// Total tokens sent through stream
    pub tokens_sent: AtomicU64,
    /// Total chunks sent through stream
    pub chunks_sent: AtomicU64,
    /// Buffer overflow events
    pub buffer_overflows: AtomicU64,
    /// Merge operations performed
    pub merge_operations: AtomicU64,
    /// Chunks dropped due to errors
    pub dropped_chunks: AtomicU64,
    /// Total latency accumulated (nanoseconds)
    pub total_latency_nanos: AtomicU64,
    /// Peak buffer usage observed
    pub peak_buffer_usage: AtomicU64,
    /// Stream start timestamp
    pub stream_start_nanos: AtomicU64,
}

impl Default for StreamingMetrics {
    #[inline(always)]
    fn default() -> Self {
        let now_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        Self {
            tokens_sent: AtomicU64::new(0),
            chunks_sent: AtomicU64::new(0),
            buffer_overflows: AtomicU64::new(0),
            merge_operations: AtomicU64::new(0),
            dropped_chunks: AtomicU64::new(0),
            total_latency_nanos: AtomicU64::new(0),
            peak_buffer_usage: AtomicU64::new(0),
            stream_start_nanos: AtomicU64::new(now_nanos),
        }
    }
}

impl StreamingMetrics {
    /// Create new metrics instance with current timestamp
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record token sent with latency tracking
    #[inline(always)]
    pub fn record_token_sent(&self, latency_nanos: u64) {
        self.tokens_sent.fetch_add(1, AtomicOrdering::Relaxed);
        self.total_latency_nanos
            .fetch_add(latency_nanos, AtomicOrdering::Relaxed);
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

    /// Update peak buffer usage with compare-and-swap optimization
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

    /// Get tokens per second rate with zero-allocation calculation
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

    /// Get average latency per token with blazing-fast calculation
    pub fn average_latency_nanos(&self) -> f64 {
        let total_latency = self.total_latency_nanos.load(AtomicOrdering::Relaxed) as f64;
        let tokens = self.tokens_sent.load(AtomicOrdering::Relaxed) as f64;

        if tokens > 0.0 {
            total_latency / tokens
        } else {
            0.0
        }
    }

    /// Get buffer overflow rate with zero-allocation analysis
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
