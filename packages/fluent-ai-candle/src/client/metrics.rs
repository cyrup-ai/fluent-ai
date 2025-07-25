//! Performance metrics and tracking for CandleCompletionClient
//! 
//! This module provides lock-free performance metrics aligned with provider patterns
//! for monitoring client performance and statistics.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

/// Lock-free performance metrics aligned with provider patterns
#[derive(Debug)]
pub struct CandleMetrics {
    pub total_requests: AtomicUsize,
    pub successful_requests: AtomicUsize,
    pub failed_requests: AtomicUsize,
    pub concurrent_requests: AtomicUsize,
    pub total_tokens_generated: AtomicUsize,
    pub streaming_requests: AtomicUsize,
    pub batch_requests: AtomicUsize,
    pub cache_hit_rate: AtomicUsize,
}

impl CandleMetrics {
    #[inline]
    pub fn new() -> Self {
        Self {
            total_requests: AtomicUsize::new(0),
            successful_requests: AtomicUsize::new(0),
            failed_requests: AtomicUsize::new(0),
            concurrent_requests: AtomicUsize::new(0),
            total_tokens_generated: AtomicUsize::new(0),
            streaming_requests: AtomicUsize::new(0),
            batch_requests: AtomicUsize::new(0),
            cache_hit_rate: AtomicUsize::new(0),
        }
    }

    /// Get metrics as tuple for easy access
    #[inline]
    pub fn get_metrics(&self) -> (usize, usize, usize, usize, usize, usize, usize, usize) {
        (
            self.total_requests.load(Ordering::Acquire),
            self.successful_requests.load(Ordering::Acquire),
            self.failed_requests.load(Ordering::Acquire),
            self.concurrent_requests.load(Ordering::Acquire),
            self.total_tokens_generated.load(Ordering::Acquire),
            self.streaming_requests.load(Ordering::Acquire),
            self.batch_requests.load(Ordering::Acquire),
            self.cache_hit_rate.load(Ordering::Acquire),
        )
    }

    /// Increment total requests counter
    #[inline]
    pub fn increment_total_requests(&self) {
        self.total_requests.fetch_add(1, Ordering::AcqRel);
    }

    /// Increment successful requests counter
    #[inline]
    pub fn increment_successful_requests(&self) {
        self.successful_requests.fetch_add(1, Ordering::AcqRel);
    }

    /// Increment failed requests counter
    #[inline]
    pub fn increment_failed_requests(&self) {
        self.failed_requests.fetch_add(1, Ordering::AcqRel);
    }

    /// Increment concurrent requests counter
    #[inline]
    pub fn increment_concurrent_requests(&self) {
        self.concurrent_requests.fetch_add(1, Ordering::AcqRel);
    }

    /// Decrement concurrent requests counter
    #[inline]
    pub fn decrement_concurrent_requests(&self) {
        self.concurrent_requests.fetch_sub(1, Ordering::AcqRel);
    }

    /// Add tokens to generated counter
    #[inline]
    pub fn add_tokens_generated(&self, tokens: usize) {
        self.total_tokens_generated.fetch_add(tokens, Ordering::AcqRel);
    }

    /// Increment streaming requests counter
    #[inline]
    pub fn increment_streaming_requests(&self) {
        self.streaming_requests.fetch_add(1, Ordering::AcqRel);
    }

    /// Increment batch requests counter
    #[inline]
    pub fn increment_batch_requests(&self) {
        self.batch_requests.fetch_add(1, Ordering::AcqRel);
    }

    /// Update cache hit rate
    #[inline]
    pub fn update_cache_hit_rate(&self, rate: usize) {
        self.cache_hit_rate.store(rate, Ordering::Release);
    }
}

impl Default for CandleMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Global lock-free performance metrics
pub static CANDLE_METRICS: LazyLock<CandleMetrics> = LazyLock::new(CandleMetrics::new);