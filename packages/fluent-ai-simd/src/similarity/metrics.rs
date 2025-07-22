//! Metrics collection and tracking for similarity operations

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Tracks metrics for similarity operations
#[derive(Debug, Default)]
pub(crate) struct SimilarityMetrics {
    /// Total similarity calculations performed
    total_calculations: AtomicU64,
    /// Total vector elements processed
    total_elements_processed: AtomicU64,
    /// Total time spent in SIMD operations (nanoseconds)
    simd_time_ns: AtomicU64,
}

impl SimilarityMetrics {
    /// Record a similarity calculation with timing information
    #[inline]
    pub(crate) fn record_calculation(&self, elements: usize, duration: std::time::Duration) {
        self.total_calculations.fetch_add(1, Ordering::Relaxed);
        self.total_elements_processed
            .fetch_add(elements as u64, Ordering::Relaxed);
        self.simd_time_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Get a snapshot of current metrics
    pub(crate) fn get_metrics(&self) -> SimilarityMetricsSnapshot {
        SimilarityMetricsSnapshot {
            total_calculations: self.total_calculations.load(Ordering::Relaxed),
            total_elements_processed: self.total_elements_processed.load(Ordering::Relaxed),
            simd_time_ns: self.simd_time_ns.load(Ordering::Relaxed),
        }
    }

    /// Reset all metrics to zero
    pub(crate) fn reset(&self) {
        self.total_calculations.store(0, Ordering::Relaxed);
        self.total_elements_processed.store(0, Ordering::Relaxed);
        self.simd_time_ns.store(0, Ordering::Relaxed);
    }
}

/// A snapshot of similarity metrics at a point in time
#[derive(Debug, Clone, Copy)]
pub struct SimilarityMetricsSnapshot {
    /// Total similarity calculations performed
    pub total_calculations: u64,
    /// Total vector elements processed
    pub total_elements_processed: u64,
    /// Total time spent in SIMD operations (nanoseconds)
    pub simd_time_ns: u64,
}

impl SimilarityMetricsSnapshot {
    /// Calculate the average time per similarity calculation in nanoseconds
    pub fn avg_time_ns(&self) -> f64 {
        if self.total_calculations > 0 {
            self.simd_time_ns as f64 / self.total_calculations as f64
        } else {
            0.0
        }
    }

    /// Calculate the average elements processed per second
    pub fn elements_per_second(&self) -> f64 {
        if self.simd_time_ns > 0 {
            (self.total_elements_processed as f64 * 1_000_000_000.0) / self.simd_time_ns as f64
        } else {
            0.0
        }
    }
}

/// A guard that records the duration of a similarity calculation
pub(crate) struct MetricsGuard<'a> {
    metrics: &'a SimilarityMetrics,
    start: Instant,
    elements: usize,
}

impl<'a> MetricsGuard<'a> {
    /// Create a new metrics guard that will record the duration when dropped
    pub(crate) fn new(metrics: &'a SimilarityMetrics, elements: usize) -> Self {
        Self {
            metrics,
            start: Instant::now(),
            elements,
        }
    }
}

impl Drop for MetricsGuard<'_> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.metrics.record_calculation(self.elements, duration);
    }
}

// Global metrics instance
use std::sync::Arc;

use lazy_static::lazy_static;

lazy_static! {
    static ref GLOBAL_METRICS: Arc<SimilarityMetrics> = Arc::new(SimilarityMetrics::default());
}

/// Get global similarity metrics
pub fn get_similarity_metrics() -> SimilarityMetricsSnapshot {
    GLOBAL_METRICS.get_metrics()
}

/// Reset global similarity metrics
pub fn reset_similarity_metrics() {
    GLOBAL_METRICS.reset();
}
