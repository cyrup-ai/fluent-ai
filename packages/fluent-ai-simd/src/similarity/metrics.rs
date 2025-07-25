//! Lock-free metrics tracking for similarity operations
//!
//! This module provides production-quality, zero-overhead metrics collection
//! using atomic counters for high-performance similarity computations.
//! All operations are lock-free and use relaxed memory ordering where possible
//! for optimal performance in concurrent environments.

use std::sync::atomic::{AtomicU64, Ordering};

/// Lock-free metrics collector for similarity operations
#[derive(Default, Debug)]
pub struct SimilarityMetrics {
    /// Total number of similarity calculations performed
    total_calculations: AtomicU64,
    /// Total number of vector elements processed
    total_elements_processed: AtomicU64,
}

impl SimilarityMetrics {
    /// Increment the calculation counter (relaxed ordering)
    #[inline(always)]
    pub fn increment_calculations(&self) {
        self.total_calculations.fetch_add(1, Ordering::Relaxed);
    }

    /// Add to the elements processed counter (relaxed ordering)
    #[inline(always)]
    pub fn add_elements(&self, count: u64) {
        self.total_elements_processed
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Get a consistent snapshot of current metrics (uses SeqCst for consistency)
    #[inline]
    pub fn get_metrics(&self) -> SimilarityMetricsSnapshot {
        SimilarityMetricsSnapshot {
            total_calculations: self.total_calculations.load(Ordering::SeqCst),
            total_elements_processed: self.total_elements_processed.load(Ordering::SeqCst),
        }
    }

    /// Reset all metrics to zero (SeqCst for consistency)
    #[inline]
    pub fn reset(&self) {
        self.total_calculations.store(0, Ordering::SeqCst);
        self.total_elements_processed.store(0, Ordering::SeqCst);
    }
}

/// Snapshot of metrics at a point in time
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimilarityMetricsSnapshot {
    /// Total similarity calculations
    pub total_calculations: u64,
    /// Total elements processed
    pub total_elements_processed: u64,
}

impl SimilarityMetricsSnapshot {
    /// Compute average vector length from snapshot
    #[inline]
    pub fn average_vector_length(&self) -> f64 {
        if self.total_calculations == 0 {
            0.0
        } else {
            self.total_elements_processed as f64 / self.total_calculations as f64
        }
    }
}

/// RAII guard for automatic metrics collection
///
/// Increments counters on creation and can be used for timing if extended.
pub struct MetricsGuard<'a> {
    metrics: &'a SimilarityMetrics,
}

impl<'a> MetricsGuard<'a> {
    /// Create a new guard and increment counters
    #[inline]
    pub fn new(metrics: &'a SimilarityMetrics, elements: usize) -> Self {
        metrics.increment_calculations();
        metrics.add_elements(elements as u64);
        Self { metrics }
    }
}

impl<'a> Drop for MetricsGuard<'a> {
    #[inline]
    fn drop(&mut self) {
        // Currently no-op; can be extended for timing metrics
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_metrics_collection() {
        let metrics = Arc::new(SimilarityMetrics::default());

        {
            let _guard = MetricsGuard::new(&metrics, 8);
        }

        let snapshot = metrics.get_metrics();
        assert_eq!(snapshot.total_calculations, 1);
        assert_eq!(snapshot.total_elements_processed, 8);
        assert_eq!(snapshot.average_vector_length(), 8.0);

        {
            let _guard = MetricsGuard::new(&metrics, 4);
        }

        let snapshot = metrics.get_metrics();
        assert_eq!(snapshot.total_calculations, 2);
        assert_eq!(snapshot.total_elements_processed, 12);
        assert_eq!(snapshot.average_vector_length(), 6.0);

        metrics.reset();
        let snapshot = metrics.get_metrics();
        assert_eq!(snapshot.total_calculations, 0);
        assert_eq!(snapshot.total_elements_processed, 0);
        assert_eq!(snapshot.average_vector_length(), 0.0);
    }

    #[test]
    fn test_concurrent_metrics() {
        use std::thread;

        let metrics = Arc::new(SimilarityMetrics::default());

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let metrics = metrics.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let _guard = MetricsGuard::new(&metrics, 16);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let snapshot = metrics.get_metrics();
        assert_eq!(snapshot.total_calculations, 1000);
        assert_eq!(snapshot.total_elements_processed, 16000);
        assert_eq!(snapshot.average_vector_length(), 16.0);
    }
}
