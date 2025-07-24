//! Model statistics and metrics methods
//!
//! Provides methods for tracking and retrieving model performance statistics
//! with zero-allocation patterns and blazing-fast atomic operations.

use std::sync::atomic::Ordering;

use super::CandleModel;
use crate::model::metrics::ModelMetrics;

impl CandleModel {
    /// Get comprehensive performance metrics with zero-allocation access
    #[inline(always)]
    pub fn get_metrics(&self) -> ModelMetrics {
        let cache_stats = Some(self.cache_manager.get_stats());
        let model_memory = self.memory_usage.load(Ordering::Relaxed);

        let mut metrics =
            ModelMetrics::with_cache_stats(cache_stats.as_ref().unwrap().clone(), model_memory);

        // Update with current generation stats using blazing-fast atomic access
        metrics.performance.total_tokens_generated =
            self.total_tokens_generated.load(Ordering::Relaxed);
        metrics.performance.avg_tokens_per_second =
            self.avg_tokens_per_second.load(Ordering::Relaxed);
        metrics.performance.current_sequence_id = self.current_sequence_id.load(Ordering::Relaxed);

        metrics.generation.current_sequence = self.current_sequence_id.load(Ordering::Relaxed);
        metrics.generation.total_tokens = self.total_tokens_generated.load(Ordering::Relaxed);

        metrics.cache_stats = cache_stats;

        metrics
    }

    /// Update generation statistics atomically with blazing-fast performance
    #[inline(always)]
    pub(super) fn update_generation_stats(&self, tokens_generated: u64, duration_nanos: u64) {
        self.total_tokens_generated
            .fetch_add(tokens_generated, Ordering::Relaxed);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        self.last_generation_time.store(now, Ordering::Relaxed);

        if duration_nanos > 0 {
            let tokens_per_second = (tokens_generated * 1_000_000_000) / duration_nanos;

            // Update moving average with zero-allocation calculation
            let current_avg = self.avg_tokens_per_second.load(Ordering::Relaxed);
            let new_avg = if current_avg == 0 {
                tokens_per_second
            } else {
                // Exponential moving average with decay factor 0.9
                ((current_avg as f64 * 0.9) + (tokens_per_second as f64 * 0.1)) as u64
            };

            self.avg_tokens_per_second.store(new_avg, Ordering::Relaxed);
        }
    }
}
