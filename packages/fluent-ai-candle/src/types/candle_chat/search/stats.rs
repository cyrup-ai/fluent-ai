//! Search statistics and metrics tracking
//!
//! This module provides comprehensive statistics tracking with atomic counters,
//! performance monitoring, and zero-allocation patterns for production use.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_utils::CachePadded;
use fluent_ai_async::{AsyncStream, emit, handle_error};
use serde::{Deserialize, Serialize};

use super::types::SearchStatistics;

/// Performance metrics for search operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchPerformanceMetrics {
    /// Average query processing time in microseconds
    pub avg_query_time_us: f64,
    /// Minimum query processing time in microseconds
    pub min_query_time_us: u64,
    /// Maximum query processing time in microseconds
    pub max_query_time_us: u64,
    /// 95th percentile query time in microseconds
    pub p95_query_time_us: u64,
    /// 99th percentile query time in microseconds
    pub p99_query_time_us: u64,
    /// Queries per second
    pub queries_per_second: f64,
    /// Index operations per second
    pub index_ops_per_second: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
}

/// Search statistics tracker with atomic counters
#[derive(Debug)]
pub struct SearchStatsTracker {
    /// Total search queries performed - cache-padded for performance
    total_queries: CachePadded<AtomicUsize>,
    /// Total index operations performed
    total_index_ops: CachePadded<AtomicUsize>,
    /// Total cache hits
    cache_hits: CachePadded<AtomicUsize>,
    /// Total cache misses
    cache_misses: CachePadded<AtomicUsize>,
    /// Total query processing time in microseconds
    total_query_time_us: CachePadded<AtomicU64>,
    /// Minimum query time in microseconds
    min_query_time_us: CachePadded<AtomicU64>,
    /// Maximum query time in microseconds
    max_query_time_us: CachePadded<AtomicU64>,
    /// Start time for rate calculations
    start_time: Instant,
    /// Query time samples for percentile calculations (simplified)
    query_times: Arc<parking_lot::Mutex<Vec<u64>>>,
}

impl SearchStatsTracker {
    /// Create a new statistics tracker
    pub fn new() -> Self {
        Self {
            total_queries: CachePadded::new(AtomicUsize::new(0)),
            total_index_ops: CachePadded::new(AtomicUsize::new(0)),
            cache_hits: CachePadded::new(AtomicUsize::new(0)),
            cache_misses: CachePadded::new(AtomicUsize::new(0)),
            total_query_time_us: CachePadded::new(AtomicU64::new(0)),
            min_query_time_us: CachePadded::new(AtomicU64::new(u64::MAX)),
            max_query_time_us: CachePadded::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            query_times: Arc::new(parking_lot::Mutex::new(Vec::new())),
        }
    }

    /// Record a search query with timing
    pub fn record_query(&self, duration: Duration) {
        let duration_us = duration.as_micros() as u64;
        
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        self.total_query_time_us.fetch_add(duration_us, Ordering::Relaxed);
        
        // Update min/max times
        self.update_min_time(duration_us);
        self.update_max_time(duration_us);
        
        // Store sample for percentile calculations (with size limit)
        if let Ok(mut times) = self.query_times.try_lock() {
            times.push(duration_us);
            // Keep only recent samples to avoid unbounded growth
            if times.len() > 10000 {
                times.drain(0..5000); // Remove oldest half
            }
        }
    }

    /// Record an index operation
    pub fn record_index_operation(&self) {
        self.total_index_ops.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current statistics using fluent-ai-async streaming architecture
    pub fn get_statistics(&self) -> AsyncStream<SearchStatistics> {
        let total_queries = self.total_queries.load(Ordering::Relaxed);
        let total_index_ops = self.total_index_ops.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        let total_query_time_us = self.total_query_time_us.load(Ordering::Relaxed);
        
        AsyncStream::with_channel(move |sender| {
            let avg_query_time = if total_queries > 0 {
                (total_query_time_us as f64) / (total_queries as f64) / 1000.0 // Convert to milliseconds
            } else {
                0.0
            };

            let stats = SearchStatistics {
                total_messages: 0, // Would be provided by index
                total_terms: 0,    // Would be provided by index
                total_queries,
                average_query_time: avg_query_time,
                index_size: 0,     // Would be calculated by index
                last_index_update: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                search_operations: total_queries,
                index_operations: total_index_ops,
                cache_hits,
                cache_misses,
            };

            emit!(sender, stats);
        })
    }

    /// Get performance metrics using fluent-ai-async streaming architecture
    pub fn get_performance_metrics(&self) -> AsyncStream<SearchPerformanceMetrics> {
        let total_queries = self.total_queries.load(Ordering::Relaxed);
        let total_index_ops = self.total_index_ops.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        let total_query_time_us = self.total_query_time_us.load(Ordering::Relaxed);
        let min_time = self.min_query_time_us.load(Ordering::Relaxed);
        let max_time = self.max_query_time_us.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed();
        let query_times = Arc::clone(&self.query_times);
        
        AsyncStream::with_channel(move |sender| {
            let avg_query_time_us = if total_queries > 0 {
                (total_query_time_us as f64) / (total_queries as f64)
            } else {
                0.0
            };

            let queries_per_second = if elapsed.as_secs() > 0 {
                (total_queries as f64) / elapsed.as_secs_f64()
            } else {
                0.0
            };

            let index_ops_per_second = if elapsed.as_secs() > 0 {
                (total_index_ops as f64) / elapsed.as_secs_f64()
            } else {
                0.0
            };

            let cache_hit_rate = if cache_hits + cache_misses > 0 {
                (cache_hits as f64) / ((cache_hits + cache_misses) as f64)
            } else {
                0.0
            };

            // Calculate percentiles (simplified)
            let (p95, p99) = if let Ok(times) = query_times.try_lock() {
                Self::calculate_percentiles(&times)
            } else {
                (0, 0)
            };

            let metrics = SearchPerformanceMetrics {
                avg_query_time_us,
                min_query_time_us: if min_time == u64::MAX { 0 } else { min_time },
                max_query_time_us: max_time,
                p95_query_time_us: p95,
                p99_query_time_us: p99,
                queries_per_second,
                index_ops_per_second,
                cache_hit_rate,
            };

            emit!(sender, metrics);
        })
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.total_queries.store(0, Ordering::Relaxed);
        self.total_index_ops.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.total_query_time_us.store(0, Ordering::Relaxed);
        self.min_query_time_us.store(u64::MAX, Ordering::Relaxed);
        self.max_query_time_us.store(0, Ordering::Relaxed);
        
        if let Ok(mut times) = self.query_times.try_lock() {
            times.clear();
        }
    }

    /// Update minimum query time atomically
    fn update_min_time(&self, new_time: u64) {
        let mut current = self.min_query_time_us.load(Ordering::Relaxed);
        while new_time < current {
            match self.min_query_time_us.compare_exchange_weak(
                current,
                new_time,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    /// Update maximum query time atomically
    fn update_max_time(&self, new_time: u64) {
        let mut current = self.max_query_time_us.load(Ordering::Relaxed);
        while new_time > current {
            match self.max_query_time_us.compare_exchange_weak(
                current,
                new_time,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    /// Calculate percentiles from query time samples
    fn calculate_percentiles(times: &[u64]) -> (u64, u64) {
        if times.is_empty() {
            return (0, 0);
        }

        let mut sorted_times = times.to_vec();
        sorted_times.sort_unstable();

        let p95_index = (sorted_times.len() as f64 * 0.95) as usize;
        let p99_index = (sorted_times.len() as f64 * 0.99) as usize;

        let p95 = sorted_times.get(p95_index.saturating_sub(1)).copied().unwrap_or(0);
        let p99 = sorted_times.get(p99_index.saturating_sub(1)).copied().unwrap_or(0);

        (p95, p99)
    }

    /// Get total queries count
    pub fn total_queries(&self) -> usize {
        self.total_queries.load(Ordering::Relaxed)
    }

    /// Get total index operations count
    pub fn total_index_operations(&self) -> usize {
        self.total_index_ops.load(Ordering::Relaxed)
    }

    /// Get cache hit count
    pub fn cache_hits(&self) -> usize {
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// Get cache miss count
    pub fn cache_misses(&self) -> usize {
        self.cache_misses.load(Ordering::Relaxed)
    }

    /// Get uptime duration
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for SearchStatsTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SearchStatsTracker {
    fn clone(&self) -> Self {
        // Create a new tracker with current values
        let new_tracker = Self::new();
        
        // Copy current values
        new_tracker.total_queries.store(
            self.total_queries.load(Ordering::Relaxed),
            Ordering::Relaxed
        );
        new_tracker.total_index_ops.store(
            self.total_index_ops.load(Ordering::Relaxed),
            Ordering::Relaxed
        );
        new_tracker.cache_hits.store(
            self.cache_hits.load(Ordering::Relaxed),
            Ordering::Relaxed
        );
        new_tracker.cache_misses.store(
            self.cache_misses.load(Ordering::Relaxed),
            Ordering::Relaxed
        );
        new_tracker.total_query_time_us.store(
            self.total_query_time_us.load(Ordering::Relaxed),
            Ordering::Relaxed
        );
        new_tracker.min_query_time_us.store(
            self.min_query_time_us.load(Ordering::Relaxed),
            Ordering::Relaxed
        );
        new_tracker.max_query_time_us.store(
            self.max_query_time_us.load(Ordering::Relaxed),
            Ordering::Relaxed
        );
        
        // Copy query times if possible
        if let (Ok(source_times), Ok(mut dest_times)) = 
            (self.query_times.try_lock(), new_tracker.query_times.try_lock()) {
            dest_times.extend_from_slice(&source_times);
        }
        
        new_tracker
    }
}
