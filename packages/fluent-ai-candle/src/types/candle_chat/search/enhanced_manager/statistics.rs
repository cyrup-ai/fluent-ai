//! Statistics tracking for enhanced history manager
//!
//! This module provides comprehensive statistics and performance
//! monitoring for the history management system.

use serde::{Deserialize, Serialize};

/// Statistics for history manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryManagerStatistics {
    /// Total messages managed
    pub total_messages: usize,
    /// Total searches performed
    pub total_searches: usize,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Average search time
    pub avg_search_time_ms: f64,
    /// Index size in bytes
    pub index_size_bytes: usize,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Last cleanup timestamp
    pub last_cleanup: chrono::DateTime<chrono::Utc>,
    /// Performance statistics
    pub performance_stats: HashMap<String, f64>,
    /// Error counts
    pub error_counts: HashMap<String, usize>}

impl Default for HistoryManagerStatistics {
    fn default() -> Self {
        Self {
            total_messages: 0,
            total_searches: 0,
            cache_hit_rate: 0.0,
            avg_search_time_ms: 0.0,
            index_size_bytes: 0,
            memory_usage_bytes: 0,
            last_cleanup: chrono::Utc::now(),
            performance_stats: HashMap::new(),
            error_counts: HashMap::new()}
    }
}

impl HistoryManagerStatistics {
    /// Update search statistics
    pub fn update_search_stats(&mut self, search_time_ms: f64, cache_hit: bool) {
        self.total_searches += 1;
        
        // Update average search time
        let total_time = self.avg_search_time_ms * (self.total_searches - 1) as f64;
        self.avg_search_time_ms = (total_time + search_time_ms) / self.total_searches as f64;
        
        // Update cache hit rate
        if cache_hit {
            let cache_hits = (self.cache_hit_rate * (self.total_searches - 1) as f32) + 1.0;
            self.cache_hit_rate = cache_hits / self.total_searches as f32;
        } else {
            let cache_hits = self.cache_hit_rate * (self.total_searches - 1) as f32;
            self.cache_hit_rate = cache_hits / self.total_searches as f32;
        }
    }

    /// Update message count
    pub fn update_message_count(&mut self, count: usize) {
        self.total_messages += count;
    }

    /// Update index size
    pub fn update_index_size(&mut self, size_bytes: usize) {
        self.index_size_bytes = size_bytes;
    }

    /// Update memory usage
    pub fn update_memory_usage(&mut self, usage_bytes: usize) {
        self.memory_usage_bytes = usage_bytes;
    }

    /// Record cleanup event
    pub fn record_cleanup(&mut self) {
        self.last_cleanup = chrono::Utc::now();
    }

    /// Add performance metric
    pub fn add_performance_metric(&mut self, metric_name: String, value: f64) {
        self.performance_stats.insert(metric_name, value);
    }

    /// Increment error count
    pub fn increment_error(&mut self, error_type: String) {
        *self.error_counts.entry(error_type).or_insert(0) += 1;
    }

    /// Get formatted statistics report
    pub fn get_report(&self) -> String {
        format!(
            "History Manager Statistics:\n\
             Total Messages: {}\n\
             Total Searches: {}\n\
             Cache Hit Rate: {:.1}%\n\
             Average Search Time: {:.2}ms\n\
             Index Size: {} bytes\n\
             Memory Usage: {} bytes\n\
             Last Cleanup: {}",
            self.total_messages,
            self.total_searches,
            self.cache_hit_rate * 100.0,
            self.avg_search_time_ms,
            self.index_size_bytes,
            self.memory_usage_bytes,
            self.last_cleanup.format("%Y-%m-%d %H:%M:%S UTC")
        )
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &HashMap<String, f64> {
        &self.performance_stats
    }

    /// Get error counts
    pub fn get_error_counts(&self) -> &HashMap<String, usize> {
        &self.error_counts
    }
}