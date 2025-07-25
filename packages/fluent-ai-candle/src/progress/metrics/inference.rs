//! Inference metrics collection and analysis

use std::time::Duration;

/// Aggregated metrics for inference operations
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Total tokens generated
    pub total_tokens: u64,
    /// Average tokens per second
    pub avg_tokens_per_sec: f64,
    /// Peak tokens per second
    pub peak_tokens_per_sec: f64,
    /// Average inference latency (milliseconds)
    pub avg_latency_ms: f64,
    /// Peak inference latency (milliseconds)
    pub peak_latency_ms: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Total inference operations
    pub total_operations: u64,
    /// Failed operations count
    pub failed_operations: u64,
    /// Current memory usage (MB)
    pub memory_usage_mb: f64,
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64}

impl InferenceMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self {
            total_tokens: 0,
            avg_tokens_per_sec: 0.0,
            peak_tokens_per_sec: 0.0,
            avg_latency_ms: 0.0,
            peak_latency_ms: 0.0,
            cache_hit_rate: 0.0,
            total_operations: 0,
            failed_operations: 0,
            memory_usage_mb: 0.0,
            peak_memory_mb: 0.0}
    }

    /// Update metrics with new generation data
    pub fn update_generation(&mut self, tokens: u64, duration: Duration, cache_hits: u64, cache_total: u64) {
        self.total_tokens += tokens;
        self.total_operations += 1;

        let latency_ms = duration.as_millis() as f64;
        let tokens_per_sec = if duration.as_secs_f64() > 0.0 {
            tokens as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        // Update averages
        let ops = self.total_operations as f64;
        self.avg_latency_ms = (self.avg_latency_ms * (ops - 1.0) + latency_ms) / ops;
        self.avg_tokens_per_sec = (self.avg_tokens_per_sec * (ops - 1.0) + tokens_per_sec) / ops;

        // Update peaks
        if tokens_per_sec > self.peak_tokens_per_sec {
            self.peak_tokens_per_sec = tokens_per_sec;
        }
        if latency_ms > self.peak_latency_ms {
            self.peak_latency_ms = latency_ms;
        }

        // Update cache hit rate
        if cache_total > 0 {
            let hit_rate = cache_hits as f64 / cache_total as f64;
            self.cache_hit_rate = (self.cache_hit_rate * (ops - 1.0) + hit_rate) / ops;
        }
    }

    /// Update memory usage
    pub fn update_memory(&mut self, current_mb: f64) {
        self.memory_usage_mb = current_mb;
        if current_mb > self.peak_memory_mb {
            self.peak_memory_mb = current_mb;
        }
    }

    /// Record failed operation
    pub fn record_failure(&mut self) {
        self.failed_operations += 1;
        self.total_operations += 1;
    }

    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            return 1.0;
        }
        let successful = self.total_operations - self.failed_operations;
        successful as f64 / self.total_operations as f64
    }

    /// Get efficiency score (tokens per second per MB)
    pub fn efficiency_score(&self) -> f64 {
        if self.memory_usage_mb > 0.0 {
            self.avg_tokens_per_sec / self.memory_usage_mb
        } else {
            0.0
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Get throughput summary
    pub fn throughput_summary(&self) -> ThroughputSummary {
        ThroughputSummary {
            avg_tokens_per_sec: self.avg_tokens_per_sec,
            peak_tokens_per_sec: self.peak_tokens_per_sec,
            total_tokens: self.total_tokens,
            total_operations: self.total_operations}
    }

    /// Get latency summary
    pub fn latency_summary(&self) -> LatencySummary {
        LatencySummary {
            avg_latency_ms: self.avg_latency_ms,
            peak_latency_ms: self.peak_latency_ms,
            total_operations: self.total_operations}
    }

    /// Get memory summary
    pub fn memory_summary(&self) -> MemorySummary {
        MemorySummary {
            current_mb: self.memory_usage_mb,
            peak_mb: self.peak_memory_mb,
            efficiency_score: self.efficiency_score()}
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Throughput metrics summary
#[derive(Debug, Clone)]
pub struct ThroughputSummary {
    pub avg_tokens_per_sec: f64,
    pub peak_tokens_per_sec: f64,
    pub total_tokens: u64,
    pub total_operations: u64}

/// Latency metrics summary
#[derive(Debug, Clone)]
pub struct LatencySummary {
    pub avg_latency_ms: f64,
    pub peak_latency_ms: f64,
    pub total_operations: u64}

/// Memory usage summary
#[derive(Debug, Clone)]
pub struct MemorySummary {
    pub current_mb: f64,
    pub peak_mb: f64,
    pub efficiency_score: f64}