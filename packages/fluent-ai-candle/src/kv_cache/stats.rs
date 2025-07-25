//! Lock-Free Cache Statistics and Performance Monitoring
//!
//! Ultra-high-performance statistics collection with:
//! - Atomic counters for zero-allocation tracking
//! - Cache-line aligned structures for optimal performance
//! - Real-time metrics computation with minimal overhead
//! - Comprehensive performance analytics

use std::sync::atomic::{AtomicU64, Ordering};

/// Lock-free cache statistics with atomic counters
#[repr(C, align(64))] // Cache line aligned
pub struct CacheStats {
    /// Total cache hits
    hits: AtomicU64,

    /// Total cache misses
    misses: AtomicU64,

    /// Total stores
    stores: AtomicU64,

    /// Total evictions
    evictions: AtomicU64,

    /// Total errors
    errors: AtomicU64,

    /// Creation timestamp
    created_at_nanos: u64,

    /// Last activity timestamp
    last_activity_nanos: AtomicU64,
}

impl CacheStats {
    /// Create new statistics
    #[inline(always)]
    pub fn new() -> Self {
        let now = Self::current_time_nanos();
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            stores: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            created_at_nanos: now,
            last_activity_nanos: AtomicU64::new(now),
        }
    }

    /// Record cache hit
    #[inline(always)]
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.update_activity_timestamp();
    }

    /// Record cache miss
    #[inline(always)]
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.update_activity_timestamp();
    }

    /// Record store operation
    #[inline(always)]
    pub fn record_store(&self) {
        self.stores.fetch_add(1, Ordering::Relaxed);
        self.update_activity_timestamp();
    }

    /// Record eviction operations
    #[inline(always)]
    pub fn record_evictions(&self, count: usize) {
        self.evictions.fetch_add(count as u64, Ordering::Relaxed);
        self.update_activity_timestamp();
    }

    /// Record error
    #[inline(always)]
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
        self.update_activity_timestamp();
    }

    /// Get hit count
    #[inline(always)]
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Get miss count
    #[inline(always)]
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Get store count
    #[inline(always)]
    pub fn stores(&self) -> u64 {
        self.stores.load(Ordering::Relaxed)
    }

    /// Get eviction count
    #[inline(always)]
    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    /// Get error count
    #[inline(always)]
    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Get hit ratio
    #[inline(always)]
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.hits();
        let total = hits + self.misses();

        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get miss ratio
    #[inline(always)]
    pub fn miss_ratio(&self) -> f64 {
        1.0 - self.hit_ratio()
    }

    /// Get total operations
    #[inline(always)]
    pub fn total_operations(&self) -> u64 {
        self.hits() + self.misses() + self.stores()
    }

    /// Get operations per second
    #[inline(always)]
    pub fn operations_per_second(&self) -> f64 {
        let total_ops = self.total_operations();
        let elapsed_nanos = Self::current_time_nanos().saturating_sub(self.created_at_nanos);

        if elapsed_nanos > 0 {
            (total_ops as f64) * 1_000_000_000.0 / (elapsed_nanos as f64)
        } else {
            0.0
        }
    }

    /// Get uptime in nanoseconds
    #[inline(always)]
    pub fn uptime_nanos(&self) -> u64 {
        Self::current_time_nanos().saturating_sub(self.created_at_nanos)
    }

    /// Get uptime in seconds
    #[inline(always)]
    pub fn uptime_seconds(&self) -> f64 {
        self.uptime_nanos() as f64 / 1_000_000_000.0
    }

    /// Get last activity timestamp
    #[inline(always)]
    pub fn last_activity_nanos(&self) -> u64 {
        self.last_activity_nanos.load(Ordering::Relaxed)
    }

    /// Get time since last activity in nanoseconds
    #[inline(always)]
    pub fn time_since_last_activity_nanos(&self) -> u64 {
        Self::current_time_nanos().saturating_sub(self.last_activity_nanos())
    }

    /// Get eviction rate (evictions per operation)
    #[inline(always)]
    pub fn eviction_rate(&self) -> f64 {
        let total_ops = self.total_operations();
        if total_ops > 0 {
            self.evictions() as f64 / total_ops as f64
        } else {
            0.0
        }
    }

    /// Get error rate (errors per operation)
    #[inline(always)]
    pub fn error_rate(&self) -> f64 {
        let total_ops = self.total_operations();
        if total_ops > 0 {
            self.errors() as f64 / total_ops as f64
        } else {
            0.0
        }
    }

    /// Check if cache is performing well (high hit ratio, low error rate)
    #[inline(always)]
    pub fn is_performing_well(&self) -> bool {
        self.hit_ratio() > 0.8 && self.error_rate() < 0.01
    }

    /// Check if cache needs attention (low hit ratio or high error rate)
    #[inline(always)]
    pub fn needs_attention(&self) -> bool {
        self.hit_ratio() < 0.5 || self.error_rate() > 0.05
    }

    /// Get performance grade (A-F based on hit ratio and error rate)
    pub fn performance_grade(&self) -> char {
        let hit_ratio = self.hit_ratio();
        let error_rate = self.error_rate();

        if hit_ratio >= 0.95 && error_rate <= 0.001 {
            'A'
        } else if hit_ratio >= 0.9 && error_rate <= 0.005 {
            'B'
        } else if hit_ratio >= 0.8 && error_rate <= 0.01 {
            'C'
        } else if hit_ratio >= 0.6 && error_rate <= 0.05 {
            'D'
        } else {
            'F'
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.stores.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
        let now = Self::current_time_nanos();
        self.last_activity_nanos.store(now, Ordering::Relaxed);
    }

    /// Get comprehensive summary of all metrics
    pub fn summary(&self) -> StatsSummary {
        StatsSummary {
            hits: self.hits(),
            misses: self.misses(),
            stores: self.stores(),
            evictions: self.evictions(),
            errors: self.errors(),
            hit_ratio: self.hit_ratio(),
            miss_ratio: self.miss_ratio(),
            total_operations: self.total_operations(),
            operations_per_second: self.operations_per_second(),
            uptime_seconds: self.uptime_seconds(),
            eviction_rate: self.eviction_rate(),
            error_rate: self.error_rate(),
            performance_grade: self.performance_grade(),
            time_since_last_activity_seconds: self.time_since_last_activity_nanos() as f64 / 1_000_000_000.0,
        }
    }

    /// Get current high-precision timestamp
    #[inline(always)]
    fn current_time_nanos() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    }

    /// Update last activity timestamp
    #[inline(always)]
    fn update_activity_timestamp(&self) {
        self.last_activity_nanos.store(Self::current_time_nanos(), Ordering::Relaxed);
    }
}

impl Default for CacheStats {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheStats")
            .field("hits", &self.hits())
            .field("misses", &self.misses())
            .field("stores", &self.stores())
            .field("evictions", &self.evictions())
            .field("errors", &self.errors())
            .field("hit_ratio", &format!("{:.3}", self.hit_ratio()))
            .field("operations_per_second", &format!("{:.1}", self.operations_per_second()))
            .field("uptime_seconds", &format!("{:.1}", self.uptime_seconds()))
            .field("performance_grade", &self.performance_grade())
            .finish()
    }
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CacheStats(hits={}, misses={}, hit_ratio={:.1}%, ops/sec={:.0}, grade={})",
            self.hits(),
            self.misses(),
            self.hit_ratio() * 100.0,
            self.operations_per_second(),
            self.performance_grade()
        )
    }
}

/// Comprehensive statistics summary
#[derive(Debug, Clone)]
pub struct StatsSummary {
    /// Total hits
    pub hits: u64,
    /// Total misses
    pub misses: u64,
    /// Total stores
    pub stores: u64,
    /// Total evictions
    pub evictions: u64,
    /// Total errors
    pub errors: u64,
    /// Hit ratio (0.0-1.0)
    pub hit_ratio: f64,
    /// Miss ratio (0.0-1.0)
    pub miss_ratio: f64,
    /// Total operations
    pub total_operations: u64,
    /// Operations per second
    pub operations_per_second: f64,
    /// Uptime in seconds
    pub uptime_seconds: f64,
    /// Eviction rate (evictions per operation)
    pub eviction_rate: f64,
    /// Error rate (errors per operation)
    pub error_rate: f64,
    /// Performance grade (A-F)
    pub performance_grade: char,
    /// Time since last activity in seconds
    pub time_since_last_activity_seconds: f64,
}

impl StatsSummary {
    /// Check if cache is idle (no recent activity)
    #[inline(always)]
    pub fn is_idle(&self) -> bool {
        self.time_since_last_activity_seconds > 60.0 // No activity for 1 minute
    }

    /// Check if cache is under heavy load
    #[inline(always)]
    pub fn is_under_heavy_load(&self) -> bool {
        self.operations_per_second > 10000.0
    }

    /// Check if cache has high eviction pressure
    #[inline(always)]
    pub fn has_high_eviction_pressure(&self) -> bool {
        self.eviction_rate > 0.1 // More than 10% of operations result in evictions
    }

    /// Get health status description
    pub fn health_status(&self) -> &'static str {
        match self.performance_grade {
            'A' => "Excellent",
            'B' => "Good", 
            'C' => "Fair",
            'D' => "Poor",
            'F' => "Critical",
            _ => "Unknown"
        }
    }

    /// Get recommendations based on performance
    pub fn recommendations(&self) -> Vec<&'static str> {
        let mut recommendations = Vec::new();

        if self.hit_ratio < 0.7 {
            recommendations.push("Consider increasing cache size");
        }

        if self.eviction_rate > 0.05 {
            recommendations.push("High eviction rate - consider different eviction strategy");
        }

        if self.error_rate > 0.01 {
            recommendations.push("High error rate - check input validation");
        }

        if self.operations_per_second < 100.0 && self.total_operations > 1000 {
            recommendations.push("Low throughput - check for bottlenecks");
        }

        if self.is_idle() {
            recommendations.push("Cache is idle - consider reducing memory allocation");
        }

        if self.has_high_eviction_pressure() {
            recommendations.push("High eviction pressure - increase cache capacity");
        }

        if recommendations.is_empty() {
            recommendations.push("Cache is performing optimally");
        }

        recommendations
    }
}

impl std::fmt::Display for StatsSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Cache Statistics Summary ===")?;
        writeln!(f, "Operations: {} hits, {} misses, {} stores", self.hits, self.misses, self.stores)?;
        writeln!(f, "Performance: {:.1}% hit rate, {:.0} ops/sec, grade {}", 
                 self.hit_ratio * 100.0, self.operations_per_second, self.performance_grade)?;
        writeln!(f, "Evictions: {} ({:.2}% rate)", self.evictions, self.eviction_rate * 100.0)?;
        writeln!(f, "Errors: {} ({:.3}% rate)", self.errors, self.error_rate * 100.0)?;
        writeln!(f, "Uptime: {:.1}s, Last activity: {:.1}s ago", 
                 self.uptime_seconds, self.time_since_last_activity_seconds)?;
        writeln!(f, "Status: {}", self.health_status())?;
        
        writeln!(f, "Recommendations:")?;
        for recommendation in self.recommendations() {
            writeln!(f, "  - {}", recommendation)?;
        }

        Ok(())
    }
}

/// Performance benchmarking utilities
pub struct StatsBenchmark {
    start_time: u64,
    start_operations: u64,
}

impl StatsBenchmark {
    /// Start benchmarking period
    pub fn start(stats: &CacheStats) -> Self {
        Self {
            start_time: CacheStats::current_time_nanos(),
            start_operations: stats.total_operations(),
        }
    }

    /// End benchmarking and get results
    pub fn end(self, stats: &CacheStats) -> BenchmarkResult {
        let end_time = CacheStats::current_time_nanos();
        let end_operations = stats.total_operations();

        let duration_nanos = end_time.saturating_sub(self.start_time);
        let operations_completed = end_operations.saturating_sub(self.start_operations);

        let throughput = if duration_nanos > 0 {
            (operations_completed as f64) * 1_000_000_000.0 / (duration_nanos as f64)
        } else {
            0.0
        };

        BenchmarkResult {
            duration_nanos,
            operations_completed,
            throughput_ops_per_second: throughput,
        }
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Duration of benchmark in nanoseconds
    pub duration_nanos: u64,
    /// Number of operations completed
    pub operations_completed: u64,
    /// Throughput in operations per second
    pub throughput_ops_per_second: f64,
}

impl BenchmarkResult {
    /// Get duration in seconds
    #[inline(always)]
    pub fn duration_seconds(&self) -> f64 {
        self.duration_nanos as f64 / 1_000_000_000.0
    }

    /// Get average time per operation in nanoseconds
    #[inline(always)]
    pub fn avg_time_per_operation_nanos(&self) -> f64 {
        if self.operations_completed > 0 {
            self.duration_nanos as f64 / self.operations_completed as f64
        } else {
            0.0
        }
    }

    /// Get average time per operation in microseconds
    #[inline(always)]
    pub fn avg_time_per_operation_micros(&self) -> f64 {
        self.avg_time_per_operation_nanos() / 1000.0
    }
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Benchmark: {} ops in {:.3}s = {:.0} ops/sec (avg {:.1}μs/op)",
            self.operations_completed,
            self.duration_seconds(),
            self.throughput_ops_per_second,
            self.avg_time_per_operation_micros()
        )
    }
}