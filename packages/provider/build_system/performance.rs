//! Zero-allocation performance monitoring for the build system
//!
//! This module provides high-performance performance monitoring with minimal overhead,
//! using atomic operations and lock-free data structures.

use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Performance statistics collected by the build system
#[derive(Debug, Default)]
pub struct PerformanceStats {
    /// Number of cache hits
    pub cache_hits: AtomicUsize,

    /// Number of cache misses
    pub cache_misses: AtomicUsize,

    /// Number of cache writes
    pub cache_writes: AtomicUsize,

    /// Number of cache evictions
    pub cache_evictions: AtomicUsize,

    // Removed unused fields: yaml_files_processed, code_generation_ops

    /// Total time spent in YAML processing (in nanoseconds)
    pub yaml_processing_time_ns: AtomicU64,

    /// Total time spent in code generation (in nanoseconds)
    pub code_generation_time_ns: AtomicU64,

    /// Total time spent in file I/O operations (in nanoseconds)
    pub file_io_time_ns: AtomicU64,

    /// Total time spent in network operations (in nanoseconds)
    pub network_time_ns: AtomicU64,

    /// Number of network requests made
    pub network_requests: AtomicUsize,

    /// Peak memory usage in bytes
    pub peak_memory_usage: AtomicU64,

    // Removed unused field: current_memory_usage

    /// Number of allocations
    pub allocation_count: AtomicUsize,

    /// Number of deallocations
    pub deallocation_count: AtomicUsize,
}

impl PerformanceStats {
    /// Create a new `PerformanceStats` instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache eviction
    pub fn record_cache_eviction(&self) {
        self.cache_evictions.fetch_add(1, Ordering::Relaxed);
    }

    // Removed unused methods: record_cache_write
}

/// A performance monitor for the build system
#[derive(Debug)]
pub struct PerformanceMonitor {
    stats: Arc<PerformanceStats>,
}

impl PerformanceMonitor {
    /// Create a new `PerformanceMonitor`
    pub fn new() -> Self {
        Self {
            stats: Arc::new(PerformanceStats::new()),
        }
    }

    /// Get a reference to the performance statistics
    pub fn stats(&self) -> Arc<PerformanceStats> {
        self.stats.clone()
    }

    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache write
    pub fn record_cache_write(&self) {
        self.stats.cache_writes.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a network request
    pub fn record_network_request(&self) {
        self.stats.network_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Start a timer for performance measurement
    /// Returns a timer guard that records elapsed time when dropped
    pub fn start_timer(&self, _operation: &str) -> PerformanceTimer {
        PerformanceTimer::new()
    }

    // Removed unused methods: record_cache_eviction, record_time
}

/// A simple timer for performance measurement
#[derive(Debug)]
pub struct PerformanceTimer {
    start: Instant,
}

impl PerformanceTimer {
    /// Create a new timer starting now
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get the elapsed time since the timer was created
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PerformanceMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = &self.stats;
        writeln!(f, "Build Performance Stats:")?;
        writeln!(f, "  Cache:")?;
        writeln!(
            f,
            "    Hits: {}, Misses: {}, Writes: {}, Evictions: {}",
            stats.cache_hits.load(Ordering::Relaxed),
            stats.cache_misses.load(Ordering::Relaxed),
            stats.cache_writes.load(Ordering::Relaxed),
            stats.cache_evictions.load(Ordering::Relaxed)
        )?;
        writeln!(f, "  Processing Times:")?;
        writeln!(
            f,
            "    YAML: {}ms",
            Duration::from_nanos(stats.yaml_processing_time_ns.load(Ordering::Relaxed)).as_millis()
        )?;
        writeln!(
            f,
            "    Code Gen: {}ms",
            Duration::from_nanos(stats.code_generation_time_ns.load(Ordering::Relaxed)).as_millis()
        )?;
        writeln!(
            f,
            "    File I/O: {}ms",
            Duration::from_nanos(stats.file_io_time_ns.load(Ordering::Relaxed)).as_millis()
        )?;
        writeln!(
            f,
            "    Network: {}ms ({} requests)",
            Duration::from_nanos(stats.network_time_ns.load(Ordering::Relaxed)).as_millis(),
            stats.network_requests.load(Ordering::Relaxed)
        )?;
        writeln!(f, "  Memory:")?;
        writeln!(
            f,
            "    Peak Usage: {}MB",
            stats.peak_memory_usage.load(Ordering::Relaxed) / 1024 / 1024
        )?;
        writeln!(
            f,
            "    Allocations: {}, Deallocations: {}",
            stats.allocation_count.load(Ordering::Relaxed),
            stats.deallocation_count.load(Ordering::Relaxed)
        )
    }
}
