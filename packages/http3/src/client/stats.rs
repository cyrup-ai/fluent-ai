//! Client statistics tracking with atomic counters
//!
//! Provides thread-safe statistics collection for HTTP client operations
//! using atomic operations for lock-free performance.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Thread-safe client statistics with atomic counters
#[derive(Debug, Default, Clone)]
pub struct ClientStats {
    /// Total number of HTTP requests made
    pub request_count: Arc<AtomicUsize>,
    /// Total number of connections established
    pub connection_count: Arc<AtomicUsize>,
    /// Total bytes sent in request bodies
    pub total_bytes_sent: Arc<AtomicU64>,
    /// Total bytes received in response bodies
    pub total_bytes_received: Arc<AtomicU64>,
    /// Total response time across all requests in nanoseconds
    pub total_response_time_nanos: Arc<AtomicU64>,
    /// Number of successful requests (2xx status codes)
    pub successful_requests: Arc<AtomicUsize>,
    /// Number of failed requests (4xx/5xx status codes or network errors)
    pub failed_requests: Arc<AtomicUsize>,
    /// Number of cache hits
    pub cache_hits: Arc<AtomicUsize>,
    /// Number of cache misses
    pub cache_misses: Arc<AtomicUsize>,
}

/// Immutable snapshot of client statistics at a point in time
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientStatsSnapshot {
    /// Total number of HTTP requests made
    pub request_count: usize,
    /// Total number of connections established
    pub connection_count: usize,
    /// Total bytes sent in request bodies
    pub total_bytes_sent: u64,
    /// Total bytes received in response bodies
    pub total_bytes_received: u64,
    /// Total response time across all requests in nanoseconds
    pub total_response_time_nanos: u64,
    /// Number of successful requests (2xx status codes)
    pub successful_requests: usize,
    /// Number of failed requests (4xx/5xx status codes or network errors)
    pub failed_requests: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
}

impl ClientStats {
    /// Create a snapshot of the current statistics.
    ///
    /// All values are read atomically using relaxed ordering for maximum performance
    /// while ensuring consistent point-in-time values across all metrics.
    #[inline]
    pub fn snapshot(&self) -> ClientStatsSnapshot {
        ClientStatsSnapshot {
            request_count: self.request_count.load(Ordering::Relaxed),
            connection_count: self.connection_count.load(Ordering::Relaxed),
            total_bytes_sent: self.total_bytes_sent.load(Ordering::Relaxed),
            total_bytes_received: self.total_bytes_received.load(Ordering::Relaxed),
            total_response_time_nanos: self.total_response_time_nanos.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
        }
    }
}
