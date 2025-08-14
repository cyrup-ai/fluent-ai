//! Core HTTP client structure and basic functionality
//!
//! Provides the foundational HttpClient struct with essential accessor methods
//! and basic functionality for HTTP operations.

use super::{ClientStats, ClientStatsSnapshot};
use crate::HttpConfig;

/// Lightweight HTTP client that orchestrates specialized operation modules
#[derive(Debug, Clone)]
pub struct HttpClient {
    /// Underlying HTTP client (delegated to operation modules)
    pub(crate) inner: crate::hyper::Client,
    /// Client configuration
    pub(crate) config: HttpConfig,
    /// Atomic metrics for lock-free statistics
    pub(crate) stats: ClientStats,
}

impl HttpClient {
    /// Create new HttpClient with provided components
    ///
    /// This is an internal constructor used by the configuration module.
    /// External users should use `HttpClient::with_config()` or `HttpClient::default()`.
    #[inline]
    pub(crate) fn new(inner: crate::hyper::Client, config: HttpConfig, stats: ClientStats) -> Self {
        Self {
            inner,
            config,
            stats,
        }
    }

    /// Returns a reference to the inner `crate::hyper::Client`.
    ///
    /// Provides access to the underlying HTTP client for advanced operations
    /// that require direct http3 functionality.
    #[inline]
    pub fn inner(&self) -> &crate::hyper::Client {
        &self.inner
    }

    /// Returns a reference to the `HttpConfig`.
    ///
    /// Allows inspection of the current client configuration including
    /// timeouts, HTTP version preferences, and protocol settings.
    #[inline]
    pub fn config(&self) -> &HttpConfig {
        &self.config
    }

    /// Get client statistics snapshot
    ///
    /// Returns an immutable snapshot of current client statistics including
    /// request counts, timing metrics, and success/failure rates. Statistics
    /// are gathered atomically for consistent reporting.
    #[inline]
    pub fn stats_snapshot(&self) -> ClientStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get mutable reference to client statistics for internal updates
    ///
    /// Used by execution modules to update statistics during request processing.
    /// This method is internal to the client modules.
    #[inline]
    pub(crate) fn stats_mut(&mut self) -> &mut ClientStats {
        &mut self.stats
    }

    /// Get client statistics for monitoring and observability
    ///
    /// Returns statistical information about the HTTP client including
    /// request counts, success rates, response times, and cache performance.
    /// This is part of the public API for application monitoring.
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::HttpClient;
    ///
    /// let client = HttpClient::new();
    /// let stats = client.stats();
    /// println!("Success rate: {:.2}%", stats.success_rate() * 100.0);
    /// ```
    #[inline]
    pub fn stats(&self) -> &ClientStats {
        &self.stats
    }

    /// Reset all client statistics to zero
    ///
    /// Clears all accumulated statistics including request counts, response times,
    /// success/failure counts, and cache metrics. Useful for testing or periodic
    /// monitoring reset scenarios.
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::HttpClient;
    /// use fluent_ai_http3::config::Config;
    ///
    /// let mut client = HttpClient::with_config(Config::default())
    ///     .expect("Failed to create HTTP client with default config");
    /// client.reset_stats();
    /// assert_eq!(client.stats().request_count(), 0);
    /// ```
    pub fn reset_stats(&mut self) {
        use std::sync::atomic::Ordering;
        let stats = self.stats_mut();

        // Reset all atomic counters to zero
        stats.request_count.store(0, Ordering::Relaxed);
        stats.connection_count.store(0, Ordering::Relaxed);
        stats.total_bytes_sent.store(0, Ordering::Relaxed);
        stats.total_bytes_received.store(0, Ordering::Relaxed);
        stats.total_response_time_nanos.store(0, Ordering::Relaxed);
        stats.successful_requests.store(0, Ordering::Relaxed);
        stats.failed_requests.store(0, Ordering::Relaxed);
        stats.cache_hits.store(0, Ordering::Relaxed);
        stats.cache_misses.store(0, Ordering::Relaxed);
    }
}
