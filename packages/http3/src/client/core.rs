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
    pub(crate) inner: reqwest::Client,
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
    pub(crate) fn new(inner: reqwest::Client, config: HttpConfig, stats: ClientStats) -> Self {
        Self {
            inner,
            config,
            stats,
        }
    }

    /// Returns a reference to the inner `reqwest::Client`.
    ///
    /// Provides access to the underlying HTTP client for advanced operations
    /// that require direct reqwest functionality.
    #[inline]
    pub fn inner(&self) -> &reqwest::Client {
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

    /// Get shared reference to client statistics for internal access
    ///
    /// Used by execution modules to read current statistics state.
    /// This method is internal to the client modules.
    #[inline]
    pub(crate) fn stats(&self) -> &ClientStats {
        &self.stats
    }
}
