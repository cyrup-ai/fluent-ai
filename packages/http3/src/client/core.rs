//! Core HTTP client structure with pure AsyncStream architecture
//!
//! Provides the foundational HttpClient struct using fluent_ai_async directly
//! with NO middleware, NO abstractions - pure streaming protocols

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit};

use super::{ClientStats, ClientStatsSnapshot};
use crate::{HttpChunk, HttpConfig, HttpRequest};

/// Pure streaming HTTP client using fluent_ai_async directly - NO middleware
#[derive(Debug, Clone)]
pub struct HttpClient {
    /// Client configuration
    pub(crate) config: HttpConfig,
    /// Atomic metrics for lock-free statistics
    pub(crate) stats: ClientStats,
}

impl HttpClient {
    /// Create new HttpClient with direct fluent_ai_async streaming - NO middleware
    #[inline]
    pub(crate) fn new_direct(config: HttpConfig, stats: ClientStats) -> Self {
        Self { config, stats }
    }

    /// Execute request using direct H2/H3/Quiche protocols - NO middleware
    #[inline]
    pub fn execute_direct_streaming(&self, request: HttpRequest) -> AsyncStream<HttpChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Determine protocol based on URL scheme and server capabilities
            let uri = request.uri();
            let scheme = uri.scheme_str().unwrap_or("https");

            match scheme {
                "https" => {
                    // Try H3 first, fallback to H2
                    let h3_stream =
                        crate::async_impl::connection::h3_connection::H3Connection::new()
                            .connect_and_request(request.clone());

                    // Forward H3 chunks, converting to HttpChunk
                    for h3_chunk in h3_stream {
                        let http_chunk = HttpChunk::from_h3_chunk(h3_chunk);
                        emit!(sender, http_chunk);
                    }
                }
                "http" => {
                    // Use H2 for HTTP
                    let h2_stream =
                        crate::async_impl::connection::h2_connection::H2ConnectionManager::new()
                            .establish_connection_stream(request);

                    // Forward H2 chunks, converting to HttpChunk
                    for h2_chunk in h2_stream {
                        let http_chunk = HttpChunk::from_h2_chunk(h2_chunk);
                        emit!(sender, http_chunk);
                    }
                }
                _ => {
                    emit!(
                        sender,
                        HttpChunk::bad_chunk(format!("Unsupported scheme: {}", scheme))
                    );
                }
            }
        })
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
    /// match HttpClient::with_config(Config::default()) {
    ///     Ok(mut client) => {
    ///         client.reset_stats();
    ///         assert_eq!(client.stats().request_count(), 0);
    ///     }
    ///     Err(e) => eprintln!("Failed to create HTTP client: {}", e),
    /// }
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
