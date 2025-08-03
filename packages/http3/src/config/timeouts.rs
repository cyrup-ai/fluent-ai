//! Timeout and connection-related configuration methods
//!
//! Provides builder methods for configuring timeouts, connection pooling,
//! and QUIC window sizes for optimal performance tuning.

use std::time::Duration;

use super::core::HttpConfig;

impl HttpConfig {
    /// Set the request timeout
    ///
    /// Controls how long to wait for a complete request/response cycle before timing out.
    /// This includes connection establishment, request sending, and response receiving.
    ///
    /// # Arguments
    /// * `timeout` - Maximum duration to wait for request completion
    ///
    /// # Examples
    /// ```no_run
    /// use std::time::Duration;
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_timeout(Duration::from_secs(30));
    /// assert_eq!(config.timeout, Duration::from_secs(30));
    /// ```
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the connection timeout
    ///
    /// Controls how long to wait when establishing initial connections to servers.
    /// This only covers the TCP/QUIC handshake time, not the full request time.
    ///
    /// # Arguments
    /// * `timeout` - Maximum duration to wait for connection establishment
    ///
    /// # Examples
    /// ```no_run
    /// use std::time::Duration;
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_connect_timeout(Duration::from_secs(5));
    /// assert_eq!(config.connect_timeout, Duration::from_secs(5));
    /// ```
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Set the connection pool size
    ///
    /// Controls the maximum number of active connections maintained in the pool.
    /// Larger pools can handle more concurrent requests but use more resources.
    ///
    /// # Arguments
    /// * `size` - Maximum number of connections in the pool
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_pool_size(50);
    /// assert_eq!(config.pool_size, 50);
    /// ```
    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    /// Set the maximum idle connections per host
    ///
    /// Controls how many idle connections to keep alive for each host to enable
    /// connection reuse. Higher values improve performance for repeated requests
    /// to the same host.
    ///
    /// # Arguments
    /// * `max_idle` - Maximum idle connections per host
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_max_idle_per_host(64);
    /// assert_eq!(config.pool_max_idle_per_host, 64);
    /// ```
    pub fn with_max_idle_per_host(mut self, max_idle: usize) -> Self {
        self.pool_max_idle_per_host = max_idle;
        self
    }

    /// Set the pool idle timeout
    ///
    /// Controls how long idle connections are kept alive before being closed.
    /// Longer timeouts improve connection reuse but consume more resources.
    ///
    /// # Arguments
    /// * `timeout` - Duration to keep idle connections alive
    ///
    /// # Examples
    /// ```no_run
    /// use std::time::Duration;
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_pool_idle_timeout(Duration::from_secs(120));
    /// assert_eq!(config.pool_idle_timeout, Duration::from_secs(120));
    /// ```
    pub fn with_pool_idle_timeout(mut self, timeout: Duration) -> Self {
        self.pool_idle_timeout = timeout;
        self
    }

    /// Set TCP keep-alive duration
    ///
    /// Enables TCP keep-alive with the specified interval. This helps detect
    /// dead connections and maintain long-lived connections through NATs/firewalls.
    ///
    /// # Arguments
    /// * `duration` - Keep-alive interval, or None to disable
    ///
    /// # Examples
    /// ```no_run
    /// use std::time::Duration;
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_tcp_keepalive(Some(Duration::from_secs(30)));
    /// assert_eq!(config.tcp_keepalive, Some(Duration::from_secs(30)));
    /// ```
    pub fn with_tcp_keepalive(mut self, duration: Option<Duration>) -> Self {
        self.tcp_keepalive = duration;
        self
    }

    /// Set HTTP/2 keep-alive interval
    ///
    /// Controls how frequently HTTP/2 PING frames are sent to keep connections alive.
    /// More frequent pings improve connection reliability but increase overhead.
    ///
    /// # Arguments
    /// * `interval` - PING frame interval, or None to disable
    ///
    /// # Examples
    /// ```no_run
    /// use std::time::Duration;
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_http2_keep_alive_interval(Some(Duration::from_secs(20)));
    /// assert_eq!(config.http2_keep_alive_interval, Some(Duration::from_secs(20)));
    /// ```
    pub fn with_http2_keep_alive_interval(mut self, interval: Option<Duration>) -> Self {
        self.http2_keep_alive_interval = interval;
        self
    }

    /// Set HTTP/2 keep-alive timeout
    ///
    /// Controls how long to wait for PING frame responses before considering
    /// the connection dead.
    ///
    /// # Arguments
    /// * `timeout` - PING response timeout, or None for default
    ///
    /// # Examples
    /// ```no_run
    /// use std::time::Duration;
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_http2_keep_alive_timeout(Some(Duration::from_secs(10)));
    /// assert_eq!(config.http2_keep_alive_timeout, Some(Duration::from_secs(10)));
    /// ```
    pub fn with_http2_keep_alive_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.http2_keep_alive_timeout = timeout;
        self
    }

    // ===== QUIC Timeout and Window Configuration =====

    /// Set QUIC connection maximum idle timeout
    ///
    /// Controls how long a QUIC connection can remain idle before being closed.
    /// This affects HTTP/3 connection lifecycle and resource usage.
    ///
    /// # Arguments
    /// * `timeout` - Maximum idle time before connection closure
    ///
    /// # Examples
    /// ```no_run
    /// use std::time::Duration;
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_quic_max_idle_timeout(Duration::from_secs(60));
    /// assert_eq!(config.quic_max_idle_timeout, Some(Duration::from_secs(60)));
    /// ```
    pub fn with_quic_max_idle_timeout(mut self, timeout: Duration) -> Self {
        self.quic_max_idle_timeout = Some(timeout);
        self
    }

    /// Set QUIC per-stream receive window size
    ///
    /// Controls flow control for individual HTTP/3 streams. Larger windows
    /// allow more data buffering but use more memory per stream.
    ///
    /// # Arguments
    /// * `window_size` - Receive window size in bytes
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_quic_stream_receive_window(512 * 1024); // 512KB
    /// assert_eq!(config.quic_stream_receive_window, Some(512 * 1024));
    /// ```
    pub fn with_quic_stream_receive_window(mut self, window_size: u32) -> Self {
        self.quic_stream_receive_window = Some(window_size);
        self
    }

    /// Set QUIC connection-wide receive window size
    ///
    /// Controls aggregate flow control across all streams in a connection.
    /// This limits total buffered data for the entire connection.
    ///
    /// # Arguments
    /// * `window_size` - Connection receive window size in bytes
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_quic_receive_window(2 * 1024 * 1024); // 2MB
    /// assert_eq!(config.quic_receive_window, Some(2 * 1024 * 1024));
    /// ```
    pub fn with_quic_receive_window(mut self, window_size: u32) -> Self {
        self.quic_receive_window = Some(window_size);
        self
    }

    /// Set QUIC send window size
    ///
    /// Controls how much data can be sent without acknowledgment. Larger
    /// windows improve throughput over high-latency networks.
    ///
    /// # Arguments
    /// * `window_size` - Send window size in bytes
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_quic_send_window(1024 * 1024); // 1MB
    /// assert_eq!(config.quic_send_window, Some(1024 * 1024));
    /// ```
    pub fn with_quic_send_window(mut self, window_size: u64) -> Self {
        self.quic_send_window = Some(window_size);
        self
    }

    /// Set DNS cache duration
    ///
    /// Controls how long DNS resolution results are cached. Longer caching
    /// improves performance but may delay detection of DNS changes.
    ///
    /// # Arguments
    /// * `duration` - DNS cache lifetime
    ///
    /// # Examples
    /// ```no_run
    /// use std::time::Duration;
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_dns_cache_duration(Duration::from_secs(600)); // 10 minutes
    /// assert_eq!(config.dns_cache_duration, Duration::from_secs(600));
    /// ```
    pub fn with_dns_cache_duration(mut self, duration: Duration) -> Self {
        self.dns_cache_duration = duration;
        self
    }
}
