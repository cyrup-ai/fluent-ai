//! Core HTTP configuration types and default implementations
//!
//! Provides the main HttpConfig struct and related enums for HTTP client configuration.

use std::time::Duration;

/// HTTP client configuration
///
/// Central configuration struct containing all HTTP client settings including
/// connection pools, timeouts, HTTP/2 and HTTP/3 parameters, and security options.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Maximum number of idle connections per host
    pub pool_max_idle_per_host: usize,

    /// Pool idle timeout
    pub pool_idle_timeout: Duration,

    /// Request timeout
    pub timeout: Duration,

    /// Connection timeout
    pub connect_timeout: Duration,

    /// TCP keep-alive duration
    pub tcp_keepalive: Option<Duration>,

    /// Enable TCP_NODELAY
    pub tcp_nodelay: bool,

    /// Enable HTTP/2 adaptive window
    pub http2_adaptive_window: bool,

    /// HTTP/2 max frame size
    pub http2_max_frame_size: Option<u32>,

    /// Use native root certificates
    pub use_native_certs: bool,

    /// Require HTTPS
    pub https_only: bool,

    /// Enable gzip compression
    pub gzip: bool,

    /// Enable brotli compression
    pub brotli: bool,

    /// Enable deflate compression
    pub deflate: bool,

    /// User agent string
    pub user_agent: String,

    /// Enable HTTP/3 (QUIC)
    pub http3_enabled: bool,

    /// Connection pool size
    pub pool_size: usize,

    /// Maximum number of redirects to follow
    pub max_redirects: usize,

    /// Enable cookie storage
    pub cookie_store: bool,

    /// Enable response compression
    pub response_compression: bool,

    /// Enable request compression
    pub request_compression: bool,

    /// DNS cache duration
    pub dns_cache_duration: Duration,

    /// Enable DNS over HTTPS
    pub dns_over_https: bool,

    /// Enable happy eyeballs for IPv6
    pub happy_eyeballs: bool,

    /// Local address to bind to
    pub local_address: Option<std::net::IpAddr>,

    /// Interface to bind to
    pub interface: Option<String>,

    /// Enable HTTP/2 server push
    pub http2_server_push: bool,

    /// HTTP/2 initial stream window size
    pub http2_initial_stream_window_size: Option<u32>,

    /// HTTP/2 initial connection window size
    pub http2_initial_connection_window_size: Option<u32>,

    /// HTTP/2 max concurrent streams
    pub http2_max_concurrent_streams: Option<u32>,

    /// Enable HTTP/2 keep-alive
    pub http2_keep_alive: bool,

    /// HTTP/2 keep-alive interval
    pub http2_keep_alive_interval: Option<Duration>,

    /// HTTP/2 keep-alive timeout
    pub http2_keep_alive_timeout: Option<Duration>,

    /// Enable HTTP/2 adaptive window scaling
    pub http2_adaptive_window_scaling: bool,

    /// Trust DNS
    pub trust_dns: bool,

    /// Enable metrics collection
    pub metrics_enabled: bool,

    /// Enable tracing
    pub tracing_enabled: bool,

    /// Connection reuse strategy
    pub connection_reuse: ConnectionReuse,

    /// Retry policy
    pub retry_policy: RetryPolicy,

    // ===== HTTP/3 (QUIC) Configuration =====
    /// QUIC connection maximum idle timeout before closing
    /// Controls how long a QUIC connection can remain idle before being closed
    pub quic_max_idle_timeout: Option<Duration>,

    /// QUIC per-stream receive window size in bytes
    /// Controls flow control for individual HTTP/3 streams
    pub quic_stream_receive_window: Option<u32>,

    /// QUIC connection-wide receive window size in bytes  
    /// Controls aggregate flow control across all streams in a connection
    pub quic_receive_window: Option<u32>,

    /// QUIC send window size in bytes
    /// Controls how much data can be sent without acknowledgment
    pub quic_send_window: Option<u64>,

    /// Use BBR congestion control algorithm instead of CUBIC
    /// BBR typically provides better performance over high-latency networks
    pub quic_congestion_bbr: bool,

    /// Enable TLS 1.3 early data (0-RTT) for QUIC connections
    /// Reduces connection establishment latency for resumed connections
    pub tls_early_data: bool,

    /// Maximum HTTP/3 header field section size in bytes
    /// Controls maximum size of HTTP/3 headers to prevent memory exhaustion
    pub h3_max_field_section_size: Option<u64>,

    /// Enable HTTP/3 protocol grease
    /// Sends random grease values to ensure protocol extensibility
    pub h3_enable_grease: bool,
}

/// Connection reuse strategy
///
/// Defines how aggressively the client should reuse existing connections.
#[derive(Debug, Clone)]
pub enum ConnectionReuse {
    /// Reuse connections aggressively for maximum performance
    Aggressive,
    /// Reuse connections conservatively for better reliability
    Conservative,
    /// Disable connection reuse entirely
    Disabled,
}

/// Retry policy configuration
///
/// Configures automatic retry behavior for failed requests including
/// exponential backoff, jitter, and conditions for retry attempts.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retries
    pub max_retries: usize,

    /// Base delay between retries
    pub base_delay: Duration,

    /// Maximum delay between retries
    pub max_delay: Duration,

    /// Exponential backoff factor
    pub backoff_factor: f64,

    /// Jitter factor (0.0 to 1.0)
    pub jitter_factor: f64,

    /// Retry on specific status codes
    pub retry_on_status: Vec<u16>,

    /// Retry on specific errors
    pub retry_on_errors: Vec<RetryableError>,
}

/// Retryable error types
///
/// Classifies the types of errors that should trigger automatic retry attempts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryableError {
    /// Network connectivity errors
    Network,
    /// Request timeout errors
    Timeout,
    /// Connection establishment errors
    Connection,
    /// DNS resolution errors
    Dns,
    /// TLS handshake errors
    Tls,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            pool_max_idle_per_host: 32,
            pool_idle_timeout: Duration::from_secs(90),
            timeout: Duration::from_secs(86400),
            connect_timeout: Duration::from_secs(10),
            tcp_keepalive: Some(Duration::from_secs(60)),
            tcp_nodelay: true,
            http2_adaptive_window: true,
            http2_max_frame_size: Some(1 << 20), // 1MB
            use_native_certs: true,
            https_only: false,
            gzip: true,
            brotli: true,
            deflate: true,
            user_agent: "fluent-ai-http3/0.1.0 (QUIC/HTTP3+rustls)".to_string(),
            http3_enabled: true,
            pool_size: 10,
            max_redirects: 10,
            cookie_store: false,
            response_compression: true,
            request_compression: true,
            dns_cache_duration: Duration::from_secs(300),
            dns_over_https: false,
            happy_eyeballs: true,
            local_address: None,
            interface: None,
            http2_server_push: false,
            http2_initial_stream_window_size: None,
            http2_initial_connection_window_size: None,
            http2_max_concurrent_streams: None,
            http2_keep_alive: true,
            http2_keep_alive_interval: Some(Duration::from_secs(30)),
            http2_keep_alive_timeout: Some(Duration::from_secs(5)),
            http2_adaptive_window_scaling: true,
            trust_dns: false,
            metrics_enabled: true,
            tracing_enabled: false,
            connection_reuse: ConnectionReuse::Aggressive,
            retry_policy: RetryPolicy::default(),

            // HTTP/3 (QUIC) defaults - conservative but optimized values
            quic_max_idle_timeout: Some(Duration::from_secs(30)),
            quic_stream_receive_window: Some(256 * 1024), // 256KB per stream
            quic_receive_window: Some(1024 * 1024),       // 1MB connection window
            quic_send_window: Some(512 * 1024),           // 512KB send window
            quic_congestion_bbr: false,                   // Use CUBIC by default for compatibility
            tls_early_data: false,                        // Disabled by default for security
            h3_max_field_section_size: Some(16 * 1024),   // 16KB header limit
            h3_enable_grease: true,                       // Enable grease for protocol evolution
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_factor: 2.0,
            jitter_factor: 0.1,
            retry_on_status: vec![429, 500, 502, 503, 504],
            retry_on_errors: vec![
                RetryableError::Network,
                RetryableError::Timeout,
                RetryableError::Connection,
                RetryableError::Dns,
            ],
        }
    }
}

impl HttpConfig {
    /// Enable or disable HTTP/3 (QUIC) support
    ///
    /// Controls whether the client will attempt to use HTTP/3 over QUIC
    /// when available. HTTP/3 can provide better performance and features
    /// compared to HTTP/2, but may have compatibility considerations.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable HTTP/3 support
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_http3(true);
    /// assert!(config.http3_enabled);
    /// ```
    pub fn with_http3(mut self, enabled: bool) -> Self {
        self.http3_enabled = enabled;
        self
    }

    /// Enable or disable compression algorithms
    ///
    /// Controls support for gzip, brotli, and deflate compression.
    /// Compression reduces bandwidth usage but increases CPU overhead.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable compression support
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_compression(true);
    /// assert!(config.gzip && config.brotli && config.deflate);
    /// ```
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.gzip = enabled;
        self.brotli = enabled;
        self.deflate = enabled;
        self
    }

    /// Enable or disable metrics collection
    ///
    /// Controls whether the client collects performance and usage metrics.
    /// Metrics can be useful for monitoring and debugging but add overhead.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable metrics collection
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_metrics(true);
    /// assert!(config.metrics_enabled);
    /// ```
    pub fn with_metrics(mut self, enabled: bool) -> Self {
        self.metrics_enabled = enabled;
        self
    }

    /// Enable or disable tracing
    ///
    /// Controls whether the client produces detailed trace information
    /// for debugging and observability. Tracing provides detailed insights
    /// but can significantly impact performance.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable tracing
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::config::HttpConfig;
    ///
    /// let config = HttpConfig::default()
    ///     .with_tracing(true);
    /// assert!(config.tracing_enabled);
    /// ```
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.tracing_enabled = enabled;
        self
    }
}
