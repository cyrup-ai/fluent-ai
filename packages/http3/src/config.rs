//! HTTP client configuration

use std::time::Duration;

/// HTTP client configuration
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
    pub h3_enable_grease: bool}

/// Connection reuse strategy
#[derive(Debug, Clone)]
pub enum ConnectionReuse {
    /// Reuse connections aggressively
    Aggressive,
    /// Reuse connections conservatively
    Conservative,
    /// Disable connection reuse
    Disabled}

/// Retry policy configuration
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
    pub retry_on_errors: Vec<RetryableError>}

/// Retryable error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryableError {
    /// Network errors
    Network,
    /// Timeout errors
    Timeout,
    /// Connection errors
    Connection,
    /// DNS errors
    Dns,
    /// TLS errors
    Tls}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            pool_max_idle_per_host: 32,
            pool_idle_timeout: Duration::from_secs(90),
            timeout: Duration::from_secs(300),
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
            quic_receive_window: Some(1024 * 1024), // 1MB connection window
            quic_send_window: Some(512 * 1024), // 512KB send window
            quic_congestion_bbr: false, // Use CUBIC by default for compatibility
            tls_early_data: false, // Disabled by default for security
            h3_max_field_section_size: Some(16 * 1024), // 16KB header limit
            h3_enable_grease: true} // Enable grease for protocol evolution
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
            ]}
    }
}

impl HttpConfig {
    /// Create a new configuration optimized for AI providers
    pub fn ai_optimized() -> Self {
        Self {
            pool_max_idle_per_host: 64,
            pool_idle_timeout: Duration::from_secs(120),
            timeout: Duration::from_secs(300),
            connect_timeout: Duration::from_secs(5),
            tcp_keepalive: Some(Duration::from_secs(30)),
            tcp_nodelay: true,
            http2_adaptive_window: true,
            http2_max_frame_size: Some(2 << 20), // 2MB for large responses
            use_native_certs: true,
            https_only: true,
            gzip: true,
            brotli: true,
            deflate: true,
            user_agent: "fluent-ai-http3/0.1.0 (AI-optimized QUIC/HTTP3+rustls)".to_string(),
            http3_enabled: true,
            pool_size: 20,
            max_redirects: 3,
            cookie_store: false,
            response_compression: true,
            request_compression: true,
            dns_cache_duration: Duration::from_secs(600),
            dns_over_https: true,
            happy_eyeballs: true,
            local_address: None,
            interface: None,
            http2_server_push: false,
            http2_initial_stream_window_size: Some(2 << 20), // 2MB
            http2_initial_connection_window_size: Some(8 << 20), // 8MB
            http2_max_concurrent_streams: Some(100),
            http2_keep_alive: true,
            http2_keep_alive_interval: Some(Duration::from_secs(20)),
            http2_keep_alive_timeout: Some(Duration::from_secs(5)),
            http2_adaptive_window_scaling: true,
            trust_dns: true,
            metrics_enabled: true,
            tracing_enabled: true,
            connection_reuse: ConnectionReuse::Aggressive,
            retry_policy: RetryPolicy {
                max_retries: 5,
                base_delay: Duration::from_millis(50),
                max_delay: Duration::from_secs(60),
                backoff_factor: 1.5,
                jitter_factor: 0.2,
                retry_on_status: vec![429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
                retry_on_errors: vec![
                    RetryableError::Network,
                    RetryableError::Timeout,
                    RetryableError::Connection,
                    RetryableError::Dns,
                    RetryableError::Tls,
                ],
            },
            
            // AI-optimized HTTP/3 (QUIC) settings for maximum performance
            quic_max_idle_timeout: Some(Duration::from_secs(120)), // Longer idle for AI workloads
            quic_stream_receive_window: Some(512 * 1024), // 512KB per stream for large AI responses
            quic_receive_window: Some(4 * 1024 * 1024), // 4MB connection window for high throughput
            quic_send_window: Some(2 * 1024 * 1024), // 2MB send window for large requests
            quic_congestion_bbr: true, // BBR for optimal AI provider performance
            tls_early_data: true, // Enable 0-RTT for repeat connections
            h3_max_field_section_size: Some(64 * 1024), // 64KB for large AI headers
            h3_enable_grease: true} // Enable grease for future compatibility
    }

    /// Create a new configuration optimized for streaming
    pub fn streaming_optimized() -> Self {
        let mut config = Self::ai_optimized();
        config.timeout = Duration::from_secs(600); // 10 minutes for streaming
        config.pool_idle_timeout = Duration::from_secs(300); // 5 minutes
        config.http2_initial_stream_window_size = Some(4 << 20); // 4MB
        config.http2_initial_connection_window_size = Some(16 << 20); // 16MB
        config.http2_max_concurrent_streams = Some(10); // Fewer concurrent streams for streaming
        config.http2_keep_alive_interval = Some(Duration::from_secs(10)); // More frequent keep-alive
        config.retry_policy.max_retries = 2; // Fewer retries for streaming
        
        // Streaming-optimized QUIC settings
        config.quic_max_idle_timeout = Some(Duration::from_secs(600)); // Match streaming timeout
        config.quic_stream_receive_window = Some(1024 * 1024); // 1MB per stream for streaming
        config.quic_receive_window = Some(8 * 1024 * 1024); // 8MB connection window for streaming
        config.quic_send_window = Some(4 * 1024 * 1024); // 4MB send window for streaming
        config.h3_max_field_section_size = Some(32 * 1024); // 32KB headers for streaming metadata
        
        config
    }

    /// Create a new configuration optimized for batch processing
    pub fn batch_optimized() -> Self {
        let mut config = Self::ai_optimized();
        config.pool_max_idle_per_host = 128;
        config.pool_size = 50;
        config.timeout = Duration::from_secs(900); // 15 minutes for batch
        config.http2_max_concurrent_streams = Some(200);
        config.retry_policy.max_retries = 10;
        config.retry_policy.max_delay = Duration::from_secs(120);
        
        // Batch-optimized QUIC settings for high throughput
        config.quic_max_idle_timeout = Some(Duration::from_secs(300)); // 5 minutes for batch efficiency
        config.quic_stream_receive_window = Some(2 * 1024 * 1024); // 2MB per stream for batch
        config.quic_receive_window = Some(16 * 1024 * 1024); // 16MB connection window for batch
        config.quic_send_window = Some(8 * 1024 * 1024); // 8MB send window for batch uploads
        config.h3_max_field_section_size = Some(128 * 1024); // 128KB headers for batch metadata
        
        config
    }

    /// Create a new configuration for low-latency applications
    pub fn low_latency() -> Self {
        let mut config = Self::ai_optimized();
        config.connect_timeout = Duration::from_secs(2);
        config.timeout = Duration::from_secs(10);
        config.tcp_keepalive = Some(Duration::from_secs(10));
        config.http2_keep_alive_interval = Some(Duration::from_secs(5));
        config.retry_policy.max_retries = 1;
        config.retry_policy.base_delay = Duration::from_millis(10);
        config.retry_policy.max_delay = Duration::from_secs(5);
        
        // Low-latency optimized QUIC settings
        config.quic_max_idle_timeout = Some(Duration::from_secs(15)); // Short idle for low latency
        config.quic_stream_receive_window = Some(128 * 1024); // 128KB per stream for low latency
        config.quic_receive_window = Some(512 * 1024); // 512KB connection window for low latency
        config.quic_send_window = Some(256 * 1024); // 256KB send window for low latency
        config.quic_congestion_bbr = true; // BBR for better latency characteristics
        config.tls_early_data = true; // Critical for low latency - enable 0-RTT
        config.h3_max_field_section_size = Some(8 * 1024); // 8KB headers to minimize latency
        
        config
    }

    /// Enable HTTP/3 (QUIC) support
    pub fn with_http3(mut self, enabled: bool) -> Self {
        self.http3_enabled = enabled;
        self
    }

    /// Set the timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the connect timeout
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Set the pool size
    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    /// Set the user agent
    pub fn with_user_agent(mut self, user_agent: String) -> Self {
        self.user_agent = user_agent;
        self
    }

    /// Enable or disable compression
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.gzip = enabled;
        self.brotli = enabled;
        self.deflate = enabled;
        self
    }

    /// Enable or disable HTTPS only
    pub fn with_https_only(mut self, enabled: bool) -> Self {
        self.https_only = enabled;
        self
    }
    
    // ===== HTTP/3 (QUIC) Configuration Methods =====
    
    /// Set QUIC connection maximum idle timeout
    pub fn with_quic_max_idle_timeout(mut self, timeout: Duration) -> Self {
        self.quic_max_idle_timeout = Some(timeout);
        self
    }
    
    /// Set QUIC per-stream receive window size
    pub fn with_quic_stream_receive_window(mut self, window_size: u32) -> Self {
        self.quic_stream_receive_window = Some(window_size);
        self
    }
    
    /// Set QUIC connection-wide receive window size  
    pub fn with_quic_receive_window(mut self, window_size: u32) -> Self {
        self.quic_receive_window = Some(window_size);
        self
    }
    
    /// Set QUIC send window size
    pub fn with_quic_send_window(mut self, window_size: u64) -> Self {
        self.quic_send_window = Some(window_size);
        self
    }
    
    /// Enable or disable BBR congestion control algorithm
    pub fn with_quic_congestion_bbr(mut self, enabled: bool) -> Self {
        self.quic_congestion_bbr = enabled;
        self
    }
    
    /// Enable or disable TLS 1.3 early data (0-RTT)
    pub fn with_tls_early_data(mut self, enabled: bool) -> Self {
        self.tls_early_data = enabled;
        self
    }
    
    /// Set maximum HTTP/3 header field section size
    pub fn with_h3_max_field_section_size(mut self, size: u64) -> Self {
        self.h3_max_field_section_size = Some(size);
        self
    }
    
    /// Enable or disable HTTP/3 protocol grease
    pub fn with_h3_enable_grease(mut self, enabled: bool) -> Self {
        self.h3_enable_grease = enabled;
        self
    }

    /// Set the retry policy
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Enable or disable metrics
    pub fn with_metrics(mut self, enabled: bool) -> Self {
        self.metrics_enabled = enabled;
        self
    }

    /// Enable or disable tracing
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.tracing_enabled = enabled;
        self
    }
}
