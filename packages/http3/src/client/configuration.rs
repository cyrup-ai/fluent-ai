//! HTTP client configuration and construction
//!
//! Handles client creation with HTTP/3 and HTTP/2 protocol configuration,
//! TLS settings, compression, and advanced protocol optimizations.

use super::{ClientStats, HttpClient};
use crate::{HttpConfig, HttpError};

impl HttpClient {
    /// Create a new HttpClient with the specified configuration
    ///
    /// Enables HTTP/3 (QUIC) optimization when config.http3_enabled is true.
    /// Provides comprehensive configuration of all HTTP protocol features
    /// including compression, timeouts, and advanced QUIC parameters.
    pub fn with_config(config: HttpConfig) -> Result<Self, HttpError> {
        let mut builder = crate::hyper::Client::builder()
            .pool_max_idle_per_host(config.pool_max_idle_per_host)
            .pool_idle_timeout(config.pool_idle_timeout)
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .tcp_nodelay(config.tcp_nodelay)
            .use_rustls_tls()
            .tls_built_in_root_certs(config.use_native_certs)
            .https_only(config.https_only)
            .user_agent(&config.user_agent);

        // HTTP/3 (QUIC) configuration when enabled
        if config.http3_enabled {
            Self::configure_http3(&mut builder, &config);
        } else {
            Self::configure_http2(&mut builder, &config);
        }

        // Compression configuration
        builder = builder.gzip(config.gzip).deflate(config.deflate);

        #[cfg(feature = "brotli")]
        {
            builder = builder.brotli(config.brotli);
        }

        // Build the client with error handling
        let inner = builder
            .build()
            .map_err(|e| HttpError::Configuration(format!("Failed to build HTTP client: {}", e)))?;

        Ok(Self::new(inner, config, ClientStats::default()))
    }

    /// Configure HTTP/3 (QUIC) protocol settings
    ///
    /// Applies HTTP/3 specific configuration including advanced QUIC parameters
    /// when the http3_unstable feature is enabled for maximum performance.
    #[inline]
    fn configure_http3(builder: &mut crate::hyper::ClientBuilder, config: &HttpConfig) {
        // Enable HTTP/3 protocol version preference
        *builder = std::mem::take(builder).http3_prior_knowledge();

        // Note: Advanced QUIC configuration methods require the 'http3_unstable' feature
        // which is not enabled by default. For full HTTP/3 optimization, enable this feature.
        #[cfg(feature = "http3_unstable")]
        {
            Self::configure_advanced_quic(builder, config);
        }

        #[cfg(not(feature = "http3_unstable"))]
        {
            // Basic HTTP/3 is enabled but advanced QUIC optimizations are not available
            // without the http3_unstable feature. HTTP/3 will still provide significant
            // performance improvements over HTTP/2 for most use cases.
            log::debug!(
                "HTTP/3 enabled but advanced QUIC optimizations require 'http3_unstable' feature"
            );
        }
    }

    /// Configure advanced QUIC parameters when unstable features are enabled
    ///
    /// Applies detailed QUIC configuration for maximum performance including
    /// congestion control, window sizes, and protocol-specific optimizations.
    #[cfg(feature = "http3_unstable")]
    #[inline]
    fn configure_advanced_quic(builder: &mut crate::hyper::ClientBuilder, config: &HttpConfig) {
        // Configure QUIC connection parameters when unstable features are enabled
        if let Some(idle_timeout) = config.quic_max_idle_timeout {
            *builder = std::mem::take(builder).http3_max_idle_timeout(idle_timeout);
        }

        if let Some(stream_window) = config.quic_stream_receive_window {
            *builder = std::mem::take(builder).http3_stream_receive_window(stream_window as u64);
        }

        if let Some(conn_window) = config.quic_receive_window {
            *builder = std::mem::take(builder).http3_conn_receive_window(conn_window as u64);
        }

        if let Some(send_window) = config.quic_send_window {
            *builder = std::mem::take(builder).http3_send_window(send_window);
        }

        // Enable BBR congestion control if requested
        if config.quic_congestion_bbr {
            *builder = std::mem::take(builder).http3_congestion_bbr();
        }

        // Configure HTTP/3 protocol parameters
        if let Some(header_size) = config.h3_max_field_section_size {
            *builder = std::mem::take(builder).http3_max_field_section_size(header_size);
        }

        if config.h3_enable_grease {
            *builder = std::mem::take(builder).http3_send_grease(true);
        }
    }

    /// Configure HTTP/2 protocol settings when HTTP/3 is disabled
    ///
    /// Applies HTTP/2 specific optimizations including adaptive windowing
    /// and frame size configuration for optimal HTTP/2 performance.
    #[inline]
    fn configure_http2(builder: &mut crate::hyper::ClientBuilder, config: &HttpConfig) {
        *builder = std::mem::take(builder)
            .http2_prior_knowledge()
            .http2_adaptive_window(config.http2_adaptive_window);

        if let Some(frame_size) = config.http2_max_frame_size {
            *builder = std::mem::take(builder).http2_max_frame_size(frame_size);
        }
    }
}

impl Default for HttpClient {
    /// Create HttpClient with default configuration
    ///
    /// Uses the default HttpConfig and falls back to a basic http3 client
    /// if configuration fails. This ensures the client can always be constructed
    /// even in constrained environments.
    fn default() -> Self {
        // Use the new with_config constructor with default configuration
        // If configuration fails, fall back to a basic http3 client
        let config = HttpConfig::default();
        Self::with_config(config).unwrap_or_else(|_| {
            Self::new(
                crate::hyper::Client::new(),
                HttpConfig::default(),
                ClientStats::default(),
            )
        })
    }
}
