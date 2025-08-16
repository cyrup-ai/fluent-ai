//! HTTP client configuration module
//!
//! Provides comprehensive configuration management for HTTP clients including
//! type definitions, default values, and debug formatting capabilities.
//!
//! This module is organized into logical components:
//! - `types`: Core configuration structures and enums
//! - `defaults`: Default value implementations
//! - `debug`: Debug formatting for configuration inspection

pub mod debug;
pub mod defaults;
pub mod types;

// Re-export core types for public API
pub use types::{Config, HttpVersionPref};

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_config_default_creation() {
        let config = Config::default();

        // Verify key defaults are set correctly
        assert!(config.referer);
        assert!(config.nodelay);
        assert_eq!(config.pool_idle_timeout, Some(Duration::from_secs(90)));
        assert_eq!(config.tcp_keepalive, Some(Duration::from_secs(60)));
        assert!(config.auto_sys_proxy);
        assert!(!config.https_only);
        assert!(!config.http09_responses);
        assert!(!config.http1_title_case_headers);
        assert!(matches!(config.http_version_pref, HttpVersionPref::All));
    }

    #[test]
    fn test_config_headers_initialization() {
        let config = Config::default();

        // Verify default headers are set
        assert!(!config.headers.is_empty());
        assert!(config.headers.contains_key("accept"));
    }

    #[test]
    fn test_http_version_pref_variants() {
        // Test that all HttpVersionPref variants can be created
        let _http1 = HttpVersionPref::Http1;
        let _all = HttpVersionPref::All;

        #[cfg(feature = "http2")]
        let _http2 = HttpVersionPref::Http2;

        #[cfg(feature = "http3")]
        let _http3 = HttpVersionPref::Http3;
    }

    #[test]
    fn test_config_tls_defaults() {
        let config = Config::default();

        #[cfg(feature = "__tls")]
        {
            assert!(config.hostname_verification);
            assert!(config.certs_verification);
            assert!(config.tls_sni);
            assert!(config.tls_built_in_root_certs);
            assert!(!config.tls_info);
        }
    }

    #[test]
    fn test_config_http2_defaults() {
        let config = Config::default();

        #[cfg(feature = "http2")]
        {
            assert_eq!(config.http2_initial_stream_window_size, None);
            assert_eq!(config.http2_initial_connection_window_size, None);
            assert!(!config.http2_adaptive_window);
            assert_eq!(config.http2_max_frame_size, None);
            assert_eq!(config.http2_max_header_list_size, None);
            assert_eq!(config.http2_keep_alive_interval, None);
            assert_eq!(config.http2_keep_alive_timeout, None);
            assert!(!config.http2_keep_alive_while_idle);
        }
    }

    #[test]
    fn test_config_http3_defaults() {
        let config = Config::default();

        #[cfg(feature = "http3")]
        {
            assert!(!config.tls_enable_early_data);
            assert_eq!(config.quic_max_idle_timeout, None);
            assert_eq!(config.quic_stream_receive_window, None);
            assert_eq!(config.quic_receive_window, None);
            assert_eq!(config.quic_send_window, None);
            assert!(!config.quic_congestion_bbr);
            assert_eq!(config.h3_max_field_section_size, None);
            assert_eq!(config.h3_send_grease, None);
        }
    }

    #[test]
    fn test_config_dns_defaults() {
        let config = Config::default();

        assert!(config.dns_overrides.is_empty());
        assert!(config.dns_resolver.is_none());
        assert_eq!(config.hickory_dns, cfg!(feature = "hickory-dns"));
    }

    #[test]
    fn test_config_connection_defaults() {
        let config = Config::default();

        assert_eq!(config.connect_timeout, None);
        assert!(!config.connection_verbose);
        assert_eq!(config.pool_max_idle_per_host, usize::MAX);
        assert_eq!(config.tcp_keepalive_interval, None);
        assert_eq!(config.tcp_keepalive_retries, None);
        assert_eq!(config.read_timeout, None);
        assert_eq!(config.timeout, None);
        assert_eq!(config.local_address, None);
    }

    #[test]
    fn test_config_proxy_defaults() {
        let config = Config::default();

        assert!(config.proxies.is_empty());
        assert!(config.auto_sys_proxy);
    }

    #[test]
    fn test_config_error_state() {
        let config = Config::default();

        assert!(config.error.is_none());
    }
}
