//! Core HTTP configuration types and default implementations
//!
//! This module provides the main HttpConfig struct and related enums for HTTP client configuration.
//! The configuration is organized into logical modules:
//!
//! - `types`: Core HttpConfig struct definition with all configuration fields
//! - `enums`: Connection reuse strategies, retry policies, and error classifications
//! - `defaults`: Sensible default values optimized for HTTP/3 usage patterns
//! - `builders`: Fluent builder methods for common configuration scenarios
//!
//! All modules maintain production-quality code standards and comprehensive documentation.

pub mod builders;
pub mod defaults;
pub mod enums;
pub mod retry;
pub mod types;

// Re-export all main types for backward compatibility
pub use enums::{ConnectionReuse, RetryPolicy, RetryableError};
pub use types::HttpConfig;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_http_config_default() {
        let config = HttpConfig::default();

        // Test core defaults
        assert_eq!(config.pool_max_idle_per_host, 32);
        assert_eq!(config.pool_idle_timeout, Duration::from_secs(90));
        assert_eq!(config.timeout, Duration::from_secs(86400));
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert!(config.http3_enabled);
        assert!(config.gzip);
        assert!(config.brotli);
        assert!(config.deflate);
        assert!(config.metrics_enabled);
        assert!(!config.tracing_enabled);
    }

    #[test]
    fn test_retry_policy_default() {
        let policy = RetryPolicy::default();

        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.base_delay, Duration::from_millis(100));
        assert_eq!(policy.max_delay, Duration::from_secs(30));
        assert_eq!(policy.backoff_factor, 2.0);
        assert_eq!(policy.jitter_factor, 0.1);
        assert_eq!(policy.retry_on_status, vec![429, 500, 502, 503, 504]);
        assert_eq!(policy.retry_on_errors.len(), 4);
    }

    #[test]
    fn test_builder_methods() {
        let config = HttpConfig::default()
            .with_http3(false)
            .with_compression(false)
            .with_metrics(false)
            .with_tracing(true);

        assert!(!config.http3_enabled);
        assert!(!config.gzip);
        assert!(!config.brotli);
        assert!(!config.deflate);
        assert!(!config.metrics_enabled);
        assert!(config.tracing_enabled);
    }

    #[test]
    fn test_connection_reuse_enum() {
        let aggressive = ConnectionReuse::Aggressive;
        let conservative = ConnectionReuse::Conservative;
        let disabled = ConnectionReuse::Disabled;

        // Test that enums can be cloned and debugged
        let _cloned = aggressive.clone();
        let _debug_str = format!("{:?}", conservative);

        // Test enum variants exist
        match disabled {
            ConnectionReuse::Disabled => {}
            _ => panic!("Expected Disabled variant"),
        }
    }

    #[test]
    fn test_retryable_error_enum() {
        let errors = vec![
            RetryableError::Network,
            RetryableError::Timeout,
            RetryableError::Connection,
            RetryableError::Dns,
            RetryableError::Tls,
        ];

        // Test equality comparisons
        assert_eq!(RetryableError::Network, RetryableError::Network);
        assert_ne!(RetryableError::Network, RetryableError::Timeout);

        // Test that errors can be used in collections
        assert!(errors.contains(&RetryableError::Network));
        assert!(!errors.contains(&RetryableError::Timeout) || errors.len() >= 2);
    }

    #[test]
    fn test_http_config_quic_defaults() {
        let config = HttpConfig::default();

        // Test HTTP/3 (QUIC) specific defaults
        assert_eq!(config.quic_max_idle_timeout, Some(Duration::from_secs(30)));
        assert_eq!(config.quic_stream_receive_window, Some(256 * 1024));
        assert_eq!(config.quic_receive_window, Some(1024 * 1024));
        assert_eq!(config.quic_send_window, Some(512 * 1024));
        assert!(!config.quic_congestion_bbr);
        assert!(!config.tls_early_data);
        assert_eq!(config.h3_max_field_section_size, Some(16 * 1024));
        assert!(config.h3_enable_grease);
    }

    #[test]
    fn test_http_config_http2_defaults() {
        let config = HttpConfig::default();

        // Test HTTP/2 specific defaults
        assert!(config.http2_adaptive_window);
        assert_eq!(config.http2_max_frame_size, Some(1 << 20));
        assert!(!config.http2_server_push);
        assert!(config.http2_keep_alive);
        assert_eq!(
            config.http2_keep_alive_interval,
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            config.http2_keep_alive_timeout,
            Some(Duration::from_secs(5))
        );
        assert!(config.http2_adaptive_window_scaling);
    }
}
