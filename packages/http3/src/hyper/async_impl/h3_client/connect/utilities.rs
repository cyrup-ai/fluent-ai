//! Utility functions for H3 connection handling
//!
//! Helper functions for error handling, connection validation,
//! and retry logic for HTTP/3 connections.

use std::io::Error;

/// Check if a connection error is retryable
pub fn is_retryable_error(error: &Error) -> bool {
    match error.kind() {
        std::io::ErrorKind::ConnectionRefused
        | std::io::ErrorKind::ConnectionReset
        | std::io::ErrorKind::ConnectionAborted
        | std::io::ErrorKind::TimedOut
        | std::io::ErrorKind::Interrupted => true,
        _ => false,
    }
}

/// Validate server name for TLS connections
pub fn validate_server_name(server_name: &str) -> Result<&str, String> {
    if server_name.is_empty() {
        return Err("Server name cannot be empty".to_string());
    }

    // Basic validation - in production would be more comprehensive
    if server_name.contains(' ') {
        return Err("Server name cannot contain spaces".to_string());
    }

    Ok(server_name)
}

/// Calculate retry delay with exponential backoff
pub fn calculate_retry_delay(attempt: u32, base_delay_ms: u64) -> std::time::Duration {
    let delay_ms = base_delay_ms * 2_u64.pow(attempt.min(10)); // Cap at 2^10
    let max_delay_ms = 30_000; // 30 seconds max

    std::time::Duration::from_millis(delay_ms.min(max_delay_ms))
}

/// Check if address is valid for connection
pub fn is_valid_address(addr: &std::net::SocketAddr) -> bool {
    !addr.ip().is_unspecified() && addr.port() > 0
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    use super::*;

    #[test]
    fn test_retryable_errors() {
        let retryable = Error::new(std::io::ErrorKind::ConnectionRefused, "test");
        assert!(is_retryable_error(&retryable));

        let non_retryable = Error::new(std::io::ErrorKind::PermissionDenied, "test");
        assert!(!is_retryable_error(&non_retryable));
    }

    #[test]
    fn test_server_name_validation() {
        assert!(validate_server_name("example.com").is_ok());
        assert!(validate_server_name("").is_err());
        assert!(validate_server_name("invalid name").is_err());
    }

    #[test]
    fn test_retry_delay() {
        let delay1 = calculate_retry_delay(0, 100);
        let delay2 = calculate_retry_delay(1, 100);
        assert!(delay2 > delay1);

        let max_delay = calculate_retry_delay(20, 100);
        assert_eq!(max_delay, std::time::Duration::from_millis(30_000));
    }

    #[test]
    fn test_address_validation() {
        let valid_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        assert!(is_valid_address(&valid_addr));

        let invalid_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 0);
        assert!(!is_valid_address(&invalid_addr));
    }
}
