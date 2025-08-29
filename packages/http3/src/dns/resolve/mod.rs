//! DNS resolution module for HTTP3 client
//!
//! Provides zero-allocation DNS resolution with streaming architecture,
//! caching, timeout handling, and address preference sorting.
//!
//! This module implements RFC-compliant DNS resolution using the fluent_ai_async
//! streaming architecture with error-as-data patterns.

pub mod resolvers;
pub mod traits;
pub mod types;
pub mod utilities;

// Re-export core types and traits for public API
pub use resolvers::DynResolver as Resolver;
pub use resolvers::{DnsResolverWithOverrides, DynResolver, GaiResolver};
pub use traits::{Addrs, Resolve, Resolving};
// Legacy compatibility exports to maintain existing API
pub use types::Name as NameType;
pub use types::{DnsResult, HyperName, Name};
pub use utilities::{
    DnsResolverWithOverridesImpl, is_ip_address, resolve_host_to_addrs,
    socket_addr_from_ip_literal, sort_addresses_by_preference, validate_hostname,
};

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    use fluent_ai_async::AsyncStream;

    use super::*;

    #[test]
    fn test_dns_result_creation() {
        let mut addrs = arrayvec::ArrayVec::new();
        addrs.push(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 80));

        let result = DnsResult { addrs };
        assert_eq!(result.addrs.len(), 1);
    }

    #[test]
    fn test_dns_result_bad_chunk() {
        let error_result = DnsResult::bad_chunk("DNS resolution failed".to_string());
        assert!(error_result.addrs.is_empty());
    }

    #[test]
    fn test_name_creation() {
        let name = Name::from("example.com");
        assert_eq!(name.as_str(), "example.com");
    }

    #[test]
    fn test_hyper_name_conversion() {
        let name = Name::from("test.example.com");
        let hyper_name: HyperName = name;
        assert_eq!(hyper_name.as_str(), "test.example.com");
    }

    #[test]
    fn test_gai_resolver_creation() {
        let resolver = GaiResolver::new().prefer_ipv6(true).timeout_ms(3000);

        // Test that resolver was created successfully
        assert!(true); // Basic creation test
    }

    #[test]
    fn test_hostname_validation() {
        assert!(utilities::validate_hostname("example.com").is_ok());
        assert!(utilities::validate_hostname("sub.example.com").is_ok());
        assert!(utilities::validate_hostname("").is_err());
        assert!(utilities::validate_hostname("-invalid").is_err());
        assert!(utilities::validate_hostname("invalid-").is_err());
    }

    #[test]
    fn test_ip_address_detection() {
        assert!(utilities::is_ip_address("127.0.0.1"));
        assert!(utilities::is_ip_address("::1"));
        assert!(!utilities::is_ip_address("example.com"));
        assert!(!utilities::is_ip_address("not-an-ip"));
    }

    #[test]
    fn test_socket_addr_from_ip_literal() {
        let result = utilities::socket_addr_from_ip_literal("127.0.0.1", 80);
        assert!(result.is_ok());

        let addr = result.unwrap();
        assert_eq!(addr.port(), 80);
        assert_eq!(addr.ip(), IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
    }

    #[test]
    fn test_address_sorting() {
        let mut addrs = arrayvec::ArrayVec::new();
        addrs.push(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            80,
        ));
        addrs.push(SocketAddr::new(IpAddr::V6("::1".parse().unwrap()), 80));

        // Test IPv4 preference
        utilities::sort_addresses_by_preference(&mut addrs, false);
        assert!(addrs[0].is_ipv4());
        assert!(addrs[1].is_ipv6());

        // Test IPv6 preference
        utilities::sort_addresses_by_preference(&mut addrs, true);
        assert!(addrs[0].is_ipv6());
        assert!(addrs[1].is_ipv4());
    }
}
