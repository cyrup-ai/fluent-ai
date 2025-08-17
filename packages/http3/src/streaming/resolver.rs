//! DNS Resolution for HTTP/3 Client
//!
//! Production DNS resolution with hickory-dns integration and fallback to std::net.

use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

use fluent_ai_async::{AsyncStream, emit, prelude::MessageChunk};
use url::Url;

use crate::streaming::chunks::HttpChunk;
use crate::types::{HttpVersion, TimeoutConfig};

/// DNS resolution result
#[derive(Debug, Clone)]
pub struct ResolvedAddress {
    pub ip: IpAddr,
    pub port: u16,
    pub hostname: Arc<str>,
}

impl MessageChunk for ResolvedAddress {
    #[inline]
    fn bad_chunk(error: String) -> Self {
        ResolvedAddress {
            ip: IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED),
            port: 0,
            hostname: Arc::from(error.as_str()),
        }
    }

    #[inline]
    fn error(&self) -> Option<&str> {
        if self.port == 0 {
            Some(self.hostname.as_ref())
        } else {
            None
        }
    }

    #[inline]
    fn is_error(&self) -> bool {
        self.port == 0
    }
}

impl Default for ResolvedAddress {
    #[inline]
    fn default() -> Self {
        ResolvedAddress {
            ip: IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            port: 443,
            hostname: Arc::from("localhost"),
        }
    }
}

/// DNS resolver with multiple backend support
#[derive(Debug, Clone)]
pub struct DnsResolver {
    pub timeout: Duration,
    pub use_hickory: bool,
}

impl Default for DnsResolver {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            use_hickory: cfg!(feature = "hickory-dns"),
        }
    }
}

impl DnsResolver {
    /// Create new resolver with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set DNS resolution timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable hickory-dns resolver
    pub fn with_hickory(mut self) -> Self {
        self.use_hickory = true;
        self
    }

    /// Resolve hostname to socket addresses
    pub fn resolve(&self, url: &Url) -> AsyncStream<ResolvedAddress, 64> {
        let hostname = url.host_str().unwrap_or("localhost").to_string();
        let port = url.port_or_known_default().unwrap_or(443);
        let timeout = self.timeout;
        let use_hickory = self.use_hickory;

        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                // Try hickory-dns first if enabled
                if use_hickory {
                    if let Ok(addresses) = resolve_with_hickory(&hostname, port, timeout) {
                        for addr in addresses {
                            let resolved = ResolvedAddress {
                                ip: addr.ip(),
                                port: addr.port(),
                                hostname: Arc::from(hostname.as_str()),
                            };
                            emit!(sender, resolved);
                        }
                        return;
                    }
                }

                // Fallback to std::net resolution
                match resolve_with_std(&hostname, port) {
                    Ok(addresses) => {
                        for addr in addresses {
                            let resolved = ResolvedAddress {
                                ip: addr.ip(),
                                port: addr.port(),
                                hostname: Arc::from(hostname.as_str()),
                            };
                            emit!(sender, resolved);
                        }
                    }
                    Err(e) => {
                        tracing::error!("DNS resolution failed for {}: {}", hostname, e);
                        let error_addr =
                            ResolvedAddress::bad_chunk(format!("DNS resolution failed: {}", e));
                        emit!(sender, error_addr);
                    }
                }
            });
        })
    }

    /// Resolve URL and return first available address
    pub fn resolve_first(&self, url: &Url) -> Option<SocketAddr> {
        let hostname = url.host_str()?;
        let port = url.port_or_known_default().unwrap_or(443);

        // Try std::net first for synchronous resolution
        if let Ok(mut addrs) = format!("{}:{}", hostname, port).to_socket_addrs() {
            return addrs.next();
        }

        None
    }
}

/// Resolve using standard library
fn resolve_with_std(hostname: &str, port: u16) -> Result<Vec<SocketAddr>, std::io::Error> {
    let addr_str = format!("{}:{}", hostname, port);
    let addrs: Vec<SocketAddr> = addr_str.to_socket_addrs()?.collect();
    Ok(addrs)
}

/// Resolve using hickory-dns (async DNS resolver)
fn resolve_with_hickory(
    hostname: &str,
    port: u16,
    timeout: Duration,
) -> Result<Vec<SocketAddr>, Box<dyn std::error::Error + Send + Sync>> {
    use hickory_resolver::Resolver;
    use hickory_resolver::config::*;

    let resolver = Resolver::new(ResolverConfig::default(), ResolverOpts::default())?;

    // Perform A and AAAA lookups
    let mut addresses = Vec::new();

    // IPv4 lookup
    if let Ok(response) = resolver.lookup_ip(hostname) {
        for ip in response.iter() {
            addresses.push(SocketAddr::new(ip, port));
        }
    }

    if addresses.is_empty() {
        return Err(format!("No addresses found for {}", hostname).into());
    }

    Ok(addresses)
}

#[cfg(not(feature = "hickory-dns"))]
fn resolve_with_hickory(
    _hostname: &str,
    _port: u16,
    _timeout: Duration,
) -> Result<Vec<SocketAddr>, Box<dyn std::error::Error + Send + Sync>> {
    Err("hickory-dns feature not enabled".into())
}

/// Validate hostname format
pub fn validate_hostname(hostname: &str) -> bool {
    if hostname.is_empty() || hostname.len() > 253 {
        return false;
    }

    // Basic hostname validation
    hostname
        .chars()
        .all(|c| c.is_alphanumeric() || c == '.' || c == '-')
        && !hostname.starts_with('-')
        && !hostname.ends_with('-')
}

/// Extract hostname from URL
pub fn extract_hostname(url: &Url) -> Option<&str> {
    url.host_str()
}

/// Get default port for scheme
pub fn default_port_for_scheme(scheme: &str) -> u16 {
    match scheme {
        "http" => 80,
        "https" => 443,
        "ftp" => 21,
        "ssh" => 22,
        _ => 80,
    }
}
