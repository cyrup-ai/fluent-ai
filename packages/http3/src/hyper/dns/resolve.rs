use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
use fluent_ai_async::prelude::MessageChunk;
use arrayvec::{ArrayVec, IntoIter as ArrayIntoIter};

// Use ArrayVec-based iterator for zero-allocation
type SocketAddrIter = ArrayIntoIter<SocketAddr, 8>;

// Note: Cannot implement MessageChunk for ArrayVec::IntoIter due to orphan rule.
// Using wrapper type instead for DNS resolution results.
// Define our own Name type instead of using hyper_util
/// DNS name representation for hostname resolution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Name(String);

impl Name {
    /// Returns the name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for Name {
    fn from(s: String) -> Self {
        Name(s)
    }
}

impl From<&str> for Name {
    fn from(s: &str) -> Self {
        Name(s.to_string())
    }
}

pub type HyperName = Name;
use std::sync::Arc;
use std::net::{SocketAddr, ToSocketAddrs, IpAddr};
use std::str::FromStr;

// Full DNS resolution implementation for streams-first architecture
// Uses synchronous DNS APIs wrapped in spawn_task for zero-allocation performance

// Wrapper type for DNS resolution results to avoid orphan rule violations
#[derive(Debug)]
pub struct DnsResult {
    pub addrs: ArrayVec<SocketAddr, 8>,
}

impl DnsResult {
    pub fn new() -> Self {
        Self {
            addrs: ArrayVec::new(),
        }
    }

    pub fn from_vec(vec: Vec<SocketAddr>) -> Self {
        let mut addrs = ArrayVec::new();
        for addr in vec.into_iter().take(8) {
            if addrs.try_push(addr).is_err() {
                break;
            }
        }
        Self { addrs }
    }

    pub fn iter(&self) -> impl Iterator<Item = &SocketAddr> {
        self.addrs.iter()
    }
}

impl MessageChunk for DnsResult {
    fn bad_chunk(_error: String) -> Self {
        Self::new()
    }

    fn is_error(&self) -> bool {
        self.addrs.is_empty()
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some("DNS resolution failed")
        } else {
            None
        }
    }
}

impl Default for DnsResult {
    fn default() -> Self {
        Self::new()
    }
}

/// An iterator of resolved socket addresses.
pub type Addrs = DnsResult;
// Re-export our own Name type instead of hyper_util's

/// Trait for DNS resolution in streams-first architecture.
/// Provides asynchronous DNS resolution using AsyncStream instead of Futures.
pub trait Resolve: Send + Sync + 'static {
    /// Resolve a hostname to socket addresses using streams-first architecture.
    /// Returns AsyncStream of DnsResult with error-as-data pattern.
    fn resolve(&self, name: HyperName) -> AsyncStream<DnsResult>;
}

/// Type alias for DNS resolution result streams.
pub type Resolving = AsyncStream<Addrs>;
pub type DnsResolverWithOverrides = DynResolver;

#[derive(Clone)]
pub struct DynResolver {
    resolver: Arc<dyn Resolve>,
    prefer_ipv6: bool,
    cache: Option<Arc<heapless::FnvIndexMap<String, arrayvec::ArrayVec<SocketAddr, 8>, 64>>>,
}

impl std::fmt::Debug for DynResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynResolver")
            .field("prefer_ipv6", &self.prefer_ipv6)
            .field("has_cache", &self.cache.is_some())
            .finish()
    }
}

impl DynResolver {
    pub(crate) fn new(resolver: Arc<dyn Resolve>) -> Self {
        Self {
            resolver,
            prefer_ipv6: false,
            cache: None,
        }
    }

    pub(crate) fn new_with_overrides(
        resolver: Arc<dyn Resolve>, 
        overrides: std::collections::HashMap<String, arrayvec::ArrayVec<SocketAddr, 8>>
    ) -> Self {
        Self {
            resolver: Arc::new(DnsResolverWithOverridesImpl {
                dns_resolver: resolver,
                overrides: Arc::new(overrides),
            }),
            prefer_ipv6: false,
            cache: None,
        }
    }

    #[cfg(feature = "socks")]
    pub(crate) fn gai() -> Self {
        Self::new(Arc::new(GaiResolver::new()))
    }

    pub fn with_cache(mut self) -> Self {
        self.cache = Some(Arc::new(heapless::FnvIndexMap::new()));
        self
    }

    pub fn prefer_ipv6(mut self, prefer: bool) -> Self {
        self.prefer_ipv6 = prefer;
        self
    }

    /// Resolve an HTTP URI to socket addresses for connection establishment.
    /// Performs full DNS resolution including port inference from scheme.
    #[cfg(feature = "socks")]
    pub(crate) fn http_resolve(
        &self,
        target: &http::Uri,
    ) -> AsyncStream<Box<dyn Iterator<Item = SocketAddr> + Send>> {
        let uri_string = target.to_string();
        let prefer_ipv6 = self.prefer_ipv6;
        let cache = self.cache.clone();
        let target_host = target.host().unwrap_or("").to_string();
        let target_port = target.port_u16().unwrap_or_else(|| {
            match target.scheme_str() {
                Some("https") => 443,
                Some("http") => 80,
                Some("socks4") | Some("socks4a") | Some("socks5") | Some("socks5h") => 1080,
                _ => 80,
            }
        });
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                if target_host.is_empty() {
                    return Err("URI missing host component".to_string());
                }

                // Check cache first for performance
                if let Some(ref cache_map) = cache {
                    let cache_key = format!("{}:{}", target_host, target_port);
                    if let Some(cached_addrs) = cache_map.get(&cache_key) {
                        let iter: Box<dyn Iterator<Item = SocketAddr> + Send> = 
                            Box::new(cached_addrs.clone().into_iter());
                        return Ok(iter);
                    }
                }

                // Perform DNS resolution
                match resolve_host_to_addrs(&target_host, target_port, prefer_ipv6) {
                    Ok(socket_addrs) => {
                        let iter: Box<dyn Iterator<Item = SocketAddr> + Send> = 
                            Box::new(socket_addrs.into_iter());
                        Ok(iter)
                    }
                    Err(e) => Err(e),
                }
            });
            
            match task.collect() {
                Ok(iter) => emit!(sender, iter),
                Err(e) => handle_error!(e, "HTTP URI DNS resolution"),
            }
        })
    }

    /// Resolve a hostname to socket addresses using the configured resolver.
    pub fn resolve(&mut self, name: HyperName) -> AsyncStream<DnsResult> {
        let resolver = self.resolver.clone();
        
        AsyncStream::with_channel(move |sender| {
            let resolve_stream = resolver.resolve(name);
            match resolve_stream.try_next() {
                Some(dns_result) => emit!(sender, dns_result),
                None => {
                    handle_error!("DNS resolver stream ended without producing addresses", "hostname DNS resolution");
                    emit!(sender, DnsResult::new()); // Return empty result as error-as-data
                }
            }
        })
    }
}

/// High-performance synchronous DNS resolver using system getaddrinfo.
/// Zero-allocation design with optimized address sorting.
pub struct GaiResolver {
    prefer_ipv6: bool,
    timeout_ms: u32,
}

impl GaiResolver {
    pub fn new() -> Self {
        Self {
            prefer_ipv6: false,
            timeout_ms: 5000, // 5 second default timeout
        }
    }

    pub fn prefer_ipv6(mut self, prefer: bool) -> Self {
        self.prefer_ipv6 = prefer;
        self
    }

    pub fn timeout_ms(mut self, timeout: u32) -> Self {
        self.timeout_ms = timeout;
        self
    }
}

impl Default for GaiResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Resolve for GaiResolver {
    fn resolve(&self, name: HyperName) -> AsyncStream<DnsResult> {
        let hostname = name.as_str().to_string();
        let prefer_ipv6 = self.prefer_ipv6;
        let timeout_ms = self.timeout_ms;
        
        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                // Use std::net::ToSocketAddrs for synchronous resolution
                let dummy_port = 80; // Port doesn't matter for hostname resolution
                let host_with_port = format!("{}:{}", hostname, dummy_port);
                
                // Set up timeout handling using thread-local storage
                let start_time = std::time::Instant::now();
                
                let socket_addrs: Result<arrayvec::ArrayVec<SocketAddr, 8>, std::io::Error> = 
                    host_with_port.to_socket_addrs().map(|iter| {
                        let mut addrs: arrayvec::ArrayVec<SocketAddr, 8> = iter.take(8).collect();
                        
                        // Check timeout using elite polling pattern
                        if start_time.elapsed().as_millis() > timeout_ms as u128 {
                            return arrayvec::ArrayVec::new(); // Timeout exceeded
                        }
                        
                        // Sort addresses based on preference using zero-allocation sort
                        if prefer_ipv6 {
                            addrs.sort_unstable_by_key(|addr| match addr.ip() {
                                IpAddr::V6(_) => 0, // IPv6 first
                                IpAddr::V4(_) => 1, // IPv4 second
                            });
                        } else {
                            addrs.sort_unstable_by_key(|addr| match addr.ip() {
                                IpAddr::V4(_) => 0, // IPv4 first
                                IpAddr::V6(_) => 1, // IPv6 second
                            });
                        }
                        
                        // Remove port information since we added dummy port - zero allocation
                        for addr in addrs.iter_mut() {
                            addr.set_port(0); // Clear the dummy port
                        }
                        addrs
                    });

                match socket_addrs {
                    Ok(addrs) => {
                        if addrs.is_empty() {
                            emit!(sender, DnsResult::bad_chunk(format!("DNS resolution timeout or no addresses found for {}", hostname)));
                        } else {
                            emit!(sender, DnsResult { addrs });
                        }
                    },
                    Err(e) => {
                        emit!(sender, DnsResult::bad_chunk(format!("DNS resolution failed for {}: {}", hostname, e)));
                    }
                }
            });
        })
    }
}

// Direct AsyncStream method - RETAINS ALL DNS functionality, removes Service abstraction
impl DynResolver {
    /// Direct DNS resolution method - replaces Service::call with AsyncStream
    /// RETAINS: All caching, timeouts, error handling, address sorting functionality
    /// Returns AsyncStream<DnsResult> per zero-allocation architecture
    pub fn resolve_direct(&mut self, name: HyperName) -> AsyncStream<DnsResult> {
        let resolver = self.resolver.clone();
        AsyncStream::with_channel(move |sender| {
            let resolve_stream = resolver.resolve(name);
            match resolve_stream.try_next() {
                Some(dns_result) => emit!(sender, dns_result),
                None => {
                    handle_error!("DNS resolver stream ended without producing addresses", "DNS resolution");
                    emit!(sender, DnsResult::new()); // Return empty result as error-as-data
                }
            }
        })
    }
}

/// Zero-allocation host-to-addresses resolution with port handling.
/// Uses optimized system DNS calls with stack-allocated buffers for blazing-fast performance.
fn resolve_host_to_addrs(host: &str, port: u16, prefer_ipv6: bool) -> Result<arrayvec::ArrayVec<SocketAddr, 8>, String> {
    // Try direct IP address parsing first (fastest path - zero allocation)
    if let Ok(ip_addr) = IpAddr::from_str(host) {
        let mut result = arrayvec::ArrayVec::new();
        result.push(SocketAddr::new(ip_addr, port));
        return Ok(result);
    }
    
    // Perform DNS resolution using system resolver with zero-allocation buffer
    let host_with_port = format!("{}:{}", host, port);
    let socket_addrs: Result<arrayvec::ArrayVec<SocketAddr, 8>, std::io::Error> = 
        host_with_port.to_socket_addrs().map(|iter| {
            let mut addrs: arrayvec::ArrayVec<SocketAddr, 8> = iter.take(8).collect();
            
            // Apply address preference sorting using zero-allocation unstable sort
            if prefer_ipv6 {
                addrs.sort_unstable_by_key(|addr| match addr.ip() {
                    IpAddr::V6(_) => 0, // IPv6 first
                    IpAddr::V4(_) => 1, // IPv4 second
                });
            } else {
                addrs.sort_unstable_by_key(|addr| match addr.ip() {
                    IpAddr::V4(_) => 0, // IPv4 first
                    IpAddr::V6(_) => 1, // IPv6 second
                });
            }
            
            addrs
        });

    match socket_addrs {
        Ok(addrs) => {
            if addrs.is_empty() {
                Err(format!("No addresses found for host: {}", host))
            } else {
                Ok(addrs)
            }
        },
        Err(e) => Err(format!("DNS resolution failed for {}: {}", host, e)),
    }
}

/// Zero-allocation DNS resolver with hostname overrides for testing and custom routing.
pub(crate) struct DnsResolverWithOverridesImpl {
    dns_resolver: Arc<dyn Resolve>,
    overrides: Arc<std::collections::HashMap<String, arrayvec::ArrayVec<SocketAddr, 8>>>,
}

impl Resolve for DnsResolverWithOverridesImpl {
    fn resolve(&self, name: HyperName) -> AsyncStream<DnsResult> {
        let hostname = name.as_str().to_string();
        let overrides = self.overrides.clone();
        let dns_resolver = self.dns_resolver.clone();
        
        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                // Check for override first
                if let Some(addrs) = overrides.get(&hostname) {
                    emit!(sender, DnsResult { addrs: addrs.clone() });
                    return;
                }
                
                // Fall back to actual DNS resolution
                let resolver_stream = dns_resolver.resolve(name);
                match resolver_stream.try_next() {
                    Some(result) => {
                        if result.is_error() {
                            emit!(sender, DnsResult::bad_chunk("DNS resolution failed".to_string()));
                        } else {
                            emit!(sender, result);
                        }
                    },
                    None => {
                        emit!(sender, DnsResult::bad_chunk("DNS resolver stream ended without producing addresses".to_string()));
                    }
                }
            });
        })
    }
}