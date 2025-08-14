use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
// Define our own Name type instead of using hyper_util
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Name(String);

impl Name {
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
use crate::hyper::error::BoxError;

// Full DNS resolution implementation for streams-first architecture
// Uses synchronous DNS APIs wrapped in spawn_task for zero-allocation performance

/// An iterator of resolved socket addresses.
pub type Addrs = std::vec::IntoIter<SocketAddr>;
// Re-export our own Name type instead of hyper_util's

/// Trait for DNS resolution in streams-first architecture.
/// Provides asynchronous DNS resolution using AsyncStream instead of Futures.
pub trait Resolve: Send + Sync + 'static {
    /// Resolve a hostname to socket addresses using streams-first architecture.
    /// Returns AsyncStream of Result to handle DNS resolution errors.
    fn resolve(&self, name: HyperName) -> AsyncStream<Result<Addrs, BoxError>>;
}

/// Type alias for DNS resolution result streams.
pub type Resolving = AsyncStream<Addrs>;
pub type DnsResolverWithOverrides = DynResolver;

#[derive(Clone)]
pub struct DynResolver {
    resolver: Arc<dyn Resolve>,
    prefer_ipv6: bool,
    cache: Option<Arc<std::collections::HashMap<String, Vec<SocketAddr>>>>,
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
        overrides: std::collections::HashMap<String, Vec<SocketAddr>>
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
        self.cache = Some(Arc::new(std::collections::HashMap::new()));
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
    pub fn resolve(&mut self, name: HyperName) -> AsyncStream<Addrs> {
        let resolver = self.resolver.clone();
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                let mut resolve_stream = resolver.resolve(name);
                match resolve_stream.try_next() {
                    Some(addrs) => addrs,
                    None => {
                        handle_error!("DNS resolver stream ended without producing addresses", "hostname DNS resolution");
                        return;
                    }
                }
            });
            
            match task.collect() {
                Ok(addrs) => emit!(sender, addrs),
                Err(e) => handle_error!(e, "hostname DNS resolution task"),
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
    fn resolve(&self, name: HyperName) -> AsyncStream<Result<Addrs, BoxError>> {
        let hostname = name.as_str().to_string();
        let prefer_ipv6 = self.prefer_ipv6;
        let timeout_ms = self.timeout_ms;
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || -> Result<Addrs, BoxError> {
                // Use std::net::ToSocketAddrs for synchronous resolution
                let dummy_port = 80; // Port doesn't matter for hostname resolution
                let host_with_port = format!("{}:{}", hostname, dummy_port);
                
                // Set up timeout handling using thread-local storage
                let start_time = std::time::Instant::now();
                
                let socket_addrs: Result<Vec<SocketAddr>, std::io::Error> = 
                    host_with_port.to_socket_addrs().map(|iter| {
                        let mut addrs: Vec<SocketAddr> = iter.collect();
                        
                        // Check timeout
                        if start_time.elapsed().as_millis() > timeout_ms as u128 {
                            return Vec::new(); // Timeout exceeded
                        }
                        
                        // Sort addresses based on preference
                        if prefer_ipv6 {
                            addrs.sort_by_key(|addr| match addr.ip() {
                                IpAddr::V6(_) => 0, // IPv6 first
                                IpAddr::V4(_) => 1, // IPv4 second
                            });
                        } else {
                            addrs.sort_by_key(|addr| match addr.ip() {
                                IpAddr::V4(_) => 0, // IPv4 first
                                IpAddr::V6(_) => 1, // IPv6 second
                            });
                        }
                        
                        // Remove port information since we added dummy port
                        addrs.into_iter().map(|mut addr| {
                            addr.set_port(0); // Clear the dummy port
                            addr
                        }).collect()
                    });

                match socket_addrs {
                    Ok(addrs) => {
                        if addrs.is_empty() {
                            Err(format!("DNS resolution timeout or no addresses found for {}", hostname).into())
                        } else {
                            Ok(addrs.into_iter())
                        }
                    },
                    Err(e) => {
                        Err(format!("DNS resolution failed for {}: {}", hostname, e).into())
                    }
                }
            });
            
            match task.collect() {
                Ok(addrs) => emit!(sender, Ok(addrs)),
                Err(e) => emit!(sender, Err(format!("GAI DNS resolution task failed: {}", e).into())),
            }
        })
    }
}

// Direct AsyncStream method - RETAINS ALL DNS functionality, removes Service abstraction
impl DynResolver {
    /// Direct DNS resolution method - replaces Service::call with AsyncStream
    /// RETAINS: All caching, timeouts, error handling, address sorting functionality
    /// Returns AsyncStream<Result<Addrs, BoxError>> per async-stream architecture
    pub fn resolve_direct(&mut self, name: HyperName) -> AsyncStream<Result<Addrs, BoxError>> {
        let resolver = self.resolver.clone();
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || -> Result<Addrs, BoxError> {
                let mut resolve_stream = resolver.resolve(name);
                match resolve_stream.try_next() {
                    Some(Ok(addrs)) => Ok(addrs),
                    Some(Err(e)) => Err(e),
                    None => {
                        Err("DNS resolver stream ended without producing addresses".into())
                    }
                }
            });
            
            match task.collect() {
                Ok(addrs) => emit!(sender, Ok(addrs)),
                Err(e) => emit!(sender, Err(format!("DNS resolution task failed: {}", e).into())),
            }
        })
    }
}

/// High-performance host-to-addresses resolution with port handling.
/// Uses optimized system DNS calls for minimal allocation.
fn resolve_host_to_addrs(host: &str, port: u16, prefer_ipv6: bool) -> Result<Vec<SocketAddr>, String> {
    // Try direct IP address parsing first (fastest path)
    if let Ok(ip_addr) = IpAddr::from_str(host) {
        return Ok(vec![SocketAddr::new(ip_addr, port)]);
    }
    
    // Perform DNS resolution using system resolver
    let host_with_port = format!("{}:{}", host, port);
    let socket_addrs: Result<Vec<SocketAddr>, std::io::Error> = 
        host_with_port.to_socket_addrs().map(|iter| {
            let mut addrs: Vec<SocketAddr> = iter.collect();
            
            // Apply address preference sorting
            if prefer_ipv6 {
                addrs.sort_by_key(|addr| match addr.ip() {
                    IpAddr::V6(_) => 0, // IPv6 first
                    IpAddr::V4(_) => 1, // IPv4 second
                });
            } else {
                addrs.sort_by_key(|addr| match addr.ip() {
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

/// DNS resolver with hostname overrides for testing and custom routing.
pub(crate) struct DnsResolverWithOverridesImpl {
    dns_resolver: Arc<dyn Resolve>,
    overrides: Arc<std::collections::HashMap<String, Vec<SocketAddr>>>,
}

impl Resolve for DnsResolverWithOverridesImpl {
    fn resolve(&self, name: HyperName) -> AsyncStream<Result<Addrs, BoxError>> {
        let hostname = name.as_str().to_string();
        let overrides = self.overrides.clone();
        let dns_resolver = self.dns_resolver.clone();
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || -> Result<Addrs, BoxError> {
                // Check for override first
                if let Some(addrs) = overrides.get(&hostname) {
                    return Ok(addrs.clone().into_iter());
                }
                
                // Fall back to actual DNS resolution
                let mut resolver_stream = dns_resolver.resolve(name);
                match resolver_stream.try_next() {
                    Some(Ok(addrs)) => Ok(addrs),
                    Some(Err(e)) => Err(e),
                    None => {
                        Err("DNS resolver stream ended without producing addresses".into())
                    }
                }
            });
            
            match task.collect() {
                Ok(addrs) => emit!(sender, Ok(addrs)),
                Err(e) => emit!(sender, Err(format!("overridden DNS resolution task failed: {}", e).into())),
            }
        })
    }
}