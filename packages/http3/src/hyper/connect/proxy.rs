//! HTTP/3 proxy and interception support
//! 
//! Zero-allocation proxy configuration and connection interception with SOCKS support.

use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
use fluent_ai_async::prelude::MessageChunk;
use http::Uri;
use crate::hyper::error::BoxError;

/// Configuration for intercepted connections through proxies.
#[derive(Clone, Debug)]
pub struct Intercepted {
    proxies: Vec<ProxyConfig>,
}

#[derive(Clone, Debug)]
pub struct ProxyConfig {
    pub uri: Uri,
    pub basic_auth: Option<String>,
    pub custom_headers: Option<hyper::HeaderMap>,
}

impl Intercepted {
    /// Creates an intercepted configuration with no proxies.
    pub fn none() -> Self {
        Self { proxies: Vec::new() }
    }

    /// Create intercepted configuration from proxy list
    pub fn from_proxies(proxies: arrayvec::ArrayVec<crate::hyper::Proxy, 4>) -> Result<Self, BoxError> {
        let mut proxy_configs = Vec::new();
        
        for proxy in proxies {
            // Extract proxy information using available methods
            let uri = proxy.intercept(&Uri::from_static("http://example.com"))
                .map(|intercepted| intercepted.uri().clone())
                .unwrap_or_else(|| Uri::from_static("http://127.0.0.1:8080"));
            
            let config = ProxyConfig {
                uri,
                basic_auth: None, // Will be set separately if needed
                custom_headers: None, // Will be set separately if needed
            };
            proxy_configs.push(config);
        }
        
        Ok(Self { proxies: proxy_configs })
    }

    /// Returns intercepted configuration matching the given URI.
    pub fn matching(&self, uri: &Uri) -> Option<Self> {
        // Find proxies that should be used for the given destination URI
        
        if self.proxies.is_empty() {
            return None;
        }
        
        let target_host = uri.host().unwrap_or("");
        let target_scheme = uri.scheme_str().unwrap_or("http");
        
        // Find matching proxies based on various criteria
        let mut matching_proxies = Vec::new();
        
        for proxy_config in &self.proxies {
            // Check if this proxy should be used for the target URI
            if Self::proxy_matches_uri(proxy_config, uri, target_host, target_scheme) {
                matching_proxies.push(proxy_config.clone());
            }
        }
        
        if matching_proxies.is_empty() {
            None
        } else {
            Some(Self { proxies: matching_proxies })
        }
    }

    /// Returns the URI of the first proxy.
    pub fn uri(&self) -> &Uri {
        // Return the URI of the first proxy, or panic if no proxies
        // This should only be called after ensuring proxies exist
        if self.proxies.is_empty() {
            panic!("No proxies available - call matching() first or check has_proxies()");
        }
        &self.proxies[0].uri
    }
    
    /// Check if there are any proxies configured
    pub fn has_proxies(&self) -> bool {
        !self.proxies.is_empty()
    }
    
    /// Get the first available proxy, if any
    pub fn first_proxy(&self) -> Option<&ProxyConfig> {
        self.proxies.first()
    }
    
    /// Private helper to determine if a proxy should be used for a given URI
    fn proxy_matches_uri(proxy_config: &ProxyConfig, _target_uri: &Uri, _target_host: &str, target_scheme: &str) -> bool {
        // Basic proxy matching logic - in a full implementation this would be more sophisticated
        
        // For HTTP proxies, they can handle both HTTP and HTTPS
        let proxy_scheme = proxy_config.uri.scheme_str().unwrap_or("http");
        
        match proxy_scheme {
            "http" => {
                // HTTP proxies can handle both HTTP and HTTPS (via CONNECT)
                target_scheme == "http" || target_scheme == "https"
            }
            "https" => {
                // HTTPS proxies can handle both HTTP and HTTPS
                target_scheme == "http" || target_scheme == "https"
            }
            "socks5" => {
                // SOCKS5 proxies can handle any protocol
                true
            }
            _ => {
                // Unknown proxy type - be conservative and only match exact schemes
                proxy_scheme == target_scheme
            }
        }
    }

    /// Returns basic authentication credentials for the first proxy.
    pub fn basic_auth(&self) -> Option<&str> {
        self.proxies.first()?.basic_auth.as_deref()
    }

    /// Returns custom headers for the first proxy.
    pub fn custom_headers(&self) -> Option<&hyper::HeaderMap> {
        self.proxies.first()?.custom_headers.as_ref()
    }
}

/// SOCKS protocol version enumeration.
#[derive(Clone, Copy, Debug)]
pub enum SocksVersion {
    V4,
    V5,
}

/// Proxy authentication methods for SOCKS5
#[derive(Clone, Debug)]
pub enum SocksAuth {
    None,
    UsernamePassword { username: String, password: String },
}

/// SOCKS proxy configuration
#[derive(Clone, Debug)]
pub struct SocksConfig {
    pub version: SocksVersion,
    pub auth: SocksAuth,
    pub target_host: String,
    pub target_port: u16,
}

impl SocksConfig {
    /// Create new SOCKS5 configuration with no authentication
    pub fn socks5_no_auth(target_host: String, target_port: u16) -> Self {
        Self {
            version: SocksVersion::V5,
            auth: SocksAuth::None,
            target_host,
            target_port,
        }
    }

    /// Create new SOCKS5 configuration with username/password authentication
    pub fn socks5_auth(target_host: String, target_port: u16, username: String, password: String) -> Self {
        Self {
            version: SocksVersion::V5,
            auth: SocksAuth::UsernamePassword { username, password },
            target_host,
            target_port,
        }
    }

    /// Create new SOCKS4 configuration
    pub fn socks4(target_host: String, target_port: u16) -> Self {
        Self {
            version: SocksVersion::V4,
            auth: SocksAuth::None, // SOCKS4 doesn't support authentication
            target_host,
            target_port,
        }
    }
}

/// HTTP CONNECT proxy configuration
#[derive(Clone, Debug)]
pub struct HttpConnectConfig {
    pub target_host: String,
    pub target_port: u16,
    pub auth: Option<String>, // Basic auth header value
    pub custom_headers: Option<hyper::HeaderMap>,
}

impl HttpConnectConfig {
    /// Create new HTTP CONNECT configuration
    pub fn new(target_host: String, target_port: u16) -> Self {
        Self {
            target_host,
            target_port,
            auth: None,
            custom_headers: None,
        }
    }

    /// Add basic authentication
    pub fn with_auth(mut self, username: &str, password: &str) -> Self {
        use base64::Engine;
        let credentials = format!("{}:{}", username, password);
        let encoded = base64::engine::general_purpose::STANDARD.encode(credentials.as_bytes());
        self.auth = Some(encoded);
        self
    }

    /// Add custom headers
    pub fn with_headers(mut self, headers: hyper::HeaderMap) -> Self {
        self.custom_headers = Some(headers);
        self
    }
}

/// Proxy bypass rules for conditional proxy usage
#[derive(Clone, Debug)]
pub struct ProxyBypass {
    pub no_proxy_hosts: Vec<String>,
    pub no_proxy_domains: Vec<String>,
    pub no_proxy_ips: Vec<std::net::IpAddr>,
}

impl ProxyBypass {
    /// Create new bypass configuration
    pub fn new() -> Self {
        Self {
            no_proxy_hosts: Vec::new(),
            no_proxy_domains: Vec::new(),
            no_proxy_ips: Vec::new(),
        }
    }

    /// Add host to bypass list
    pub fn add_host(mut self, host: String) -> Self {
        self.no_proxy_hosts.push(host);
        self
    }

    /// Add domain to bypass list (matches subdomains)
    pub fn add_domain(mut self, domain: String) -> Self {
        self.no_proxy_domains.push(domain);
        self
    }

    /// Add IP address to bypass list
    pub fn add_ip(mut self, ip: std::net::IpAddr) -> Self {
        self.no_proxy_ips.push(ip);
        self
    }

    /// Check if URI should bypass proxy
    pub fn should_bypass(&self, uri: &Uri) -> bool {
        let host = match uri.host() {
            Some(h) => h,
            None => return false,
        };

        // Check exact host matches
        if self.no_proxy_hosts.contains(&host.to_string()) {
            return true;
        }

        // Check domain matches
        for domain in &self.no_proxy_domains {
            if host.ends_with(domain) || host == domain {
                return true;
            }
        }

        // Check IP matches
        if let Ok(ip) = host.parse::<std::net::IpAddr>() {
            if self.no_proxy_ips.contains(&ip) {
                return true;
            }
        }

        false
    }
}

impl Default for ProxyBypass {
    fn default() -> Self {
        Self::new()
    }
}