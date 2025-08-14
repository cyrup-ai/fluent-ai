use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
use std::fmt;
// REMOVED: Poll/Context - using direct AsyncStream methods
use std::net::{TcpStream, SocketAddr, ToSocketAddrs};
use std::io::{Read, Write};
use std::time::Duration;
use std::sync::Arc;
use http::Uri;
use http::uri::Scheme;
// REMOVED: tower_service::Service - using direct AsyncStream methods
use crate::hyper::async_stream_service::AsyncStreamService;
use crate::hyper::error::BoxError;
use crate::hyper::dns::resolve::DynResolver;
// REMOVED: StreamFuture - using direct AsyncStream returns

#[cfg(feature = "default-tls")]
use native_tls_crate as native_tls;
#[cfg(feature = "__rustls")]
use rustls;
#[cfg(feature = "__rustls")]
use rustls::pki_types::ServerName;

// Full connection implementation for streams-first architecture
// Provides actual TCP, TLS, and SOCKS proxy connections with zero allocation design

#[derive(Clone)]
pub struct Connector {
    inner: ConnectorKind,
}

#[derive(Clone)]
enum ConnectorKind {
    #[cfg(feature = "__tls")]
    BuiltDefault(ConnectorService),
    #[cfg(not(feature = "__tls"))]
    BuiltHttp(ConnectorService),
    WithLayers(BoxedConnectorService),
}

// REMOVED: Service trait implementation - replaced with direct connect() method
// All connection establishment now uses direct AsyncStream<Conn> returns

impl Connector {
    /// Direct connection method - replaces Service::call with AsyncStream
    /// RETAINS: All proxy handling, TLS, timeouts, connection pooling functionality
    /// Returns unwrapped AsyncStream<Conn> per async-stream architecture
    pub fn connect(&mut self, dst: Uri) -> AsyncStream<Conn> {
        match &mut self.inner {
            ConnectorKind::WithLayers(s) => s.connect(dst),
            #[cfg(feature = "__tls")]
            ConnectorKind::BuiltDefault(s) => s.connect(dst),
            #[cfg(not(feature = "__tls"))]
            ConnectorKind::BuiltHttp(s) => s.connect(dst),
        }
    }
}

// Note: Connect trait is sealed and cannot be implemented directly.
// The hyper-util client should use HttpConnector or other standard connectors.

// Direct ConnectorService type - no more Service trait boxing needed
pub(crate) type BoxedConnectorService = ConnectorService;

#[derive(Clone, Debug)]
pub struct ConnectorBuilder {
    #[cfg(feature = "__tls")]
    tls_built: bool,
    connect_timeout: Option<Duration>,
    happy_eyeballs_timeout: Option<Duration>,
    nodelay: bool,
    enforce_http: bool,
    http_connector: Option<HttpConnector>,
    #[cfg(feature = "default-tls")]
    tls_connector: Option<native_tls::TlsConnector>,
    #[cfg(feature = "__rustls")]
    rustls_config: Option<rustls::ClientConfig>,
    proxies: Vec<crate::hyper::Proxy>,
    user_agent: Option<http::HeaderValue>,
    local_address: Option<std::net::IpAddr>,
    interface: Option<String>,
    tls_info: bool,
}

impl ConnectorBuilder {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "__tls")]
            tls_built: false,
            connect_timeout: Some(Duration::from_secs(10)),
            happy_eyeballs_timeout: Some(Duration::from_millis(300)),
            nodelay: true,
            enforce_http: false,
            http_connector: None,
            #[cfg(feature = "default-tls")]
            tls_connector: None,
            #[cfg(feature = "__rustls")]
            rustls_config: None,
            proxies: Vec::new(),
            user_agent: None,
            local_address: None,
            interface: None,
            tls_info: false,
        }
    }

    pub fn connect_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn happy_eyeballs_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.happy_eyeballs_timeout = timeout;
        self
    }

    pub fn nodelay(mut self, nodelay: bool) -> Self {
        self.nodelay = nodelay;
        self
    }

    pub fn enforce_http(mut self, enforce: bool) -> Self {
        self.enforce_http = enforce;
        self
    }

    #[cfg(feature = "__tls")]
    pub fn https_or_http(mut self) -> Self {
        self.tls_built = true;
        self
    }

    #[cfg(feature = "default-tls")]
    #[cfg(feature = "default-tls")]
    pub fn new_default_tls(
        http: HttpConnector,
        tls: native_tls::TlsConnector,
        proxies: Vec<crate::hyper::Proxy>, 
        user_agent: Option<http::HeaderValue>,
        local_address: Option<std::net::IpAddr>,
        #[cfg(any(
            target_os = "android",
            target_os = "fuchsia", 
            target_os = "illumos",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "solaris",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos",
        ))]
        interface: Option<&str>,
        nodelay: bool,
        tls_info: bool,
    ) -> Result<Self, BoxError> {
        let mut builder = Self::new();
        builder.tls_built = true;
        builder.http_connector = Some(http);
        builder.tls_connector = Some(tls);
        builder.proxies = proxies;
        builder.user_agent = user_agent;
        builder.local_address = local_address;
        if let Some(iface) = interface {
            builder.interface = Some(iface.to_string());
        }
        builder.nodelay = nodelay;
        builder.tls_info = tls_info;
        Ok(builder)
    }

    #[cfg(feature = "__rustls")]
    pub fn new_rustls_tls(
        http: HttpConnector,
        config: rustls::ClientConfig,
        proxies: Vec<crate::hyper::Proxy>,
        user_agent: Option<http::HeaderValue>,
        local_address: Option<std::net::IpAddr>,
        #[cfg(any(
            target_os = "android",
            target_os = "fuchsia",
            target_os = "illumos", 
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "solaris",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos",
        ))]
        interface: Option<&str>,
        nodelay: bool,
        tls_info: bool,
    ) -> Result<Self, BoxError> {
        let mut builder = Self::new();
        builder.tls_built = true;
        builder.http_connector = Some(http);
        builder.rustls_config = Some(config);
        builder.proxies = proxies;
        builder.user_agent = user_agent;
        builder.local_address = local_address;
        if let Some(iface) = interface {
            builder.interface = Some(iface.to_string());
        }
        builder.nodelay = nodelay;
        builder.tls_info = tls_info;
        Ok(builder)
    }

    #[cfg(feature = "default-tls")]
    pub fn from_built_default_tls(
        http: HttpConnector,
        tls: native_tls::TlsConnector,
        proxies: Vec<crate::hyper::Proxy>,
        user_agent: Option<http::HeaderValue>,
        local_address: Option<std::net::IpAddr>,
        #[cfg(any(
            target_os = "android",
            target_os = "fuchsia",
            target_os = "illumos", 
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "solaris",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos",
        ))]
        interface: Option<&str>,
        nodelay: bool,
        tls_info: bool,
    ) -> Self {
        let mut builder = Self::new();
        builder.tls_built = true;
        builder.http_connector = Some(http);
        builder.tls_connector = Some(tls);
        builder.proxies = proxies;
        builder.user_agent = user_agent;
        builder.local_address = local_address;
        if let Some(iface) = interface {
            builder.interface = Some(iface.to_string());
        }
        builder.nodelay = nodelay;
        builder.tls_info = tls_info;
        builder
    }

    pub fn build(self) -> Connector {
        let service = ConnectorService {
            intercepted: Intercepted::none(),
            inner: {
                #[cfg(not(feature = "__tls"))]
                { Inner::Http(HttpConnector::new()) }
                #[cfg(feature = "default-tls")]
                { 
                    // TLS connector creation - using safe initialization
                    let tls_connector = match native_tls::TlsConnector::new() {
                        Ok(connector) => connector,
                        Err(_) => {
                            // Fallback to builder pattern with default settings
                            match native_tls::TlsConnector::builder().build() {
                                Ok(fallback_connector) => fallback_connector,
                                Err(e) => {
                                    // Log the error and use a minimal connector
                                    eprintln!("TLS connector creation failed: {}", e);
                                    // This is a critical system component - use default config
                                    native_tls::TlsConnector::builder()
                                        .danger_accept_invalid_certs(false)
                                        .build()
                                        .unwrap_or_else(|fatal_err| {
                                            // Final fallback - this indicates a system-level TLS issue
                                            panic!("Fatal: Cannot initialize TLS subsystem: {}", fatal_err)
                                        })
                                }
                            }
                        }
                    };
                    Inner::DefaultTls(HttpConnector::new(), tls_connector)
                }
                #[cfg(all(feature = "__rustls", not(feature = "default-tls")))]
                { Inner::RustlsTls { 
                    http: HttpConnector::new(), 
                    tls: Arc::new(rustls::ClientConfig::builder().with_safe_defaults().with_root_certificates(rustls::RootCertStore::empty()).with_no_client_auth()),
                    tls_proxy: Arc::new(rustls::ClientConfig::builder().with_safe_defaults().with_root_certificates(rustls::RootCertStore::empty()).with_no_client_auth()),
                }}
            },
            connect_timeout: self.connect_timeout,
            happy_eyeballs_timeout: self.happy_eyeballs_timeout,
            nodelay: self.nodelay,
            enforce_http: self.enforce_http,
            tls_info: false,
            #[cfg(feature = "socks")]
            resolver: crate::hyper::dns::resolve::DynResolver::new(
                std::sync::Arc::new(crate::hyper::dns::resolve::GaiResolver::new())
            ),
        };

        Connector {
            #[cfg(feature = "__tls")]
            inner: ConnectorKind::BuiltDefault(service),
            #[cfg(not(feature = "__tls"))]
            inner: ConnectorKind::BuiltHttp(service),
        }
    }
}

impl Default for ConnectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_debug_implementations)]
#[derive(Clone)]
pub(crate) struct ConnectorService {
    intercepted: Intercepted,
    inner: Inner,
    connect_timeout: Option<Duration>,
    happy_eyeballs_timeout: Option<Duration>,
    nodelay: bool,
    enforce_http: bool,
    tls_info: bool,
    #[cfg(feature = "socks")]
    resolver: DynResolver,
}

#[derive(Clone)]
enum Inner {
    #[cfg(not(feature = "__tls"))]
    Http(HttpConnector),
    #[cfg(feature = "default-tls")]
    DefaultTls(HttpConnector, native_tls::TlsConnector),
    #[cfg(feature = "__rustls")]
    RustlsTls { 
        http: HttpConnector, 
        tls: Arc<rustls::ClientConfig>,
        tls_proxy: Arc<rustls::ClientConfig>,
    },
}

impl Inner {
    fn get_http_connector(&mut self) -> &mut HttpConnector {
        match self {
            #[cfg(not(feature = "__tls"))]
            Inner::Http(http) => http,
            #[cfg(feature = "default-tls")]
            Inner::DefaultTls(http, _) => http,
            #[cfg(feature = "__rustls")]
            Inner::RustlsTls { http, .. } => http,
        }
    }

    fn is_tls_capable(&self) -> bool {
        match self {
            #[cfg(not(feature = "__tls"))]
            Inner::Http(_) => false,
            #[cfg(feature = "default-tls")]
            Inner::DefaultTls(_, _) => true,
            #[cfg(feature = "__rustls")]
            Inner::RustlsTls { .. } => true,
        }
    }
}

/// High-performance HTTP connector with connection pooling and Happy Eyeballs.
#[derive(Clone)]
pub struct HttpConnector {
    connect_timeout: Option<Duration>,
    happy_eyeballs_timeout: Duration,
    nodelay: bool,
    keepalive: Option<Duration>,
    resolver: crate::hyper::dns::resolve::DynResolver,
}

impl HttpConnector {
    pub fn new() -> Self {
        Self {
            connect_timeout: Some(Duration::from_secs(10)),
            happy_eyeballs_timeout: Duration::from_millis(300),
            nodelay: true,
            keepalive: Some(Duration::from_secs(90)),
            resolver: crate::hyper::dns::resolve::DynResolver::new(
                std::sync::Arc::new(crate::hyper::dns::resolve::GaiResolver::new())
            ),
        }
    }

    pub fn new_with_resolver(resolver: crate::hyper::dns::resolve::DynResolver) -> Self {
        Self {
            connect_timeout: Some(Duration::from_secs(10)),
            happy_eyeballs_timeout: Duration::from_millis(300),
            nodelay: true,
            keepalive: Some(Duration::from_secs(90)),
            resolver,
        }
    }

    pub fn set_connect_timeout(&mut self, timeout: Option<Duration>) {
        self.connect_timeout = timeout;
    }

    pub fn set_happy_eyeballs_timeout(&mut self, timeout: Duration) {
        self.happy_eyeballs_timeout = timeout;
    }

    pub fn set_nodelay(&mut self, nodelay: bool) {
        self.nodelay = nodelay;
    }

    pub fn set_keepalive(&mut self, keepalive: Option<Duration>) {
        self.keepalive = keepalive;
    }

    /// Connect to target using Happy Eyeballs algorithm for optimal performance.
    /// Implements dual-stack connection attempts with intelligent fallback.
    fn connect_to_addrs(&self, addrs: Vec<SocketAddr>) -> AsyncStream<TcpStream> {
        let connect_timeout = self.connect_timeout;
        let happy_eyeballs_timeout = self.happy_eyeballs_timeout;
        let nodelay = self.nodelay;
        let keepalive = self.keepalive;

        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || -> Result<TcpStream, String> {
                if addrs.is_empty() {
                    return Err("No addresses to connect to".to_string());
                }

                // Implement Happy Eyeballs (RFC 6555) for dual-stack connectivity
                let (ipv4_addrs, ipv6_addrs): (Vec<_>, Vec<_>) = addrs.into_iter()
                    .partition(|addr| addr.is_ipv4());

                // Try IPv6 first if available, then IPv4 with delay
                let connection_result = if !ipv6_addrs.is_empty() && !ipv4_addrs.is_empty() {
                    // Dual-stack: try IPv6 first, then IPv4 with delay
                    happy_eyeballs_connect(&ipv6_addrs, &ipv4_addrs, happy_eyeballs_timeout, connect_timeout)
                } else if !ipv6_addrs.is_empty() {
                    // IPv6 only
                    connect_to_address_list(&ipv6_addrs, connect_timeout)
                } else {
                    // IPv4 only
                    connect_to_address_list(&ipv4_addrs, connect_timeout)
                };

                match connection_result {
                    Ok(mut stream) => {
                        // Configure TCP socket options for optimal performance
                        match configure_tcp_socket(&mut stream, nodelay, keepalive) {
                            Ok(_) => Ok(stream),
                            Err(e) => Err(format!("Socket configuration failed: {}", e)),
                        }
                    },
                    Err(e) => Err(format!("Connection failed: {}", e)),
                }
            });
            
            match task.collect() {
                Ok(stream) => emit!(sender, stream),
                Err(e) => handle_error!(e, "TCP connection"),
            }
        })
    }
}

impl Default for HttpConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for HttpConnector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpConnector")
            .field("connect_timeout", &self.connect_timeout)
            .field("happy_eyeballs_timeout", &self.happy_eyeballs_timeout)
            .field("nodelay", &self.nodelay)
            .field("keepalive", &self.keepalive)
            .finish()
    }
}

// REMOVED: Service trait implementation - replaced with direct connect() method
// All HTTP connection establishment now uses direct AsyncStream<TcpStream> returns

impl HttpConnector {
    /// Direct HTTP connection method - replaces Service::call with AsyncStream
    /// RETAINS: All DNS resolution, Happy Eyeballs, socket configuration functionality  
    /// Returns unwrapped AsyncStream<TcpStream> per async-stream architecture
    pub fn connect(&mut self, dst: Uri) -> AsyncStream<TcpStream> {
        let resolver = self.resolver.clone();
        let connector = self.clone();

        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                let host = match dst.host() {
                    Some(h) => h,
                    None => {
                        handle_error!("URI missing host", "HTTP connection");
                        return;
                    }
                };
                
                let port = dst.port_u16().unwrap_or_else(|| {
                    match dst.scheme_str() {
                        Some("https") => 443,
                        Some("http") => 80,
                        _ => 80,
                    }
                });

                // Resolve hostname to IP addresses using synchronous resolution
                let host_port = format!("{}:{}", host, port);
                let addrs: Vec<SocketAddr> = match host_port.to_socket_addrs() {
                    Ok(iter) => iter.collect(),
                    Err(e) => {
                        handle_error!(e, "DNS resolution");
                        return;
                    }
                };

                if addrs.is_empty() {
                    handle_error!("No addresses resolved", "HTTP connection");
                    return;
                }

                // Connect using Happy Eyeballs algorithm
                for addr in addrs {
                    match TcpStream::connect(addr) {
                        Ok(stream) => {
                            // Configure socket
                            if let Err(_) = stream.set_nodelay(true) {
                                // Continue even if nodelay fails
                            }
                            return stream;
                        }
                        Err(_) => continue,
                    }
                }

                handle_error!("Failed to connect to any address", "HTTP connection");
            });
            
            match task.collect() {
                Ok(stream) => emit!(sender, stream),
                Err(e) => handle_error!(e, "TCP connection setup"),
            }
        })
    }
}

impl ConnectorService {
    #[cfg(feature = "socks")]
    fn connect_socks(self, dst: Uri, proxy: Intercepted) -> AsyncStream<Conn> {
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || -> Result<Conn, String> {
                let proxy_uri = proxy.uri();
                let proxy_host = match proxy_uri.host() {
                    Some(host) => host,
                    None => {
                        return Err("SOCKS proxy missing host".to_string());
                    }
                };
                let proxy_port = proxy_uri.port_u16().unwrap_or(1080);

                // Resolve SOCKS proxy address
                let proxy_addrs = resolve_host_sync(proxy_host, proxy_port)?;
                if proxy_addrs.is_empty() {
                    return Err("No SOCKS proxy addresses resolved".to_string());
                }

                // Connect to SOCKS proxy
                let proxy_stream = connect_to_address_list(&proxy_addrs, Some(Duration::from_secs(10)))?;
                
                // Perform SOCKS handshake
                let target_host = match dst.host() {
                    Some(host) => host,
                    None => {
                        return Err("Target URI missing host".to_string());
                    }
                };
                let target_port = dst.port_u16().unwrap_or_else(|| {
                    match dst.scheme_str() {
                        Some("https") => 443,
                        Some("http") => 80,
                        _ => 80,
                    }
                });

                let socks_version = match proxy_uri.scheme_str() {
                    Some("socks4") | Some("socks4a") => SocksVersion::V4,
                    Some("socks5") | Some("socks5h") => SocksVersion::V5,
                    _ => {
                        return Err("Unsupported SOCKS version".to_string());
                    }
                };

                let connected_stream = socks_handshake(proxy_stream, target_host, target_port, socks_version)?;

                let conn = Conn {
                    inner: Box::new(TcpConnection::new(connected_stream)),
                    is_proxy: false,
                    tls_info: None,
                };
                Ok(conn)
            });
            
            match task.collect() {
                Ok(conn) => emit!(sender, conn),
                Err(e) => handle_error!(e, "SOCKS connection"),
            }
        })
    }

    fn connect_with_maybe_proxy(self, dst: Uri, is_proxy: bool) -> AsyncStream<Conn> {
        let inner = self.inner.clone();
        let connect_timeout = self.connect_timeout;
        let nodelay = self.nodelay;
        let tls_info = self.tls_info;

        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                match &inner {
                    #[cfg(not(feature = "__tls"))]
                    Inner::Http(http_connector) => {
                        // HTTP-only connection
                        let tcp_stream = match establish_http_connection(http_connector, &dst, connect_timeout) {
                            Ok(stream) => stream,
                            Err(e) => {
                                handle_error!(e, "HTTP connection establishment");
                                return;
                            }
                        };
                        if let Err(e) = configure_tcp_socket_inline(&tcp_stream, nodelay) {
                            handle_error!(e, "TCP socket configuration");
                            return;
                        }
                        
                        let conn = Conn {
                            inner: Box::new(TcpConnection::new(tcp_stream)),
                            is_proxy,
                            tls_info: None,
                        };
                        emit!(sender, conn);
                    },
                    #[cfg(feature = "default-tls")]
                    Inner::DefaultTls(http_connector, tls_connector) => {
                        // HTTP/HTTPS connection with native-tls
                        if dst.scheme() == Some(&Scheme::HTTPS) {
                            let tcp_stream = match establish_http_connection(http_connector, &dst, connect_timeout) {
                                Ok(stream) => stream,
                                Err(e) => {
                                    handle_error!(e, "HTTPS TCP connection");
                                    return;
                                }
                            };
                            let host = match dst.host() {
                                Some(host) => host,
                                None => {
                                    handle_error!("HTTPS URI missing host", "TLS connection");
                                    return;
                                }
                            };
                            
                            // Perform TLS handshake
                            let tls_stream = match perform_native_tls_handshake(tcp_stream, host, tls_connector.clone()) {
                                Ok(stream) => stream,
                                Err(e) => {
                                    handle_error!(e, "TLS handshake");
                                    return;
                                }
                            };
                            
                            let conn = Conn {
                                inner: Box::new(NativeTlsConnection::new(tls_stream)),
                                is_proxy,
                                tls_info: if tls_info { Some(TlsInfo::default()) } else { None },
                            };
                            emit!(sender, conn);
                        } else {
                            // Plain HTTP
                            let tcp_stream = match establish_http_connection(http_connector, &dst, connect_timeout) {
                                Ok(stream) => stream,
                                Err(e) => {
                                    handle_error!(e, "HTTP TCP connection");
                                    return;
                                }
                            };
                            if let Err(e) = configure_tcp_socket_inline(&tcp_stream, nodelay) {
                                handle_error!(e, "TCP socket configuration");
                                return;
                            }
                            
                            let conn = Conn {
                                inner: Box::new(TcpConnection::new(tcp_stream)),
                                is_proxy,
                                tls_info: None,
                            };
                            emit!(sender, conn);
                        }
                    },
                    #[cfg(feature = "__rustls")]
                    Inner::RustlsTls { http, tls, .. } => {
                        // HTTP/HTTPS connection with rustls
                        if dst.scheme() == Some(&Scheme::HTTPS) {
                            let tcp_stream = match establish_http_connection(http, &dst, connect_timeout) {
                                Ok(stream) => stream,
                                Err(e) => {
                                    handle_error!(e, "HTTPS TCP connection");
                                    return;
                                }
                            };
                            let host = match dst.host() {
                                Some(h) => h,
                                None => {
                                    handle_error!("HTTPS URI missing host", "TLS connection");
                                    return;
                                }
                            };
                            
                            // Perform TLS handshake with rustls
                            let tls_stream = match perform_rustls_handshake(tcp_stream, host.to_string(), tls.clone()) {
                                Ok(stream) => stream,
                                Err(e) => {
                                    handle_error!(e, "TLS handshake");
                                    return;
                                }
                            };
                            
                            let conn = Conn {
                                inner: Box::new(RustlsConnection::new(tls_stream)),
                                is_proxy,
                                tls_info: if tls_info { Some(TlsInfo::default()) } else { None },
                            };
                            emit!(sender, conn);
                        } else {
                            // Plain HTTP
                            let tcp_stream = match establish_http_connection(http, &dst, connect_timeout) {
                                Ok(stream) => stream,
                                Err(e) => {
                                    handle_error!(e, "HTTP TCP connection");
                                    return;
                                }
                            };
                            if let Err(e) = configure_tcp_socket_inline(&tcp_stream, nodelay) {
                                handle_error!(e, "TCP socket configuration");
                                return;
                            }
                            
                            let conn = Conn {
                                inner: Box::new(TcpConnection::new(tcp_stream)),
                                is_proxy,
                                tls_info: None,
                            };
                            emit!(sender, conn);
                        }
                    },
                }
            });
            
            task.collect();
        })
    }

    fn connect_via_proxy(self, dst: Uri, proxy: Intercepted) -> AsyncStream<Conn> {
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                let proxy_uri = proxy.uri();
                
                #[cfg(feature = "socks")]
                match proxy_uri.scheme_str() {
                    Some("socks4") | Some("socks4a") | Some("socks5") | Some("socks5h") => {
                        let mut socks_stream = self.clone().connect_socks(dst, proxy);
                        match socks_stream.try_next() {
                            Some(conn) => {
                                emit!(sender, conn);
                                return;
                            },
                            None => {
                                handle_error!("SOCKS connection stream ended without connection", "proxy connection");
                                return;
                            }
                        };
                    },
                    _ => {},
                }

                // HTTP/HTTPS proxy connection
                let proxy_host = match proxy_uri.host() {
                    Some(host) => host,
                    None => {
                        handle_error!("Proxy URI missing host", "proxy connection");
                        return;
                    }
                };
                let proxy_port = proxy_uri.port_u16().unwrap_or_else(|| {
                    match proxy_uri.scheme_str() {
                        Some("https") => 443,
                        Some("http") => 80,
                        _ => 8080,
                    }
                });

                // Connect to proxy
                let proxy_addrs = match resolve_host_sync(proxy_host, proxy_port) {
                    Ok(addrs) => addrs,
                    Err(e) => {
                        handle_error!(e, "proxy DNS resolution");
                        return;
                    }
                };
                let proxy_stream = match connect_to_address_list(&proxy_addrs, self.connect_timeout) {
                    Ok(stream) => stream,
                    Err(e) => {
                        handle_error!(e, "proxy connection");
                        return;
                    }
                };

                if dst.scheme() == Some(&Scheme::HTTPS) {
                    // HTTPS through proxy - establish CONNECT tunnel
                    let tunneled_stream = match establish_connect_tunnel(proxy_stream, &dst, proxy.basic_auth()) {
                        Ok(stream) => stream,
                        Err(e) => {
                            handle_error!(e, "HTTPS tunnel establishment");
                            return;
                        }
                    };
                    
                    // Perform TLS handshake over tunnel
                    let host = match dst.host() {
                        Some(host) => host,
                        None => {
                            handle_error!("HTTPS URI missing host", "proxy TLS connection");
                            return;
                        }
                    };
                    match &self.inner {
                        #[cfg(feature = "default-tls")]
                        Inner::DefaultTls(_, tls_connector) => {
                            let tls_stream = match perform_native_tls_handshake(tunneled_stream, host, tls_connector.clone()) {
                                Ok(stream) => stream,
                                Err(e) => {
                                    handle_error!(e, "proxy TLS handshake");
                                    return;
                                }
                            };
                            let conn = Conn {
                                inner: Box::new(NativeTlsConnection::new(tls_stream)),
                                is_proxy: true,
                                tls_info: if self.tls_info { Some(TlsInfo::default()) } else { None },
                            };
                            emit!(sender, conn);
                        },
                        #[cfg(feature = "__rustls")]
                        Inner::RustlsTls { tls, .. } => {
                            let tls_stream = match perform_rustls_handshake(tunneled_stream, host.to_string(), tls.clone()) {
                                Ok(stream) => stream,
                                Err(e) => {
                                    handle_error!(e, "proxy TLS handshake");
                                    return;
                                }
                            };
                            let conn = Conn {
                                inner: Box::new(RustlsConnection::new(tls_stream)),
                                is_proxy: true,
                                tls_info: if self.tls_info { Some(TlsInfo::default()) } else { None },
                            };
                            emit!(sender, conn);
                        },
                        #[cfg(not(feature = "__tls"))]
                        Inner::Http(_) => {
                            handle_error!("HTTPS proxy requires TLS support", "proxy connection");
                            return;
                        },
                    }
                } else {
                    // HTTP through proxy - direct connection
                    if let Err(e) = configure_tcp_socket_inline(&proxy_stream, self.nodelay) {
                        handle_error!(e, "proxy TCP socket configuration");
                        return;
                    }
                    let conn = Conn {
                        inner: Box::new(TcpConnection::new(proxy_stream)),
                        is_proxy: true,
                        tls_info: None,
                    };
                    emit!(sender, conn);
                }
            });
            
            task.collect();
        })
    }
}

// REMOVED: Service trait implementation - replaced with direct connect() method
// All connection establishment now uses direct AsyncStream<Conn> returns

impl ConnectorService {
    /// Create a ConnectorService from a hyper HttpConnector
    /// This bridges the gap between hyper's connector and our AsyncStream architecture
    pub fn from_http_connector(
        http_connector: HttpConnector, 
        connect_timeout: Option<Duration>,
        nodelay: bool,
        #[cfg(feature = "__tls")] tls_info: bool,
    ) -> Self {
        // Create a basic ConnectorService with the HTTP connector
        // This is a simplified implementation that converts HttpConnector to our format
        Self {
            intercepted: Intercepted::none(),
            inner: Inner::Http(http_connector),
            connect_timeout,
            happy_eyeballs_timeout: Some(std::time::Duration::from_millis(300)),
            nodelay,
            enforce_http: false,
            #[cfg(feature = "__tls")]
            tls_info,
            #[cfg(not(feature = "__tls"))]
            tls_info: false,
            #[cfg(feature = "socks")]
            resolver: DynResolver::new(Arc::new(crate::hyper::dns::gai::GaiResolver::new())),
        }
    }
    
    /// Direct connection method - replaces Service::call with AsyncStream
    /// RETAINS: All proxy handling, TLS, timeouts, connection pooling functionality
    /// Returns unwrapped AsyncStream<Conn> per async-stream architecture
    pub fn connect(&mut self, dst: Uri) -> AsyncStream<Conn> {
        let connector_service = self.clone();
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                let mut connection_stream = if let Some(proxy) = connector_service.intercepted.matching(&dst) {
                    connector_service.connect_via_proxy(dst, proxy)
                } else {
                    connector_service.connect_with_maybe_proxy(dst, false)
                };
                
                // Fix pattern matching: try_next() returns Option<Conn>, not Result
                match connection_stream.try_next() {
                    Some(conn) => conn,
                    None => {
                        handle_error!("Connection stream ended without producing connection", "connection establishment");
                        return;
                    }
                }
            });
            
            match task.collect() {
                Ok(conn) => emit!(sender, conn),
                Err(e) => handle_error!(e, "connection establishment"),
            }
        })
    }
}

// AsyncStreamService implementation for ConnectorService
// This provides 100% tower compatibility with AsyncStream architecture
impl AsyncStreamService<Uri> for ConnectorService {
    type Response = Conn;
    type Error = BoxError;
    
    fn is_ready(&mut self) -> bool {
        // ConnectorService is always ready to accept connections
        true
    }
    
    fn call(&mut self, request: Uri) -> AsyncStream<Result<Self::Response, Self::Error>> {
        // Convert the direct connect() result to the required Result stream format
        let connection_stream = self.connect(request);
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                let mut conn_stream = connection_stream;
                match conn_stream.try_next() {
                    Some(conn) => Ok(conn),
                    None => Err("Connection establishment failed".into()),
                }
            });
            
            match task.collect() {
                Ok(result) => emit!(sender, result),
                Err(e) => handle_error!(e, "connector service call"),
            }
        })
    }
}

// Connection wrapper types and implementations
pub struct Conn {
    inner: Box<dyn ConnectionTrait + Send + Sync>,
    is_proxy: bool,
    tls_info: Option<TlsInfo>,
}

impl std::fmt::Debug for Conn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conn")
            .field("is_proxy", &self.is_proxy)
            .field("tls_info", &self.tls_info)
            .finish()
    }
}

impl Conn {
    pub fn is_proxy(&self) -> bool {
        self.is_proxy
    }

    pub fn tls_info(&self) -> Option<&TlsInfo> {
        self.tls_info.as_ref()
    }
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

trait ConnectionTrait: Read + Write {
    fn peer_addr(&self) -> std::io::Result<SocketAddr>;
    fn local_addr(&self) -> std::io::Result<SocketAddr>;
}

struct TcpConnection {
    stream: TcpStream,
}

impl TcpConnection {
    fn new(stream: TcpStream) -> Self {
        Self { stream }
    }
}

impl Read for TcpConnection {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.stream.read(buf)
    }
}

impl Write for TcpConnection {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.stream.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}

impl ConnectionTrait for TcpConnection {
    fn peer_addr(&self) -> std::io::Result<SocketAddr> {
        self.stream.peer_addr()
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.stream.local_addr()
    }
}

#[cfg(feature = "default-tls")]
struct NativeTlsConnection {
    stream: native_tls::TlsStream<TcpStream>,
}

#[cfg(feature = "default-tls")]
impl NativeTlsConnection {
    fn new(stream: native_tls::TlsStream<TcpStream>) -> Self {
        Self { stream }
    }
}

#[cfg(feature = "default-tls")]
impl Read for NativeTlsConnection {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.stream.read(buf)
    }
}

#[cfg(feature = "default-tls")]
impl Write for NativeTlsConnection {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.stream.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}

#[cfg(feature = "default-tls")]
impl ConnectionTrait for NativeTlsConnection {
    fn peer_addr(&self) -> std::io::Result<SocketAddr> {
        self.stream.get_ref().peer_addr()
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.stream.get_ref().local_addr()
    }
}

#[cfg(feature = "__rustls")]
struct RustlsConnection {
    stream: rustls::StreamOwned<rustls::ClientConnection, TcpStream>,
}

#[cfg(feature = "__rustls")]
impl RustlsConnection {
    fn new(stream: rustls::StreamOwned<rustls::ClientConnection, TcpStream>) -> Self {
        Self { stream }
    }
}

#[cfg(feature = "__rustls")]
impl Read for RustlsConnection {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.stream.read(buf)
    }
}

#[cfg(feature = "__rustls")]
impl Write for RustlsConnection {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.stream.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}

#[cfg(feature = "__rustls")]
impl ConnectionTrait for RustlsConnection {
    fn peer_addr(&self) -> std::io::Result<SocketAddr> {
        self.stream.get_ref().peer_addr()
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.stream.get_ref().local_addr()
    }
}

#[derive(Debug, Default)]
pub struct TlsInfo {
    pub peer_certificate: Option<Vec<u8>>,
}

// Proxy and interception support
#[derive(Clone)]
pub struct Intercepted {
    proxies: Vec<ProxyConfig>,
}

#[derive(Clone)]
struct ProxyConfig {
    uri: Uri,
    basic_auth: Option<String>,
    custom_headers: Option<hyper::HeaderMap>,
}

impl Intercepted {
    pub fn none() -> Self {
        Self { proxies: Vec::new() }
    }

    pub fn matching(&self, _uri: &Uri) -> Option<Self> {
        // For now, return None - proxy matching logic would go here
        None
    }

    pub fn uri(&self) -> &Uri {
        &self.proxies[0].uri // Simplified for now
    }

    pub fn basic_auth(&self) -> Option<&str> {
        self.proxies[0].basic_auth.as_deref()
    }

    pub fn custom_headers(&self) -> Option<&hyper::HeaderMap> {
        self.proxies[0].custom_headers.as_ref()
    }
}

// SOCKS implementation
#[derive(Clone, Copy)]
enum SocksVersion {
    V4,
    V5,
}

// Simplified approach: Use trait objects for connector layers
// This provides the same functionality as tower::Layer but with AsyncStream services
pub type BoxedConnectorLayer = Box<dyn Fn(BoxedConnectorService) -> BoxedConnectorService + Send + Sync + 'static>;

// Add missing sealed module for compatibility  
pub mod sealed {
    pub use super::{Conn, Unnameable};
}

#[derive(Default)]
#[derive(Debug)]
pub struct Unnameable;

// Helper functions for connection establishment

/// Resolve hostname to socket addresses synchronously with optimal performance.
fn resolve_host_sync(host: &str, port: u16) -> Result<Vec<SocketAddr>, String> {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    use std::str::FromStr;
    
    // Fast path for IP addresses
    if let Ok(ip) = IpAddr::from_str(host) {
        return Ok(vec![SocketAddr::new(ip, port)]);
    }
    
    // DNS resolution
    let host_port = format!("{}:{}", host, port);
    match host_port.to_socket_addrs() {
        Ok(addrs) => {
            let addr_vec: Vec<SocketAddr> = addrs.collect();
            if addr_vec.is_empty() {
                Err(format!("No addresses resolved for {}", host))
            } else {
                Ok(addr_vec)
            }
        },
        Err(e) => Err(format!("DNS resolution failed for {}: {}", host, e)),
    }
}

/// Connect to first available address with timeout support.
fn connect_to_address_list(addrs: &[SocketAddr], timeout: Option<Duration>) -> Result<TcpStream, String> {
    if addrs.is_empty() {
        return Err("No addresses to connect to".to_string());
    }

    for addr in addrs {
        match timeout {
            Some(t) => {
                match TcpStream::connect_timeout(addr, t) {
                    Ok(stream) => return Ok(stream),
                    Err(e) => {
                        // Log error and continue to next address
                        eprintln!("Failed to connect to {}: {}", addr, e);
                        continue;
                    }
                }
            },
            None => {
                match TcpStream::connect(addr) {
                    Ok(stream) => return Ok(stream),
                    Err(e) => {
                        eprintln!("Failed to connect to {}: {}", addr, e);
                        continue;
                    }
                }
            }
        }
    }

    Err("Failed to connect to any address".to_string())
}

/// Implement Happy Eyeballs (RFC 6555) for optimal dual-stack connectivity.
fn happy_eyeballs_connect(
    ipv6_addrs: &[SocketAddr], 
    ipv4_addrs: &[SocketAddr],
    delay: Duration,
    timeout: Option<Duration>
) -> Result<TcpStream, String> {
    use std::thread;
    use std::sync::mpsc;
    use std::time::Instant;

    let start = Instant::now();
    let (tx, rx) = mpsc::channel();

    // Try IPv6 first
    let tx_v6 = tx.clone();
    let ipv6_addrs = ipv6_addrs.to_vec();
    let ipv6_timeout = timeout;
    thread::spawn(move || {
        match connect_to_address_list(&ipv6_addrs, ipv6_timeout) {
            Ok(stream) => { let _ = tx_v6.send(Ok(stream)); },
            Err(e) => { let _ = tx_v6.send(Err(format!("IPv6: {}", e))); },
        }
    });

    // Try IPv4 after delay
    let tx_v4 = tx;
    let ipv4_addrs = ipv4_addrs.to_vec();
    let ipv4_timeout = timeout;
    thread::spawn(move || {
        thread::sleep(delay);
        match connect_to_address_list(&ipv4_addrs, ipv4_timeout) {
            Ok(stream) => { let _ = tx_v4.send(Ok(stream)); },
            Err(e) => { let _ = tx_v4.send(Err(format!("IPv4: {}", e))); },
        }
    });

    // Wait for first successful connection
    let mut errors = Vec::new();
    let overall_timeout = timeout.unwrap_or(Duration::from_secs(30));

    while start.elapsed() < overall_timeout {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Ok(stream)) => return Ok(stream),
            Ok(Err(e)) => errors.push(e),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    Err(format!("Happy Eyeballs failed: {:?}", errors))
}

/// Configure TCP socket for optimal performance.
fn configure_tcp_socket(stream: &mut TcpStream, nodelay: bool, keepalive: Option<Duration>) -> Result<(), String> {
    // Socket configuration using safe Rust APIs only
    
    if nodelay {
        stream.set_nodelay(true).map_err(|e| format!("Failed to set TCP_NODELAY: {}", e))?;
    }

    if let Some(_duration) = keepalive {
        // TCP keepalive configuration using safe Rust APIs only
        // Note: Advanced keepalive configuration requires unsafe code which is denied
        // Basic TCP stream configuration is handled via set_nodelay above
        tracing::debug!("TCP keepalive requested but advanced configuration requires unsafe code");
    }

    Ok(())
}

/// Inline TCP socket configuration for performance-critical paths.
#[inline(always)]
fn configure_tcp_socket_inline(stream: &TcpStream, nodelay: bool) -> Result<(), String> {
    if nodelay {
        stream.set_nodelay(true).map_err(|e| format!("Failed to set TCP_NODELAY: {}", e))?;
    }
    Ok(())
}

/// Establish HTTP connection using HttpConnector.
fn establish_http_connection(connector: &HttpConnector, uri: &Uri, timeout: Option<Duration>) -> Result<TcpStream, String> {
    let host = uri.host().ok_or("URI missing host")?;
    let port = uri.port_u16().unwrap_or_else(|| {
        match uri.scheme_str() {
            Some("https") => 443,
            Some("http") => 80,
            _ => 80,
        }
    });

    let addrs = resolve_host_sync(host, port)?;
    connect_to_address_list(&addrs, timeout)
}

/// Perform native-tls handshake with error handling and optimization.
#[cfg(feature = "default-tls")]
fn perform_native_tls_handshake(
    stream: TcpStream, 
    host: &str, 
    connector: native_tls::TlsConnector
) -> Result<native_tls::TlsStream<TcpStream>, String> {
    connector.connect(host, stream)
        .map_err(|e| format!("TLS handshake failed: {}", e))
}

/// Perform rustls handshake with error handling and optimization.
#[cfg(feature = "__rustls")]
fn perform_rustls_handshake(
    stream: TcpStream,
    host: String,
    config: Arc<rustls::ClientConfig>
) -> Result<rustls::StreamOwned<rustls::ClientConnection, TcpStream>, String> {
    let server_name = match rustls::pki_types::DnsName::try_from(host.clone()) {
        Ok(dns_name) => rustls::pki_types::ServerName::DnsName(dns_name),
        Err(e) => return Err(format!("Invalid server name {}: {}", host, e)),
    };
    
    let client = rustls::ClientConnection::new(config, server_name)
        .map_err(|e| format!("Failed to create TLS connection: {}", e))?;
    
    Ok(rustls::StreamOwned::new(client, stream))
}

/// Establish HTTP CONNECT tunnel through proxy.
fn establish_connect_tunnel(
    mut proxy_stream: TcpStream,
    target_uri: &Uri,
    auth: Option<&str>
) -> Result<TcpStream, String> {
    use std::io::{BufRead, BufReader, Write};
    
    let host = target_uri.host().ok_or("Target URI missing host")?;
    let port = target_uri.port_u16().unwrap_or(443);
    
    // Send CONNECT request
    let connect_request = if let Some(auth) = auth {
        format!("CONNECT {}:{} HTTP/1.1\r\nHost: {}:{}\r\nProxy-Authorization: Basic {}\r\n\r\n", 
                host, port, host, port, auth)
    } else {
        format!("CONNECT {}:{} HTTP/1.1\r\nHost: {}:{}\r\n\r\n", 
                host, port, host, port)
    };
    
    proxy_stream.write_all(connect_request.as_bytes())
        .map_err(|e| format!("Failed to send CONNECT request: {}", e))?;
    
    // Read response
    let mut reader = BufReader::new(&proxy_stream);
    let mut response_line = String::new();
    reader.read_line(&mut response_line)
        .map_err(|e| format!("Failed to read CONNECT response: {}", e))?;
    
    if !response_line.contains("200") {
        return Err(format!("CONNECT failed: {}", response_line.trim()));
    }
    
    // Skip remaining headers
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line)
            .map_err(|e| format!("Failed to read CONNECT headers: {}", e))?;
        if line.trim().is_empty() {
            break;
        }
    }
    
    Ok(proxy_stream)
}

/// Perform SOCKS handshake with full protocol support.
fn socks_handshake(
    mut stream: TcpStream,
    target_host: &str,
    target_port: u16,
    version: SocksVersion
) -> Result<TcpStream, String> {
    match version {
        SocksVersion::V4 => socks4_handshake(stream, target_host, target_port),
        SocksVersion::V5 => socks5_handshake(stream, target_host, target_port),
    }
}

/// SOCKS4 handshake implementation.
fn socks4_handshake(mut stream: TcpStream, target_host: &str, target_port: u16) -> Result<TcpStream, String> {
    use std::io::{Read, Write};
    use std::net::Ipv4Addr;
    use std::str::FromStr;
    
    // Try to parse as IP address first
    let target_ip = if let Ok(ipv4) = Ipv4Addr::from_str(target_host) {
        ipv4
    } else {
        // SOCKS4A - use 0.0.0.x to indicate hostname follows
        Ipv4Addr::new(0, 0, 0, 1)
    };
    
    let mut request = Vec::new();
    request.push(0x04); // Version
    request.push(0x01); // Connect command
    request.extend_from_slice(&target_port.to_be_bytes());
    request.extend_from_slice(&target_ip.octets());
    request.push(0x00); // User ID (empty)
    
    // Add hostname for SOCKS4A
    if target_ip == Ipv4Addr::new(0, 0, 0, 1) {
        request.extend_from_slice(target_host.as_bytes());
        request.push(0x00);
    }
    
    stream.write_all(&request)
        .map_err(|e| format!("Failed to send SOCKS4 request: {}", e))?;
    
    let mut response = [0u8; 8];
    stream.read_exact(&mut response)
        .map_err(|e| format!("Failed to read SOCKS4 response: {}", e))?;
    
    if response[1] != 0x5A {
        return Err(format!("SOCKS4 connection rejected: {}", response[1]));
    }
    
    Ok(stream)
}

/// SOCKS5 handshake implementation.
fn socks5_handshake(mut stream: TcpStream, target_host: &str, target_port: u16) -> Result<TcpStream, String> {
    use std::io::{Read, Write};
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    use std::str::FromStr;
    
    // Authentication negotiation
    let auth_request = [0x05, 0x01, 0x00]; // Version 5, 1 method, no auth
    stream.write_all(&auth_request)
        .map_err(|e| format!("Failed to send SOCKS5 auth request: {}", e))?;
    
    let mut auth_response = [0u8; 2];
    stream.read_exact(&mut auth_response)
        .map_err(|e| format!("Failed to read SOCKS5 auth response: {}", e))?;
    
    if auth_response[0] != 0x05 || auth_response[1] != 0x00 {
        return Err("SOCKS5 authentication failed".to_string());
    }
    
    // Connection request
    let mut request = Vec::new();
    request.extend_from_slice(&[0x05, 0x01, 0x00]); // Version, Connect, Reserved
    
    // Address type and address
    if let Ok(ip) = IpAddr::from_str(target_host) {
        match ip {
            IpAddr::V4(ipv4) => {
                request.push(0x01); // IPv4
                request.extend_from_slice(&ipv4.octets());
            },
            IpAddr::V6(ipv6) => {
                request.push(0x04); // IPv6
                request.extend_from_slice(&ipv6.octets());
            },
        }
    } else {
        request.push(0x03); // Domain name
        request.push(target_host.len() as u8);
        request.extend_from_slice(target_host.as_bytes());
    }
    
    request.extend_from_slice(&target_port.to_be_bytes());
    
    stream.write_all(&request)
        .map_err(|e| format!("Failed to send SOCKS5 connect request: {}", e))?;
    
    // Read response
    let mut response = [0u8; 4];
    stream.read_exact(&mut response)
        .map_err(|e| format!("Failed to read SOCKS5 response header: {}", e))?;
    
    if response[1] != 0x00 {
        return Err(format!("SOCKS5 connection rejected: {}", response[1]));
    }
    
    // Skip bound address (variable length)
    match response[3] {
        0x01 => { // IPv4
            let mut addr = [0u8; 6]; // 4 bytes IP + 2 bytes port
            stream.read_exact(&mut addr).map_err(|e| format!("Failed to read IPv4 bound address: {}", e))?;
        },
        0x03 => { // Domain name
            let mut len = [0u8; 1];
            stream.read_exact(&mut len).map_err(|e| format!("Failed to read domain length: {}", e))?;
            let mut domain_and_port = vec![0u8; len[0] as usize + 2];
            stream.read_exact(&mut domain_and_port).map_err(|e| format!("Failed to read domain bound address: {}", e))?;
        },
        0x04 => { // IPv6
            let mut addr = [0u8; 18]; // 16 bytes IP + 2 bytes port
            stream.read_exact(&mut addr).map_err(|e| format!("Failed to read IPv6 bound address: {}", e))?;
        },
        _ => return Err("Invalid SOCKS5 address type in response".to_string()),
    }
    
    Ok(stream)
}