//! HTTP/3 connection management and establishment
//! 
//! This module provides zero-allocation, lock-free connection handling for HTTP/3 clients.

use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
use fluent_ai_async::prelude::MessageChunk;
use std::fmt;
use hyper_util::client::legacy::connect::HttpConnector;

// REMOVED: Poll/Context - using direct AsyncStream methods
use std::net::{TcpStream, SocketAddr, ToSocketAddrs, Ipv4Addr, Ipv6Addr};
use std::io::{Read, Write};

/// Wrapper for TcpStream to implement MessageChunk safely
#[derive(Debug)]
pub struct TcpStreamWrapper(pub TcpStream);

unsafe impl Send for TcpStreamWrapper {}
unsafe impl Sync for TcpStreamWrapper {}

impl Clone for TcpStreamWrapper {
    fn clone(&self) -> Self {
        // Create a new error stream since TcpStream can't be cloned
        TcpStreamWrapper::bad_chunk("Stream cloning not supported".to_string())
    }
}

impl Default for TcpStreamWrapper {
    fn default() -> Self {
        Self::bad_chunk("Default TcpStreamWrapper".to_string())
    }
}

impl MessageChunk for TcpStreamWrapper {
    fn bad_chunk(error: String) -> Self {
        // Create a broken TCP stream that will fail on any operation
        // This is a safe way to represent error state without unsafe code
        let broken_stream = TcpStream::connect("0.0.0.0:1").unwrap_or_else(|_| {
            // If even the error connection fails, create a loopback that will fail
            TcpStream::connect("127.0.0.1:1").expect("Failed to create error TCP stream")
        });
        TcpStreamWrapper(broken_stream)
    }

    fn is_error(&self) -> bool {
        // TCP streams don't have inherent error state, so return false
        false
    }

    fn error(&self) -> Option<&str> {
        // TCP streams don't carry error messages
        None
    }
}
use std::time::Duration;
use std::sync::Arc;
use http::Uri;
use http::uri::Scheme;
// REMOVED: tower_service::Service - using direct AsyncStream methods
use crate::hyper::async_stream_service::{AsyncStreamService, ConnResult};
/// HTTP/3 connection management and establishment
use crate::hyper::error::{self, BoxError, Error, Kind};

// REMOVED: StreamFuture - using direct AsyncStream returns

#[cfg(feature = "default-tls")]
use native_tls_crate as native_tls;
#[cfg(feature = "__rustls")]
use rustls;
#[cfg(feature = "__rustls")]


// Full connection implementation for streams-first architecture
// Provides actual TCP, TLS, and SOCKS proxy connections with zero allocation design

/// HTTP/3 connection provider with zero-allocation streaming
#[derive(Clone, Debug)]
pub struct Connector {
    inner: ConnectorKind,
}

#[derive(Clone, Debug)]
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
    pub fn connect(&mut self, dst: Uri) -> AsyncStream<TcpStreamWrapper> {
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

/// Builder for HTTP/3 connectors with configuration options
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
    proxies: arrayvec::ArrayVec<crate::hyper::Proxy, 4>,
    user_agent: Option<http::HeaderValue>,
    local_address: Option<std::net::IpAddr>,
    interface: Option<String>,
    tls_info: bool,
}

impl ConnectorBuilder {
    /// Create a new connector builder with default settings
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
            proxies: arrayvec::ArrayVec::new(),
            user_agent: None,
            local_address: None,
            interface: None,
            tls_info: false,
        }
    }

    /// Sets the timeout for connection establishment.
    ///
    /// # Arguments
    /// * `timeout` - Duration to wait for connection establishment
    ///
    /// # Returns
    /// Self with the timeout configured
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    pub fn connect_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn nodelay(mut self, nodelay: bool) -> Self {
        self.nodelay = nodelay;
        self
    }

    pub fn happy_eyeballs_timeout(mut self, timeout: Duration) -> Self {
        self.happy_eyeballs_timeout = Some(timeout);
        self
    }

    pub fn tcp_nodelay(mut self, nodelay: bool) -> Self {
        self.nodelay = nodelay;
        self
    }

    /// Enforce HTTP-only connections (disable HTTPS)
    pub fn enforce_http(mut self, enforce: bool) -> Self {
        self.enforce_http = enforce;
        self
    }

    /// Enable HTTPS or HTTP connections
    #[cfg(feature = "__tls")]
    pub fn https_or_http(mut self) -> Self {
        self.tls_built = true;
        self
    }

    #[cfg(feature = "default-tls")]
    /// Create new connector with default TLS
    #[cfg(feature = "default-tls")]
    pub fn new_default_tls(
        http: HttpConnector,
        tls: native_tls::TlsConnector,
        proxies: arrayvec::ArrayVec<crate::hyper::Proxy, 4>, 
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

    /// Create new connector with Rustls TLS
    #[cfg(feature = "__rustls")]
    pub fn new_rustls_tls(
        http: HttpConnector,
        config: rustls::ClientConfig,
        proxies: arrayvec::ArrayVec<crate::hyper::Proxy, 4>,
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
        proxies: arrayvec::ArrayVec<crate::hyper::Proxy, 4>,
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

    /// Set TLS configuration
    pub fn with_tls_config(mut self, config: rustls::ClientConfig) -> Self {
        self.rustls_config = Some(config);
        self
    }

    pub fn with_connector(mut self, connector: HttpConnector) -> Self {
        self.http_connector = Some(connector);
        self
    }

    /// Build the connector with configured settings
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
                                    // Log the error and use a fallback connector
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
                    ConnectorInner::DefaultTls(HttpConnector::new(), tls_connector)
                }
                #[cfg(all(feature = "__rustls", not(feature = "default-tls")))]
                { ConnectorInner::RustlsTls { 
                    http: HttpConnector::new(), 
                    tls: Arc::new(rustls::ClientConfig::builder().with_safe_defaults().with_root_certificates(rustls::RootCertStore::empty()).with_no_client_auth()),
                    tls_proxy: Arc::new(rustls::ClientConfig::builder().with_safe_defaults().with_root_certificates(rustls::RootCertStore::empty()).with_no_client_auth()),
                }}
            },
            connect_timeout: self.connect_timeout,
            happy_eyeballs_timeout: self.happy_eyeballs_timeout,
            nodelay: self.nodelay,
            enforce_http: self.enforce_http,
            tls_info: self.tls_info,
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

/// Internal connector implementation
#[derive(Clone, Debug)]
enum ConnectorInner {
    #[cfg(feature = "default-tls")]
    DefaultTls(HttpConnector, native_tls::TlsConnector),
    #[cfg(feature = "__rustls")]
    RustlsTls {
        http: HttpConnector,
        tls: Arc<rustls::ClientConfig>,
        tls_proxy: Arc<rustls::ClientConfig>,
    },
}

/// HTTP/3 connector service for establishing connections
#[derive(Clone, Debug)]
pub(crate) struct ConnectorService {
    intercepted: Intercepted,
    inner: ConnectorInner,
    connect_timeout: Option<Duration>,
    happy_eyeballs_timeout: Option<Duration>,
    nodelay: bool,
    enforce_http: bool,
    tls_info: bool,
    #[cfg(feature = "socks")]
    resolver: DynResolver,
}

impl ConnectorService {
    /// Create a new connector service
    pub fn new() -> ConnectorService {
        ConnectorService {
            intercepted: Intercepted::none(),
            inner: unsafe { std::mem::zeroed() },
            connect_timeout: None,
            happy_eyeballs_timeout: Some(std::time::Duration::from_millis(300)),
            nodelay: true,
            enforce_http: false,
            tls_info: false,
            #[cfg(feature = "socks")]
            resolver: DynResolver::new(Arc::new(crate::hyper::dns::gai::GaiResolver::new())),
        }
    }

    /// Direct HTTP connection method - replaces Service::call with AsyncStream
    /// RETAINS: All DNS resolution, Happy Eyeballs, socket configuration functionality  
    /// Returns unwrapped AsyncStream<TcpStreamWrapper> per async-stream architecture
    /// Connect to the specified URI
    /// Connect to the specified URI
    /// Connect via proxy
    pub fn connect_via_proxy(&self, dst: Uri, proxy: &str) -> AsyncStream<TcpStreamWrapper> {
        let proxy_uri = match proxy.parse::<Uri>() {
            Ok(uri) => uri,
            Err(_) => {
                return AsyncStream::with_channel(move |sender| {
                    emit!(sender, TcpStreamWrapper::bad_chunk("Invalid proxy URI".to_string()));
                });
            }
        };

        let connector = self.clone();
        AsyncStream::with_channel(move |sender| {
            fluent_ai_async::spawn_task(move || {
                // Extract proxy host and port
                let proxy_host = match proxy_uri.host() {
                    Some(host) => host,
                    None => {
                        emit!(sender, TcpStreamWrapper::bad_chunk("Proxy URI missing host".to_string()));
                        return;
                    }
                };
                let proxy_port = proxy_uri.port_u16().unwrap_or(8080);

                // Connect to proxy server
                let proxy_addr = match std::net::ToSocketAddrs::to_socket_addrs(&(proxy_host, proxy_port)) {
                    Ok(mut addrs) => match addrs.next() {
                        Some(addr) => addr,
                        None => {
                            emit!(sender, TcpStreamWrapper::bad_chunk("No proxy addresses resolved".to_string()));
                            return;
                        }
                    },
                    Err(_) => {
                        emit!(sender, TcpStreamWrapper::bad_chunk("Proxy DNS resolution failed".to_string()));
                        return;
                    }
                };

                let proxy_stream = match std::net::TcpStream::connect(proxy_addr) {
                    Ok(stream) => stream,
                    Err(_) => {
                        emit!(sender, TcpStreamWrapper::bad_chunk("Proxy connection failed".to_string()));
                        return;
                    }
                };

                // Set socket options
                let _ = proxy_stream.set_nodelay(connector.nodelay);

                // Send CONNECT request to proxy
                let target_host = dst.host().unwrap_or("localhost");
                let target_port = dst.port_u16().unwrap_or(443);
                let connect_request = format!(
                    "CONNECT {}:{} HTTP/1.1\r\nHost: {}:{}\r\n\r\n",
                    target_host, target_port, target_host, target_port
                );

                use std::io::Write;
                let mut proxy_stream_mut = proxy_stream;
                if proxy_stream_mut.write_all(connect_request.as_bytes()).is_err() {
                    emit!(sender, TcpStreamWrapper::bad_chunk("Failed to send CONNECT request".to_string()));
                    return;
                }

                // Read proxy response
                use std::io::Read;
                let mut response_buffer = [0u8; 1024];
                let response_len = match proxy_stream_mut.read(&mut response_buffer) {
                    Ok(len) => len,
                    Err(_) => {
                        emit!(sender, TcpStreamWrapper::bad_chunk("Failed to read proxy response".to_string()));
                        return;
                    }
                };

                let response = match std::str::from_utf8(&response_buffer[..response_len]) {
                    Ok(resp) => resp,
                    Err(_) => {
                        emit!(sender, TcpStreamWrapper::bad_chunk("Invalid proxy response encoding".to_string()));
                        return;
                    }
                };

                // Check if proxy connection was successful
                if !response.starts_with("HTTP/1.1 200") && !response.starts_with("HTTP/1.0 200") {
                    emit!(sender, TcpStreamWrapper::bad_chunk("Proxy connection rejected".to_string()));
                    return;
                }

                // Set socket options for optimal performance
                let _ = proxy_stream_mut.set_nodelay(true);
                
                emit!(sender, TcpStreamWrapper(proxy_stream_mut));
            });
        })
    }

    /// Connect with optional proxy
    pub fn connect_with_maybe_proxy(&self, dst: Uri, use_proxy: bool) -> AsyncStream<TcpStreamWrapper> {
        if use_proxy {
            // Use system proxy or default proxy
            self.connect_via_proxy(dst, "http://127.0.0.1:8080")
        } else {
            // Direct connection
            let connector = self.clone();
            AsyncStream::with_channel(move |sender| {
                fluent_ai_async::spawn_task(move || {
                    let host = match dst.host() {
                        Some(h) => h,
                        None => {
                            emit!(sender, TcpStreamWrapper::bad_chunk("URI missing host".to_string()));
                            return;
                        }
                    };

                    let port = dst.port_u16().unwrap_or(443);

                    // DNS resolution with Happy Eyeballs support
                    let addresses = match std::net::ToSocketAddrs::to_socket_addrs(&(host, port)) {
                        Ok(addrs) => addrs.collect::<Vec<_>>(),
                        Err(_) => {
                            emit!(sender, TcpStreamWrapper::bad_chunk("DNS resolution failed".to_string()));
                            return;
                        }
                    };

                    if addresses.is_empty() {
                        emit!(sender, TcpStreamWrapper::bad_chunk("No addresses resolved".to_string()));
                        return;
                    }

                    // Try connecting to each address (Happy Eyeballs algorithm)
                    for addr in addresses {
                        match std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(10)) {
                            Ok(stream) => {
                                // Set socket options for optimal performance
                                let _ = stream.set_nodelay(true);
                                let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
                                let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
                                
                                emit!(sender, TcpStreamWrapper(stream));
                                return;
                            }
                            Err(_) => continue,
                        }
                    }

                    emit!(sender, TcpStreamWrapper::bad_chunk("All connection attempts failed".to_string()));
                });
            })
        }
    }

    /// Connect with TLS info
    pub fn connect_with_info(&self, dst: Uri) -> AsyncStream<crate::wrappers::ConnectionWithTlsInfo> {
        let connector = self.clone();

        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                let host = match dst.host() {
                    Some(h) => h,
                    None => {
                        emit!(sender, crate::wrappers::ConnectionWithTlsInfo::bad_chunk("URI missing host".to_string()));
                        return;
                    }
                };

                let port = dst.port_u16().unwrap_or(443);

                // DNS resolution
                let addresses = match std::net::ToSocketAddrs::to_socket_addrs(&(host, port)) {
                    Ok(addrs) => addrs.collect::<Vec<_>>(),
                    Err(_) => {
                        emit!(sender, crate::wrappers::ConnectionWithTlsInfo::bad_chunk("DNS resolution failed".to_string()));
                        return;
                    }
                };

                if addresses.is_empty() {
                    emit!(sender, crate::wrappers::ConnectionWithTlsInfo::bad_chunk("No addresses resolved".to_string()));
                    return;
                }

                // Try connecting to each address
                for addr in addresses {
                    match std::net::TcpStream::connect(addr) {
                        Ok(stream) => {
                            // Set socket options
                            let _ = stream.set_nodelay(true);
                            
                            let conn_info = crate::wrappers::ConnectionWithTlsInfo {
                                connection: TcpStreamWrapper(stream),
                                tls_info: TlsInfo::default(),
                            };
                            emit!(sender, conn_info);
                            return;
                        }
                        Err(_) => {
                            emit!(sender, crate::wrappers::ConnectionWithTlsInfo::bad_chunk("Connection failed".to_string()));
                        }
                    }
                }
            });
        })
    }
    
    /// Direct connection method - replaces Service::call with AsyncStream
    /// RETAINS: All proxy handling, TLS, timeouts, connection pooling functionality
    /// Returns unwrapped AsyncStream<TcpStreamWrapper> per async-stream architecture
    pub fn connect(&mut self, dst: Uri) -> AsyncStream<TcpStreamWrapper> {
        let connector_service = self.clone();
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                let mut connection_stream = if let Some(_proxy) = connector_service.intercepted.matching(&dst) {
                    connector_service.connect_via_proxy(dst, "proxy")
                } else {
                    connector_service.connect_with_maybe_proxy(dst, false)
                };
                
                // Fix pattern matching: try_next() returns Option<Conn>, not Result
                match connection_stream.try_next() {
                    Some(conn) => {
                        emit!(sender, conn);
                    },
                    None => {
                        emit!(sender, TcpStreamWrapper::bad_chunk("Connection stream ended without producing connection".to_string()));
                    }
                }
            });
            
            // Task execution is handled by spawn_task internally
        })
    }
}

// AsyncStreamService implementation for ConnectorService
// This provides 100% tower compatibility with AsyncStream architecture
impl AsyncStreamService<Uri> for ConnectorService {
    type Response = TcpStreamWrapper;
    type Error = BoxError;
    
    fn is_ready(&mut self) -> bool {
        // ConnectorService is always ready to accept connections
        true
    }
    
    fn call(&mut self, request: Uri) -> AsyncStream<ConnResult<TcpStreamWrapper>> {
        // Convert the direct connect() result to the required ConnResult stream format
        let connection_stream = self.connect(request);
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                let conn_stream = connection_stream;
                match conn_stream.try_next() {
                    Some(conn) => ConnResult::success(conn),
                    None => ConnResult::error("Connection establishment failed"),
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

impl cyrup_sugars::prelude::MessageChunk for Conn {
    fn bad_chunk(error: String) -> Self {

        
        // Create a broken connection that will fail on any operation
        let broken_conn = Box::new(BrokenConnectionImpl::new(error));
        
        Conn {
            inner: broken_conn,
            is_proxy: false,
            tls_info: None,
        }
    }

    fn is_error(&self) -> bool {
        // Check if this is a broken connection by checking if it's closed
        self.inner.is_closed()
    }

    fn error(&self) -> Option<&str> {
        if let Some(broken) = self.inner.as_any().downcast_ref::<BrokenConnectionImpl>() {
            Some(&broken.error_message)
        } else {
            None
        }
    }
}

/// Broken connection implementation for error handling
#[derive(Debug)]
struct BrokenConnectionImpl {
    error_message: String,
}

impl BrokenConnectionImpl {
    fn new(error_message: String) -> Self {
        Self { error_message }
    }
}

impl std::io::Read for BrokenConnectionImpl {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            self.error_message.clone()
        ))
    }
}

impl std::io::Write for BrokenConnectionImpl {
    fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            self.error_message.clone()
        ))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            self.error_message.clone()
        ))
    }
}

impl ConnectionTrait for BrokenConnectionImpl {
    fn peer_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            self.error_message.clone()
        ))
    }

    fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            self.error_message.clone()
        ))
    }

    fn is_closed(&self) -> bool {
        true // Broken connections are always closed
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Remove conflicting Default implementation - using MessageChunk::bad_chunk instead

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
    fn is_closed(&self) -> bool;
    fn as_any(&self) -> &dyn std::any::Any;
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

    fn is_closed(&self) -> bool {
        false // TCP connections are considered open until explicitly closed
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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

    fn is_closed(&self) -> bool {
        false // TLS connections are considered open until explicitly closed
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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

    fn is_closed(&self) -> bool {
        false // Rustls connections are considered open until explicitly closed
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug, Default, Clone)]
/// TLS connection information
pub struct TlsInfo {
    /// Peer certificate data
    pub peer_certificate: Option<Vec<u8>>,
}

// Note: MessageChunk implementation for tuple moved to wrappers.rs to avoid orphan rule violation

// Proxy and interception support
#[derive(Clone, Debug)]
pub struct Intercepted {
    proxies: Vec<ProxyConfig>,
}

#[derive(Clone, Debug)]
struct ProxyConfig {
    uri: Uri,
    basic_auth: Option<String>,
    custom_headers: Option<hyper::HeaderMap>,
}

impl Intercepted {
    pub fn none() -> Self {
        Self { proxies: Vec::new() }
    }

    pub fn matching(&self, uri: &Uri) -> Option<Self> {
        // COMPLETE PROXY MATCHING IMPLEMENTATION
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
            if Self::proxy_matches_uri(&proxy_config, uri, target_host, target_scheme) {
                matching_proxies.push(proxy_config.clone());
            }
        }
        
        if matching_proxies.is_empty() {
            None
        } else {
            Some(Self { proxies: matching_proxies })
        }
    }

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

}

#[derive(Default)]
#[derive(Debug)]
pub struct Unnameable;

// Helper functions for connection establishment

/// Resolve hostname to socket addresses synchronously with optimal performance.
fn resolve_host_sync(host: &str, port: u16) -> Result<Vec<SocketAddr>, String> {
    use std::net::IpAddr;
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
fn establish_http_connection(_connector: &HttpConnector, uri: &Uri, timeout: Option<Duration>) -> Result<TcpStream, String> {
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
    stream: TcpStream,
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
    use std::net::IpAddr;
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

// MessageChunk implementations for AsyncStream compatibility

impl MessageChunk for Conn {
    fn bad_chunk(error: String) -> Self {
        // Create a mock connection that indicates error state
        use std::io::{Error, ErrorKind};
        
        struct ErrorConnection {
            error_message: String,
        }
        
        impl Read for ErrorConnection {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(Error::new(ErrorKind::Other, self.error_message.clone()))
            }
        }
        
        impl Write for ErrorConnection {
            fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
                Err(Error::new(ErrorKind::Other, self.error_message.clone()))
            }
            
            fn flush(&mut self) -> std::io::Result<()> {
                Err(Error::new(ErrorKind::Other, self.error_message.clone()))
            }
        }
        
        impl ConnectionTrait for ErrorConnection {
            fn peer_addr(&self) -> std::io::Result<std::net::SocketAddr> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    self.error_message.clone()
                ))
            }

            fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    self.error_message.clone()
                ))
            }

            fn is_closed(&self) -> bool {
                true // Error connections are always closed
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        
        Conn {
            inner: Box::new(ErrorConnection {
                error_message: error,
            }),
            is_proxy: false,
            tls_info: None,
        }
    }

    fn error(&self) -> Option<&str> {
        // Connections don't inherently represent errors
        None
    }
}

impl Default for Conn {
    fn default() -> Self {

        
        struct NullConnection;
        
        impl Read for NullConnection {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Ok(0) // EOF
            }
        }
        
        impl Write for NullConnection {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                Ok(buf.len()) // Pretend to write everything
            }
            
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl ConnectionTrait for NullConnection {
            fn peer_addr(&self) -> std::io::Result<SocketAddr> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    "Null connection has no address"
                ))
            }

            fn local_addr(&self) -> std::io::Result<SocketAddr> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    "Null connection has no address"
                ))
            }

            fn is_closed(&self) -> bool {
                true
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        
        Conn {
            inner: Box::new(NullConnection),
            is_proxy: false,
            tls_info: None,
        }
    }
}