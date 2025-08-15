//! HTTP/3 connector service with zero-allocation connection establishment
//! 
//! Core service implementation providing TCP, TLS, and proxy connections with elite polling.

use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
use fluent_ai_async::prelude::MessageChunk;
use std::time::Duration;
use std::net::{TcpStream, SocketAddr};
use http::Uri;
use hyper_util::client::legacy::connect::HttpConnector;
use crate::hyper::error::BoxError;
use crate::hyper::async_stream_service::{AsyncStreamService, ConnResult};
use super::types::{TcpStreamWrapper, Conn, TlsInfo, ConnectionTrait, TcpConnection, BrokenConnectionImpl};
use super::proxy::Intercepted;

#[cfg(feature = "default-tls")]
use native_tls_crate as native_tls;
#[cfg(feature = "__rustls")]
use rustls;

/// Core connector service with zero-allocation streaming
#[derive(Clone, Debug)]
pub struct ConnectorService {
    pub(super) http: HttpConnector,
    #[cfg(feature = "default-tls")]
    pub(super) tls: Option<native_tls::TlsConnector>,
    #[cfg(feature = "__rustls")]
    pub(super) rustls_config: Option<std::sync::Arc<rustls::ClientConfig>>,
    pub(super) intercepted: Intercepted,
    pub(super) user_agent: Option<http::HeaderValue>,
    pub(super) local_address: Option<std::net::IpAddr>,
    pub(super) interface: Option<String>,
    pub(super) nodelay: bool,
    pub(super) connect_timeout: Option<Duration>,
    pub(super) happy_eyeballs_timeout: Option<Duration>,
    pub(super) tls_info: bool,
}

impl ConnectorService {
    /// Create new connector service with configuration
    pub fn new(
        http: HttpConnector,
        #[cfg(feature = "default-tls")]
        tls: Option<native_tls::TlsConnector>,
        #[cfg(feature = "__rustls")]
        rustls_config: Option<rustls::ClientConfig>,
        proxies: arrayvec::ArrayVec<crate::hyper::Proxy, 4>,
        user_agent: Option<http::HeaderValue>,
        local_address: Option<std::net::IpAddr>,
        interface: Option<String>,
        nodelay: bool,
        connect_timeout: Option<Duration>,
        happy_eyeballs_timeout: Option<Duration>,
        tls_info: bool,
    ) -> Result<Self, BoxError> {
        let intercepted = if proxies.is_empty() {
            Intercepted::none()
        } else {
            Intercepted::from_proxies(proxies)?
        };

        Ok(Self {
            http,
            #[cfg(feature = "default-tls")]
            tls,
            #[cfg(feature = "__rustls")]
            rustls_config: rustls_config.map(std::sync::Arc::new),
            intercepted,
            user_agent,
            local_address,
            interface,
            nodelay,
            connect_timeout,
            happy_eyeballs_timeout,
            tls_info,
        })
    }

    /// Connect with optional proxy handling
    pub fn connect_with_maybe_proxy(&self, dst: Uri, via_proxy: bool) -> AsyncStream<TcpStreamWrapper> {
        let connector_service = self.clone();
        let destination = dst.clone();
        
        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                let host = match destination.host() {
                    Some(h) => h,
                    None => {
                        emit!(sender, TcpStreamWrapper::bad_chunk("URI missing host".to_string()));
                        return;
                    }
                };

                let port = destination.port_u16().unwrap_or_else(|| {
                    match destination.scheme_str() {
                        Some("https") => 443,
                        Some("http") => 80,
                        _ => 80,
                    }
                });

                // Resolve addresses with zero allocation
                let addresses = match super::tcp::resolve_host_sync(host, port) {
                    Ok(addrs) => addrs,
                    Err(e) => {
                        emit!(sender, TcpStreamWrapper::bad_chunk(format!("DNS resolution failed: {}", e)));
                        return;
                    }
                };

                if addresses.is_empty() {
                    emit!(sender, TcpStreamWrapper::bad_chunk("No addresses resolved".to_string()));
                    return;
                }

                // Try connecting to each address with elite polling
                for addr in addresses {
                    match connector_service.connect_timeout {
                        Some(timeout) => {
                            match TcpStream::connect_timeout(&addr, timeout) {
                                Ok(mut stream) => {
                                    // Configure socket for optimal performance
                                    if connector_service.nodelay {
                                        let _ = stream.set_nodelay(true);
                                    }
                                    
                                    emit!(sender, TcpStreamWrapper(stream));
                                    return;
                                }
                                Err(_) => continue,
                            }
                        }
                        None => {
                            match TcpStream::connect(&addr) {
                                Ok(mut stream) => {
                                    if connector_service.nodelay {
                                        let _ = stream.set_nodelay(true);
                                    }
                                    
                                    emit!(sender, TcpStreamWrapper(stream));
                                    return;
                                }
                                Err(_) => continue,
                            }
                        }
                    }
                }

                emit!(sender, TcpStreamWrapper::bad_chunk("Failed to connect to any address".to_string()));
            });
        })
    }

    /// Connect via proxy with full SOCKS and HTTP CONNECT support
    pub fn connect_via_proxy(&self, dst: Uri, proxy_scheme: &str) -> AsyncStream<TcpStreamWrapper> {
        let connector_service = self.clone();
        let destination = dst.clone();
        let scheme = proxy_scheme.to_string();
        
        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                let proxy_config = match connector_service.intercepted.first_proxy() {
                    Some(config) => config,
                    None => {
                        emit!(sender, TcpStreamWrapper::bad_chunk("No proxy configuration available".to_string()));
                        return;
                    }
                };

                let proxy_uri = &proxy_config.uri;
                let proxy_host = match proxy_uri.host() {
                    Some(h) => h,
                    None => {
                        emit!(sender, TcpStreamWrapper::bad_chunk("Proxy URI missing host".to_string()));
                        return;
                    }
                };

                let proxy_port = proxy_uri.port_u16().unwrap_or(8080);

                // Connect to proxy server first
                let proxy_stream = match super::tcp::connect_to_address_list(
                    &[SocketAddr::new(
                        proxy_host.parse().unwrap_or_else(|_| std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1))),
                        proxy_port
                    )],
                    connector_service.connect_timeout
                ) {
                    Ok(stream) => stream,
                    Err(e) => {
                        emit!(sender, TcpStreamWrapper::bad_chunk(format!("Proxy connection failed: {}", e)));
                        return;
                    }
                };

                // Handle different proxy types
                let final_stream = match scheme.as_str() {
                    "http" | "https" => {
                        // HTTP CONNECT tunnel
                        match super::tcp::establish_connect_tunnel(
                            proxy_stream,
                            &destination,
                            proxy_config.basic_auth.as_deref()
                        ) {
                            Ok(stream) => stream,
                            Err(e) => {
                                emit!(sender, TcpStreamWrapper::bad_chunk(format!("CONNECT tunnel failed: {}", e)));
                                return;
                            }
                        }
                    }
                    "socks5" => {
                        // SOCKS5 proxy
                        let target_host = destination.host().unwrap_or("localhost");
                        let target_port = destination.port_u16().unwrap_or(80);
                        
                        match super::tcp::socks5_handshake(proxy_stream, target_host, target_port) {
                            Ok(stream) => stream,
                            Err(e) => {
                                emit!(sender, TcpStreamWrapper::bad_chunk(format!("SOCKS5 handshake failed: {}", e)));
                                return;
                            }
                        }
                    }
                    _ => {
                        emit!(sender, TcpStreamWrapper::bad_chunk(format!("Unsupported proxy scheme: {}", scheme)));
                        return;
                    }
                };

                // Configure final stream
                if connector_service.nodelay {
                    let _ = final_stream.set_nodelay(true);
                }

                emit!(sender, TcpStreamWrapper(final_stream));
            });
        })
    }

    /// Direct connection method - replaces Service::call with AsyncStream
    pub fn connect(&mut self, dst: Uri) -> AsyncStream<TcpStreamWrapper> {
        let connector_service = self.clone();
        
        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                let mut connection_stream = if let Some(_proxy) = connector_service.intercepted.matching(&dst) {
                    connector_service.connect_via_proxy(dst, "proxy")
                } else {
                    connector_service.connect_with_maybe_proxy(dst, false)
                };
                
                // Elite polling pattern - non-blocking stream consumption
                match connection_stream.try_next() {
                    Some(conn) => {
                        emit!(sender, conn);
                    },
                    None => {
                        emit!(sender, TcpStreamWrapper::bad_chunk("Connection stream ended without producing connection".to_string()));
                    }
                }
            });
        })
    }
}

// AsyncStreamService implementation for ConnectorService
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
            spawn_task(move || {
                match connection_stream.try_next() {
                    Some(conn) => {
                        emit!(sender, ConnResult::success(conn));
                    },
                    None => {
                        emit!(sender, ConnResult::error("Connection establishment failed"));
                    }
                }
            });
        })
    }
}

#[cfg(feature = "default-tls")]
/// Native TLS connection implementation
pub(super) struct NativeTlsConnection {
    pub stream: native_tls::TlsStream<TcpStream>,
}

#[cfg(feature = "default-tls")]
impl NativeTlsConnection {
    pub fn new(stream: native_tls::TlsStream<TcpStream>) -> Self {
        Self { stream }
    }
}

#[cfg(feature = "default-tls")]
impl std::io::Read for NativeTlsConnection {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.stream.read(buf)
    }
}

#[cfg(feature = "default-tls")]
impl std::io::Write for NativeTlsConnection {
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
/// Rustls TLS connection implementation
pub(super) struct RustlsConnection {
    pub stream: rustls::StreamOwned<rustls::ClientConnection, TcpStream>,
}

#[cfg(feature = "__rustls")]
impl RustlsConnection {
    pub fn new(stream: rustls::StreamOwned<rustls::ClientConnection, TcpStream>) -> Self {
        Self { stream }
    }
}

#[cfg(feature = "__rustls")]
impl std::io::Read for RustlsConnection {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.stream.read(buf)
    }
}

#[cfg(feature = "__rustls")]
impl std::io::Write for RustlsConnection {
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