//! HTTP/3 connection management and establishment
//! 
//! This module provides zero-allocation, lock-free connection handling for HTTP/3 clients.

pub mod builder;
pub mod proxy;
pub mod service;
pub mod tcp;
pub mod types;

// Re-export all public types for backward compatibility
pub use builder::ConnectorBuilder;
pub use proxy::{Intercepted, ProxyConfig, SocksVersion, SocksAuth, SocksConfig, HttpConnectConfig, ProxyBypass};
pub use service::ConnectorService;
pub use tcp::{
    resolve_host_sync, connect_to_address_list, happy_eyeballs_connect,
    configure_tcp_socket, configure_tcp_socket_inline, establish_http_connection,
    establish_connect_tunnel, socks_handshake, socks4_handshake, socks5_handshake,
};
pub use types::{
    Connector, ConnectorKind, BoxedConnectorService, TcpStreamWrapper, Conn, TlsInfo,
    ConnectionTrait, BrokenConnectionImpl, TcpConnection, BoxedConnectorLayer, Unnameable,
};

#[cfg(feature = "default-tls")]
pub use tcp::establish_native_tls_connection;
#[cfg(feature = "__rustls")]
pub use tcp::establish_rustls_connection;

// Direct connection method implementation for Connector
impl Connector {
    /// Direct connection method - replaces Service::call with AsyncStream
    /// RETAINS: All proxy handling, TLS, timeouts, connection pooling functionality
    /// Returns unwrapped AsyncStream<TcpStreamWrapper> per async-stream architecture
    pub fn connect(&mut self, dst: http::Uri) -> fluent_ai_async::AsyncStream<TcpStreamWrapper> {
        match &mut self.inner {
            types::ConnectorKind::WithLayers(s) => s.connect(dst),
            #[cfg(feature = "__tls")]
            types::ConnectorKind::BuiltDefault(s) => s.connect(dst),
            #[cfg(not(feature = "__tls"))]
            types::ConnectorKind::BuiltHttp(s) => s.connect(dst),
        }
    }
}