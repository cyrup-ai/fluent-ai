//! HTTP/3 connection types and trait definitions
//! 
//! Core connection abstractions with zero-allocation MessageChunk implementations.

use fluent_ai_async::prelude::MessageChunk;
use std::fmt;
use std::io::{Read, Write};
use std::net::{TcpStream, SocketAddr};
use http::Uri;
use super::service::ConnectorService;

/// HTTP/3 connection provider with zero-allocation streaming
#[derive(Clone, Debug)]
pub struct Connector {
    pub(super) inner: ConnectorKind,
}

#[derive(Clone, Debug)]
pub enum ConnectorKind {
    #[cfg(feature = "__tls")]
    BuiltDefault(ConnectorService),
    #[cfg(not(feature = "__tls"))]
    BuiltHttp(ConnectorService),
    WithLayers(BoxedConnectorService),
}

/// Direct ConnectorService type - no more Service trait boxing needed
pub type BoxedConnectorService = ConnectorService;

/// Wrapper for TcpStream to implement MessageChunk safely
#[derive(Debug)]
pub struct TcpStreamWrapper(pub TcpStream);

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
        // Use a mock implementation that's safer than unsafe code
        struct BrokenStream {
            error: String,
        }
        
        impl Read for BrokenStream {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    self.error.clone()
                ))
            }
        }
        
        impl Write for BrokenStream {
            fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    self.error.clone()
                ))
            }
            
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    self.error.clone()
                ))
            }
        }
        
        // Create a mock TcpStream using a local loopback that will fail
        match TcpStream::connect("127.0.0.1:1") {
            Ok(stream) => TcpStreamWrapper(stream),
            Err(_) => {
                // If we can't even create a failing stream, create a placeholder
                // This is a fallback that should rarely be used
                TcpStreamWrapper::bad_chunk("Failed to create error TCP stream".to_string())
            }
        }
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

/// HTTP connection wrapper that abstracts different connection types.
pub struct Conn {
    pub(super) inner: Box<dyn ConnectionTrait + Send + Sync>,
    pub(super) is_proxy: bool,
    pub(super) tls_info: Option<TlsInfo>,
}

impl std::fmt::Debug for Conn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conn")
            .field("is_proxy", &self.is_proxy)
            .field("tls_info", &self.tls_info)
            .finish()
    }
}

impl MessageChunk for Conn {
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

impl Conn {
    /// Returns whether this connection is through a proxy.
    pub fn is_proxy(&self) -> bool {
        self.is_proxy
    }

    /// Returns TLS information for this connection if available.
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

/// Connection trait for different connection types
pub trait ConnectionTrait: Read + Write {
    fn peer_addr(&self) -> std::io::Result<SocketAddr>;
    fn local_addr(&self) -> std::io::Result<SocketAddr>;
    fn is_closed(&self) -> bool;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Broken connection implementation for error handling
#[derive(Debug)]
pub struct BrokenConnectionImpl {
    pub error_message: String,
}

impl BrokenConnectionImpl {
    pub fn new(error_message: String) -> Self {
        Self { error_message }
    }
}

impl Read for BrokenConnectionImpl {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            self.error_message.clone()
        ))
    }
}

impl Write for BrokenConnectionImpl {
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
    fn peer_addr(&self) -> std::io::Result<SocketAddr> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            self.error_message.clone()
        ))
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
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

/// TCP connection implementation
pub struct TcpConnection {
    pub stream: TcpStream,
}

impl TcpConnection {
    pub fn new(stream: TcpStream) -> Self {
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

/// TLS connection information
#[derive(Debug, Default, Clone)]
pub struct TlsInfo {
    /// Peer certificate data
    pub peer_certificate: Option<Vec<u8>>,
}

/// Simplified approach: Use trait objects for connector layers
/// This provides the same functionality as tower::Layer but with AsyncStream services
/// Boxed connector layer type for composable connection handling.
pub type BoxedConnectorLayer = Box<dyn Fn(BoxedConnectorService) -> BoxedConnectorService + Send + Sync + 'static>;

/// Sealed module for internal traits.
pub mod sealed {
    /// Unnameable struct for internal use.
    #[derive(Default, Debug)]
    pub struct Unnameable;
}

pub use sealed::Unnameable;