//! HTTP connection abstractions and trait definitions
//!
//! Provides the core connection types with MessageChunk implementations
//! for error handling and connection management.

use std::io::{Read, Write};
use std::net::SocketAddr;

use fluent_ai_async::prelude::MessageChunk;

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
                    "Null connection has no address",
                ))
            }

            fn local_addr(&self) -> std::io::Result<SocketAddr> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    "Null connection has no address",
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
            self.error_message.clone(),
        ))
    }
}

impl Write for BrokenConnectionImpl {
    fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            self.error_message.clone(),
        ))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            self.error_message.clone(),
        ))
    }
}

impl ConnectionTrait for BrokenConnectionImpl {
    fn peer_addr(&self) -> std::io::Result<SocketAddr> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            self.error_message.clone(),
        ))
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            self.error_message.clone(),
        ))
    }

    fn is_closed(&self) -> bool {
        true // Broken connections are always closed
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
