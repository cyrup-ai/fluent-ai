//! TLS connection implementations for Native TLS and Rustls
//!
//! Feature-gated TLS connection wrappers with I/O trait implementations
//! for both native-tls and rustls backends.

use std::net::{SocketAddr, TcpStream};

#[cfg(feature = "default-tls")]
use native_tls_crate as native_tls;
#[cfg(feature = "__rustls")]
use rustls;

use super::super::types::ConnectionTrait;

#[cfg(feature = "default-tls")]
/// Native TLS connection implementation
pub struct NativeTlsConnection {
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
pub struct RustlsConnection {
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
