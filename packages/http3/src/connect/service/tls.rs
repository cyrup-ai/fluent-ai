//! TLS connection implementations for Native TLS and Rustls
//!
//! Feature-gated TLS connection wrappers with I/O trait implementations
//! for both native-tls and rustls backends.

use std::net::SocketAddr;
use tokio::net::TcpStream;
use tokio_rustls::TlsStream;

#[cfg(feature = "default-tls")]
use native_tls_crate as native_tls;
#[cfg(feature = "__rustls")]
use rustls;

use super::super::types::ConnectionTrait;

#[cfg(feature = "default-tls")]
/// Native TLS connection implementation
#[derive(Debug)]
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
impl tokio::io::AsyncRead for NativeTlsConnection {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.stream).poll_read(cx, buf)
    }
}

#[cfg(feature = "default-tls")]
impl tokio::io::AsyncWrite for NativeTlsConnection {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<Result<usize, std::io::Error>> {
        std::pin::Pin::new(&mut self.stream).poll_write(cx, buf)
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        std::pin::Pin::new(&mut self.stream).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        std::pin::Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

#[cfg(feature = "default-tls")]
impl Unpin for NativeTlsConnection {}

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
#[derive(Debug)]
pub struct RustlsConnection {
    pub stream: TlsStream<TcpStream>,
}

#[cfg(feature = "__rustls")]
impl RustlsConnection {
    pub fn new(stream: TlsStream<TcpStream>) -> Self {
        Self { stream }
    }
}

#[cfg(feature = "__rustls")]
impl tokio::io::AsyncRead for RustlsConnection {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.stream).poll_read(cx, buf)
    }
}

#[cfg(feature = "__rustls")]
impl tokio::io::AsyncWrite for RustlsConnection {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<Result<usize, std::io::Error>> {
        std::pin::Pin::new(&mut self.stream).poll_write(cx, buf)
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        std::pin::Pin::new(&mut self.stream).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        std::pin::Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

#[cfg(feature = "__rustls")]
impl Unpin for RustlsConnection {}

#[cfg(feature = "__rustls")]
impl ConnectionTrait for RustlsConnection {
    fn peer_addr(&self) -> std::io::Result<SocketAddr> {
        self.stream.get_ref().0.peer_addr()
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.stream.get_ref().0.local_addr()
    }

    fn is_closed(&self) -> bool {
        false // Rustls connections are considered open until explicitly closed
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
