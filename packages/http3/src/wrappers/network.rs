//! Network connection wrappers for implementing MessageChunk trait
//! Includes TCP streams, upgraded connections, DNS resolution, and connection management

use std::net::TcpStream;

use fluent_ai_async::prelude::MessageChunk;

/// Wrapper for DNS resolution results
#[derive(Debug, Clone, Default)]
pub struct DnsWrapper {
    pub addresses: Vec<std::net::SocketAddr>,
    pub error_message: Option<String>,
}

impl MessageChunk for DnsWrapper {
    fn bad_chunk(error: String) -> Self {
        Self {
            addresses: Vec::new(),
            error_message: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error_message.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

/// Wrapper for socket addresses to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct SocketAddrWrapper {
    pub address: Option<std::net::SocketAddr>,
    pub error_message: Option<String>,
}

impl MessageChunk for SocketAddrWrapper {
    fn bad_chunk(error: String) -> Self {
        Self {
            address: None,
            error_message: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error_message.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

/// Wrapper for TcpStream to implement MessageChunk
#[derive(Debug)]
pub struct TcpStreamWrapper {
    pub stream: Option<TcpStream>,
    pub error_message: Option<String>,
}

impl MessageChunk for TcpStreamWrapper {
    fn bad_chunk(error: String) -> Self {
        TcpStreamWrapper {
            stream: None,
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl From<TcpStream> for TcpStreamWrapper {
    fn from(stream: TcpStream) -> Self {
        TcpStreamWrapper {
            stream: Some(stream),
            error_message: None,
        }
    }
}

impl From<TcpStreamWrapper> for Option<TcpStream> {
    fn from(wrapper: TcpStreamWrapper) -> Self {
        wrapper.stream
    }
}

impl Default for TcpStreamWrapper {
    fn default() -> Self {
        Self::bad_chunk("Default TcpStream".to_string())
    }
}

/// Wrapper for Upgraded to implement MessageChunk
#[derive(Debug)]
pub struct UpgradedWrapper {
    pub upgraded: Option<crate::hyper::async_impl::upgrade::Upgraded>,
    pub error_message: Option<String>,
}

impl MessageChunk for UpgradedWrapper {
    fn bad_chunk(error: String) -> Self {
        UpgradedWrapper {
            upgraded: None,
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl Default for UpgradedWrapper {
    fn default() -> Self {
        Self::bad_chunk("Default UpgradedWrapper".to_string())
    }
}

impl From<crate::hyper::async_impl::upgrade::Upgraded> for UpgradedWrapper {
    fn from(upgraded: crate::hyper::async_impl::upgrade::Upgraded) -> Self {
        UpgradedWrapper {
            upgraded: Some(upgraded),
            error_message: None,
        }
    }
}

/// Wrapper for Connection to implement MessageChunk
#[derive(Debug)]
pub struct ConnWrapper {
    pub inner: Option<crate::hyper::connect::Conn>,
    pub error_message: Option<String>,
}

impl MessageChunk for ConnWrapper {
    fn bad_chunk(error: String) -> Self {
        Self {
            inner: None,
            error_message: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.inner.is_none() || self.error_message.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl From<crate::hyper::connect::Conn> for ConnWrapper {
    fn from(conn: crate::hyper::connect::Conn) -> Self {
        Self {
            inner: Some(conn),
            error_message: None,
        }
    }
}

/// Wrapper for connection with TLS info to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct ConnectionWithTlsInfo {
    pub connection: crate::hyper::connect::TcpStreamWrapper,
    pub tls_info: crate::hyper::connect::TlsInfo,
}

impl MessageChunk for ConnectionWithTlsInfo {
    fn is_error(&self) -> bool {
        self.connection.is_error()
    }

    fn error(&self) -> Option<&str> {
        self.connection.error()
    }

    fn bad_chunk(error: String) -> Self {
        Self {
            connection: crate::hyper::connect::TcpStreamWrapper::bad_chunk(error),
            tls_info: crate::hyper::connect::TlsInfo::default(),
        }
    }
}

impl
    From<(
        crate::hyper::connect::TcpStreamWrapper,
        crate::hyper::connect::TlsInfo,
    )> for ConnectionWithTlsInfo
{
    fn from(
        (connection, tls_info): (
            crate::hyper::connect::TcpStreamWrapper,
            crate::hyper::connect::TlsInfo,
        ),
    ) -> Self {
        Self {
            connection,
            tls_info,
        }
    }
}

/// Wrapper for DNS resolution results
#[derive(Debug, Clone, Default)]
pub struct DnsResultWrapper {
    pub addrs: Vec<std::net::SocketAddr>,
    pub error_message: Option<String>,
}

impl MessageChunk for DnsResultWrapper {
    fn bad_chunk(error: String) -> Self {
        Self {
            addrs: Vec::new(),
            error_message: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.addrs.is_empty() || self.error_message.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}
