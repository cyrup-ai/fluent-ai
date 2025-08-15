//! Wrapper types for external types to implement local traits
//! Solves orphan rule violations for MessageChunk implementations

use std::net::TcpStream;
use std::ops::{Deref, DerefMut};

use bytes::Bytes;
use fluent_ai_async::prelude::MessageChunk;
use http::Response;
use http_body::Frame;

/// Wrapper for unit type () to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct UnitWrapper {
    pub error_message: Option<String>,
}

impl MessageChunk for UnitWrapper {
    fn bad_chunk(error: String) -> Self {
        UnitWrapper {
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl From<()> for UnitWrapper {
    fn from(_: ()) -> Self {
        UnitWrapper {
            error_message: None,
        }
    }
}

impl From<UnitWrapper> for () {
    fn from(_: UnitWrapper) -> Self {
        ()
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

/// Wrapper for String to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct StringWrapper {
    pub data: String,
    pub error_message: Option<String>,
}

impl MessageChunk for StringWrapper {
    fn bad_chunk(error: String) -> Self {
        StringWrapper {
            data: String::new(),
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl From<String> for StringWrapper {
    fn from(data: String) -> Self {
        StringWrapper {
            data,
            error_message: None,
        }
    }
}

impl From<StringWrapper> for String {
    fn from(wrapper: StringWrapper) -> Self {
        wrapper.data
    }
}

impl Deref for StringWrapper {
    type Target = String;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for StringWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Wrapper for Bytes to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct BytesWrapper {
    pub data: Bytes,
    pub error_message: Option<String>,
}

impl MessageChunk for BytesWrapper {
    fn bad_chunk(error: String) -> Self {
        BytesWrapper {
            data: Bytes::new(),
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl BytesWrapper {
    pub fn into_bytes(self) -> Bytes {
        self.data
    }
}

impl From<Bytes> for BytesWrapper {
    fn from(data: Bytes) -> Self {
        BytesWrapper {
            data,
            error_message: None,
        }
    }
}

impl From<BytesWrapper> for Bytes {
    fn from(wrapper: BytesWrapper) -> Self {
        wrapper.data
    }
}

impl Deref for BytesWrapper {
    type Target = Bytes;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for BytesWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Generic wrapper for any type to implement MessageChunk
#[derive(Debug, Clone)]
pub struct GenericWrapper<T> {
    pub data: Option<T>,
    pub error_message: Option<String>,
}

impl<T> Default for GenericWrapper<T> {
    fn default() -> Self {
        GenericWrapper {
            data: None,
            error_message: None,
        }
    }
}

impl<T> MessageChunk for GenericWrapper<T> {
    fn bad_chunk(error: String) -> Self {
        GenericWrapper {
            data: None,
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl<T> From<T> for GenericWrapper<T> {
    fn from(data: T) -> Self {
        GenericWrapper {
            data: Some(data),
            error_message: None,
        }
    }
}

/// Wrapper for http_body::Frame<Bytes> to implement MessageChunk + Default
#[derive(Debug)]
pub struct FrameWrapper(pub Frame<Bytes>);

impl Deref for FrameWrapper {
    type Target = Frame<Bytes>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for FrameWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Frame<Bytes>> for FrameWrapper {
    fn from(frame: Frame<Bytes>) -> Self {
        Self(frame)
    }
}

impl From<FrameWrapper> for Frame<Bytes> {
    fn from(wrapper: FrameWrapper) -> Self {
        wrapper.0
    }
}

impl Default for FrameWrapper {
    fn default() -> Self {
        Self(Frame::data(Bytes::new()))
    }
}

impl MessageChunk for FrameWrapper {
    fn bad_chunk(error: String) -> Self {
        Self(Frame::data(Bytes::from(format!("ERROR: {}", error))))
    }

    fn is_error(&self) -> bool {
        if let Some(data) = self.0.data_ref() {
            data.starts_with(b"ERROR:")
        } else {
            false
        }
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            if let Some(data) = self.0.data_ref() {
                std::str::from_utf8(data).ok()
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Wrapper for BoxBody to implement MessageChunk + Default
#[derive(Debug)]
pub struct BoxBodyWrapper(
    pub http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>,
);

impl Default for BoxBodyWrapper {
    fn default() -> Self {
        use http_body_util::BodyExt;
        Self(
            http_body_util::Empty::new()
                .map_err(|never| match never {})
                .boxed(),
        )
    }
}

impl MessageChunk for BoxBodyWrapper {
    fn bad_chunk(error: String) -> Self {
        use http_body_util::BodyExt;
        let body = http_body_util::Full::new(Bytes::from(format!("ERROR: {}", error)))
            .map_err(|never| match never {})
            .boxed();
        Self(body)
    }

    fn is_error(&self) -> bool {
        false // Cannot easily inspect BoxBody contents
    }

    fn error(&self) -> Option<&str> {
        None
    }
}

/// Wrapper for Result types to implement MessageChunk + Default  
#[derive(Debug, Clone)]
pub struct ResultWrapper<T, E>(pub Result<T, E>);

impl<T, E> Default for ResultWrapper<T, E>
where
    T: Default,
    E: Default,
{
    fn default() -> Self {
        Self(Ok(T::default()))
    }
}

impl<T, E> MessageChunk for ResultWrapper<T, E>
where
    T: Clone,
    E: std::fmt::Display,
{
    fn bad_chunk(error_message: String) -> Self {
        // This requires concrete error type - will be implemented per use case
        panic!("ResultWrapper::bad_chunk requires concrete error type")
    }

    fn is_error(&self) -> bool {
        self.0.is_err()
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some("Result contains error")
        } else {
            None
        }
    }
}

/// Wrapper for http::Response to implement MessageChunk
#[derive(Debug)]
pub struct ResponseWrapper<B>(pub Response<B>);

impl<B: Default> MessageChunk for ResponseWrapper<B> {
    fn bad_chunk(error: String) -> Self {
        use http::{Response, StatusCode};
        let response = Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header("content-type", "text/plain")
            .body(B::default())
            .unwrap_or_else(|_| {
                // Use minimal safe response if even the fallback fails
                match Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(B::default()) {
                    Ok(resp) => resp,
                    Err(_) => {
                        // Absolute fallback - create response directly without builder
                        let mut resp = Response::new(B::default());
                        *resp.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                        resp
                    }
                }
            });
        Self(response)
    }

    fn is_error(&self) -> bool {
        self.0.status().is_server_error() || self.0.status().is_client_error()
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            self.0.status().canonical_reason()
        } else {
            None
        }
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
    pub upgraded: Option<crate::hyper::Upgraded>,
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

impl From<crate::hyper::Upgraded> for UpgradedWrapper {
    fn from(upgraded: crate::hyper::Upgraded) -> Self {
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

// Note: MessageChunk implementation for HttpResponseChunk already exists in response/chunk.rs

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

impl From<crate::hyper::connect::Conn> for ConnWrapper {
    fn from(conn: crate::hyper::connect::Conn) -> Self {
        Self {
            inner: Some(conn),
            error_message: None,
        }
    }
}

/// Wrapper for ConnResult to implement MessageChunk
#[derive(Debug, Clone)]
pub struct ConnResultWrapper<T> {
    pub result: crate::hyper::async_stream_service::ConnResult<T>,
}

impl<T> MessageChunk for ConnResultWrapper<T> {
    fn bad_chunk(error: String) -> Self {
        Self {
            result: crate::hyper::async_stream_service::ConnResult::Error(error),
        }
    }

    fn is_error(&self) -> bool {
        matches!(
            self.result,
            crate::hyper::async_stream_service::ConnResult::Error(_)
                | crate::hyper::async_stream_service::ConnResult::Timeout
        )
    }

    fn error(&self) -> Option<&str> {
        match &self.result {
            crate::hyper::async_stream_service::ConnResult::Error(msg) => Some(msg),
            crate::hyper::async_stream_service::ConnResult::Timeout => Some("Connection timeout"),
            _ => None,
        }
    }
}

impl<T> Default for ConnResultWrapper<T> {
    fn default() -> Self {
        Self::bad_chunk("Default ConnResult".to_string())
    }
}

impl<T> From<crate::hyper::async_stream_service::ConnResult<T>> for ConnResultWrapper<T> {
    fn from(result: crate::hyper::async_stream_service::ConnResult<T>) -> Self {
        Self { result }
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

/// Wrapper for HTTP chunks
#[derive(Debug, Clone)]
pub struct HttpChunkWrapper {
    pub chunk: crate::stream::HttpChunk,
}

impl MessageChunk for HttpChunkWrapper {
    fn bad_chunk(error: String) -> Self {
        Self {
            chunk: crate::stream::HttpChunk::Error(error),
        }
    }

    fn is_error(&self) -> bool {
        matches!(self.chunk, crate::stream::HttpChunk::Error(_))
    }

    fn error(&self) -> Option<&str> {
        match &self.chunk {
            crate::stream::HttpChunk::Error(_) => Some("HTTP chunk error"),
            _ => None,
        }
    }
}

impl Default for HttpChunkWrapper {
    fn default() -> Self {
        Self {
            chunk: crate::stream::HttpChunk::Body(bytes::Bytes::new()),
        }
    }
}

impl From<crate::stream::HttpChunk> for HttpChunkWrapper {
    fn from(chunk: crate::stream::HttpChunk) -> Self {
        Self { chunk }
    }
}

/// Wrapper for download chunks
#[derive(Debug, Clone)]
pub struct DownloadChunkWrapper {
    pub chunk: crate::stream::DownloadChunk,
}

impl MessageChunk for DownloadChunkWrapper {
    fn bad_chunk(error: String) -> Self {
        Self {
            chunk: crate::stream::DownloadChunk {
                data: bytes::Bytes::new(),
                chunk_number: 0,
                total_size: None,
                bytes_downloaded: 0,
                error_message: Some(error),
            },
        }
    }

    fn is_error(&self) -> bool {
        self.chunk.error_message.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.chunk.error_message.as_deref()
    }
}

impl Default for DownloadChunkWrapper {
    fn default() -> Self {
        Self {
            chunk: crate::stream::DownloadChunk::default(),
        }
    }
}

impl From<crate::stream::DownloadChunk> for DownloadChunkWrapper {
    fn from(chunk: crate::stream::DownloadChunk) -> Self {
        Self { chunk }
    }
}

impl<B> Default for ResponseWrapper<B>
where
    B: Default,
{
    fn default() -> Self {
        Self(Response::new(B::default()))
    }
}

impl<B> From<Response<B>> for ResponseWrapper<B> {
    fn from(response: Response<B>) -> Self {
        Self(response)
    }
}

impl<B> From<ResponseWrapper<B>> for Response<B> {
    fn from(wrapper: ResponseWrapper<B>) -> Self {
        wrapper.0
    }
}

impl<B> Deref for ResponseWrapper<B> {
    type Target = Response<B>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B> DerefMut for ResponseWrapper<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Wrapper for Vec<SocketAddr> to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct SocketAddrListWrapper(pub Vec<std::net::SocketAddr>);

impl MessageChunk for SocketAddrListWrapper {
    fn bad_chunk(_error: String) -> Self {
        Self(Vec::new())
    }

    fn error(&self) -> Option<&str> {
        if self.0.is_empty() {
            Some("Empty socket address list")
        } else {
            None
        }
    }
}

impl From<Vec<std::net::SocketAddr>> for SocketAddrListWrapper {
    fn from(addrs: Vec<std::net::SocketAddr>) -> Self {
        Self(addrs)
    }
}

impl From<SocketAddrListWrapper> for Vec<std::net::SocketAddr> {
    fn from(wrapper: SocketAddrListWrapper) -> Self {
        wrapper.0
    }
}

impl Deref for SocketAddrListWrapper {
    type Target = Vec<std::net::SocketAddr>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SocketAddrListWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Wrapper for tuple types used in H3 connections
#[derive(Debug, Clone)]
pub struct TupleWrapper<T, U>(pub T, pub U);

impl<T, U> MessageChunk for TupleWrapper<T, U>
where
    T: Default,
    U: Default,
{
    fn bad_chunk(_error: String) -> Self {
        Self(T::default(), U::default())
    }

    fn error(&self) -> Option<&str> {
        None
    }
}

impl<T, U> From<(T, U)> for TupleWrapper<T, U> {
    fn from((t, u): (T, U)) -> Self {
        Self(t, u)
    }
}

impl<T, U> From<TupleWrapper<T, U>> for (T, U) {
    fn from(wrapper: TupleWrapper<T, U>) -> Self {
        (wrapper.0, wrapper.1)
    }
}

impl<T, U> Default for TupleWrapper<T, U>
where
    T: Default,
    U: Default,
{
    fn default() -> Self {
        Self(T::default(), U::default())
    }
}

/// Generic wrapper for Option<T> to implement MessageChunk
#[derive(Debug, Clone)]
pub struct OptionWrapper<T>(pub Option<T>);

impl<T> Default for OptionWrapper<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T> MessageChunk for OptionWrapper<T> {
    fn bad_chunk(_error: String) -> Self {
        Self(None)
    }

    fn error(&self) -> Option<&str> {
        if self.0.is_none() {
            Some("No value available")
        } else {
            None
        }
    }
}

impl<T> From<Option<T>> for OptionWrapper<T> {
    fn from(option: Option<T>) -> Self {
        Self(option)
    }
}

impl<T> From<OptionWrapper<T>> for Option<T> {
    fn from(wrapper: OptionWrapper<T>) -> Self {
        wrapper.0
    }
}

impl<T> Deref for OptionWrapper<T> {
    type Target = Option<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for OptionWrapper<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
