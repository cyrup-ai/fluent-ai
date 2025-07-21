//! HTTP error types and utilities

use thiserror::Error;

/// HTTP error types
#[derive(Error, Debug)]
pub enum HttpError {
    /// Network error
    #[error("Network error: {message}")]
    NetworkError {
        /// Error message describing the network issue
        message: String,
    },

    /// Client configuration error
    #[error("Client error: {message}")]
    ClientError {
        /// Error message describing the client configuration issue
        message: String,
    },

    /// HTTP status error
    #[error("HTTP {status}: {message}")]
    HttpStatus {
        /// HTTP status code
        status: u16,
        /// Error message describing the HTTP status
        message: String,
        /// Response body content
        body: String,
    },

    /// Timeout error
    #[error("Request timeout: {message}")]
    Timeout {
        /// Error message describing the timeout
        message: String,
    },

    /// Serialization error
    #[error("Serialization error: {message}")]
    SerializationError {
        /// Error message describing the serialization issue
        message: String,
    },

    /// Deserialization error
    #[error("Deserialization error: {message}")]
    DeserializationError {
        /// Error message describing the deserialization issue
        message: String,
    },

    /// URL parsing error
    #[error("URL parsing error: {message}")]
    UrlParseError {
        /// Error message describing the URL parsing issue
        message: String,
    },

    /// Invalid URL error
    #[error("Invalid URL '{url}': {message}")]
    InvalidUrl {
        /// The invalid URL
        url: String,
        /// Error message describing the URL issue
        message: String,
    },

    /// Invalid response error
    #[error("Invalid response: {message}")]
    InvalidResponse {
        /// Error message describing the response issue
        message: String,
    },

    /// TLS error
    #[error("TLS error: {message}")]
    TlsError {
        /// Error message describing the TLS/SSL issue
        message: String,
    },

    /// Download interrupted error
    #[error("Download interrupted: {message}")]
    DownloadInterrupted {
        /// Error message describing the download interruption
        message: String,
    },

    /// Invalid content length error
    #[error("Invalid content length: {message}")]
    InvalidContentLength {
        /// Error message describing the content length issue
        message: String,
    },

    /// Chunk processing error
    #[error("Chunk processing error: {message}")]
    ChunkProcessingError {
        /// Error message describing the chunk processing issue
        message: String,
    },

    /// Connection error
    #[error("Connection error: {message}")]
    ConnectionError {
        /// Error message describing the connection issue
        message: String,
    },

    /// Stream error
    #[error("Stream error: {message}")]
    StreamError {
        /// Error message describing the stream processing issue
        message: String,
    },

    /// Cache error
    #[error("Cache error: {message}")]
    CacheError {
        /// Error message describing the cache issue
        message: String,
    },

    /// Middleware error
    #[error("Middleware error: {message}")]
    MiddlewareError {
        /// Error message describing the middleware issue
        message: String,
    },

    /// Invalid header error
    #[error("Invalid header: {message}")]
    InvalidHeader {
        /// Error message describing the invalid header
        message: String,
    },

    /// Invalid body error
    #[error("Invalid body: {message}")]
    InvalidBody {
        /// Error message describing the invalid body
        message: String,
    },

    /// Too many redirects
    #[error("Too many redirects: {message}")]
    TooManyRedirects {
        /// Error message describing the redirect issue
        message: String,
    },

    /// DNS resolution error
    #[error("DNS resolution error: {message}")]
    DnsError {
        /// Error message describing the DNS resolution issue
        message: String,
    },

    /// Proxy error
    #[error("Proxy error: {message}")]
    ProxyError {
        /// Error message describing the proxy issue
        message: String,
    },

    /// I/O error
    #[error("I/O error: {message}")]
    IoError {
        /// Error message describing the I/O issue
        message: String,
    },

    /// Unknown error
    #[error("Unknown error: {message}")]
    Unknown {
        /// Error message describing the unknown issue
        message: String,
    },
}

impl HttpError {
    /// Create a network error
    pub fn network(message: impl Into<String>) -> Self {
        Self::NetworkError {
            message: message.into(),
        }
    }

    /// Create a client error
    pub fn client(message: impl Into<String>) -> Self {
        Self::ClientError {
            message: message.into(),
        }
    }

    /// Create an HTTP status error
    pub fn http_status(status: u16, message: impl Into<String>, body: impl Into<String>) -> Self {
        Self::HttpStatus {
            status,
            message: message.into(),
            body: body.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::Timeout {
            message: message.into(),
        }
    }

    /// Create a serialization error
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::SerializationError {
            message: message.into(),
        }
    }

    /// Create a deserialization error
    pub fn deserialization(message: impl Into<String>) -> Self {
        Self::DeserializationError {
            message: message.into(),
        }
    }

    /// Create a URL parsing error
    pub fn url_parse(message: impl Into<String>) -> Self {
        Self::UrlParseError {
            message: message.into(),
        }
    }

    /// Create a TLS error
    pub fn tls(message: impl Into<String>) -> Self {
        Self::TlsError {
            message: message.into(),
        }
    }

    /// Create a connection error
    pub fn connection(message: impl Into<String>) -> Self {
        Self::ConnectionError {
            message: message.into(),
        }
    }

    /// Create a stream error
    pub fn stream(message: impl Into<String>) -> Self {
        Self::StreamError {
            message: message.into(),
        }
    }

    /// Create a cache error
    pub fn cache(message: impl Into<String>) -> Self {
        Self::CacheError {
            message: message.into(),
        }
    }

    /// Create a middleware error
    pub fn middleware(message: impl Into<String>) -> Self {
        Self::MiddlewareError {
            message: message.into(),
        }
    }

    /// Create an invalid header error
    pub fn invalid_header(message: impl Into<String>) -> Self {
        Self::InvalidHeader {
            message: message.into(),
        }
    }

    /// Create an invalid body error
    pub fn invalid_body(message: impl Into<String>) -> Self {
        Self::InvalidBody {
            message: message.into(),
        }
    }

    /// Create a too many redirects error
    pub fn too_many_redirects(message: impl Into<String>) -> Self {
        Self::TooManyRedirects {
            message: message.into(),
        }
    }

    /// Create a DNS error
    pub fn dns(message: impl Into<String>) -> Self {
        Self::DnsError {
            message: message.into(),
        }
    }

    /// Create a proxy error
    pub fn proxy(message: impl Into<String>) -> Self {
        Self::ProxyError {
            message: message.into(),
        }
    }

    /// Create an I/O error
    pub fn io(message: impl Into<String>) -> Self {
        Self::IoError {
            message: message.into(),
        }
    }

    /// Create an unknown error
    pub fn unknown(message: impl Into<String>) -> Self {
        Self::Unknown {
            message: message.into(),
        }
    }

    /// Check if error is retriable
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            HttpError::NetworkError { .. }
                | HttpError::Timeout { .. }
                | HttpError::ConnectionError { .. }
                | HttpError::DnsError { .. }
                | HttpError::HttpStatus {
                    status: 429 | 502 | 503 | 504,
                    ..
                }
        )
    }

    /// Check if error is a client error (4xx)
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            HttpError::HttpStatus {
                status: 400..=499,
                ..
            }
        )
    }

    /// Check if error is a server error (5xx)
    pub fn is_server_error(&self) -> bool {
        matches!(
            self,
            HttpError::HttpStatus {
                status: 500..=599,
                ..
            }
        )
    }

    /// Check if error is a timeout
    pub fn is_timeout(&self) -> bool {
        matches!(self, HttpError::Timeout { .. })
    }

    /// Check if error is a network error
    pub fn is_network_error(&self) -> bool {
        matches!(self, HttpError::NetworkError { .. })
    }

    /// Check if error is a connection error
    pub fn is_connection_error(&self) -> bool {
        matches!(self, HttpError::ConnectionError { .. })
    }

    /// Get the error message
    pub fn message(&self) -> &str {
        match self {
            HttpError::NetworkError { message } => message,
            HttpError::ClientError { message } => message,
            HttpError::HttpStatus { message, .. } => message,
            HttpError::Timeout { message } => message,
            HttpError::SerializationError { message } => message,
            HttpError::DeserializationError { message } => message,
            HttpError::UrlParseError { message } => message,
            HttpError::InvalidUrl { message, .. } => message,
            HttpError::InvalidResponse { message } => message,
            HttpError::TlsError { message } => message,
            HttpError::DownloadInterrupted { message } => message,
            HttpError::InvalidContentLength { message } => message,
            HttpError::ChunkProcessingError { message } => message,
            HttpError::ConnectionError { message } => message,
            HttpError::StreamError { message } => message,
            HttpError::CacheError { message } => message,
            HttpError::MiddlewareError { message } => message,
            HttpError::InvalidHeader { message } => message,
            HttpError::InvalidBody { message } => message,
            HttpError::TooManyRedirects { message } => message,
            HttpError::DnsError { message } => message,
            HttpError::ProxyError { message } => message,
            HttpError::IoError { message } => message,
            HttpError::Unknown { message } => message,
        }
    }

    /// Get the HTTP status code if applicable
    pub fn status_code(&self) -> Option<u16> {
        match self {
            HttpError::HttpStatus { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// Get the response body if applicable
    pub fn body(&self) -> Option<&str> {
        match self {
            HttpError::HttpStatus { body, .. } => Some(body),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for HttpError {
    fn from(error: reqwest::Error) -> Self {
        if error.is_timeout() {
            HttpError::Timeout {
                message: error.to_string(),
            }
        } else if error.is_connect() {
            HttpError::ConnectionError {
                message: error.to_string(),
            }
        } else if error.is_request() {
            HttpError::ClientError {
                message: error.to_string(),
            }
        } else if error.is_decode() {
            HttpError::DeserializationError {
                message: error.to_string(),
            }
        } else {
            HttpError::NetworkError {
                message: error.to_string(),
            }
        }
    }
}

impl From<serde_json::Error> for HttpError {
    fn from(error: serde_json::Error) -> Self {
        if error.is_io() {
            HttpError::NetworkError {
                message: error.to_string(),
            }
        } else if error.is_syntax() || error.is_data() {
            HttpError::DeserializationError {
                message: error.to_string(),
            }
        } else {
            HttpError::SerializationError {
                message: error.to_string(),
            }
        }
    }
}

impl From<url::ParseError> for HttpError {
    fn from(error: url::ParseError) -> Self {
        HttpError::UrlParseError {
            message: error.to_string(),
        }
    }
}

impl From<http::header::InvalidHeaderValue> for HttpError {
    fn from(error: http::header::InvalidHeaderValue) -> Self {
        HttpError::InvalidHeader {
            message: error.to_string(),
        }
    }
}

impl From<http::header::InvalidHeaderName> for HttpError {
    fn from(error: http::header::InvalidHeaderName) -> Self {
        HttpError::InvalidHeader {
            message: error.to_string(),
        }
    }
}

impl From<std::io::Error> for HttpError {
    fn from(error: std::io::Error) -> Self {
        HttpError::IoError {
            message: error.to_string(),
        }
    }
}

/// Result type for HTTP operations
pub type HttpResult<T> = Result<T, HttpError>;
