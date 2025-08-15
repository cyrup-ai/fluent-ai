use fluent_ai_async::prelude::MessageChunk;
use http::header::{InvalidHeaderName, InvalidHeaderValue};
use thiserror::Error;

/// HTTP error types
#[derive(Error, Debug, Clone)]
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

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// HTTP status error
    #[error("HTTP {status}: {message}")]
    HttpStatus {
        /// HTTP status code
        status: u16,
        /// Error message describing the HTTP error
        message: String,
        /// Response body content
        body: String,
    },

    /// Request timeout error
    #[error("Request timeout: {message}")]
    Timeout {
        /// Error message describing the timeout
        message: String,
    },

    /// SSL/TLS error
    #[error("SSL/TLS error: {message}")]
    Tls {
        /// Error message describing the TLS issue
        message: String,
    },

    /// DNS resolution error
    #[error("DNS error: {message}")]
    Dns {
        /// Error message describing the DNS issue
        message: String,
    },

    /// Connection error
    #[error("Connection error: {message}")]
    Connection {
        /// Error message describing the connection issue
        message: String,
    },

    /// Request building error
    #[error("Request error: {message}")]
    Request {
        /// Error message describing the request building issue
        message: String,
    },

    /// Response parsing error
    #[error("Response error: {message}")]
    Response {
        /// Error message describing the response parsing issue
        message: String,
    },

    /// JSON parsing error
    #[error("JSON error: {message}")]
    Json {
        /// Error message describing the JSON parsing issue
        message: String,
    },

    /// URL parsing error
    #[error("URL error: {message}")]
    Url {
        /// Error message describing the URL parsing issue
        message: String,
    },

    /// Header error
    #[error("Header error: {message}")]
    InvalidHeader {
        /// Error message describing the header issue
        message: String,
        /// Header name that caused the error
        name: String,
        /// Header value that caused the error
        value: Option<String>,
        /// Source error that caused this header error
        error_source: Option<String>,
    },

    /// File system error
    #[error("File system error: {message}")]
    FileSystemError {
        /// Error message describing the file system issue
        message: String,
    },

    /// Body error
    #[error("Body error: {message}")]
    Body {
        /// Error message describing the body issue
        message: String,
    },

    /// Redirect error
    #[error("Redirect error: {message}")]
    Redirect {
        /// Error message describing the redirect issue
        message: String,
    },

    /// Compression error
    #[error("Compression error: {message}")]
    Compression {
        /// Error message describing the compression issue
        message: String,
    },

    /// Proxy error
    #[error("Proxy error: {message}")]
    Proxy {
        /// Error message describing the proxy issue
        message: String,
    },

    /// HTTP status code error
    #[error("HTTP status code error: {code}")]
    StatusCode {
        /// HTTP status code
        code: u16,
    },

    /// Builder error
    #[error("Builder error: {message}")]
    Builder {
        /// Error message describing the builder issue
        message: String,
    },

    /// Authentication error
    #[error("Authentication error: {message}")]
    Authentication {
        /// Error message describing the authentication issue
        message: String,
    },

    /// Rate limiting error
    #[error("Rate limited: {message}")]
    RateLimit {
        /// Error message describing the rate limiting
        message: String,
    },

    /// Server error
    #[error("Server error: {message}")]
    Server {
        /// Error message describing the server issue
        message: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Generic error
    #[error("Error: {0}")]
    Other(String),

    /// Stream error
    #[error("Stream error: {message}")]
    StreamError {
        /// Error message describing the stream issue
        message: String,
    },

    /// TLS error
    #[error("TLS error: {message}")]
    TlsError {
        /// Error message describing the TLS issue
        message: String,
    },

    /// Invalid URL error
    #[error("Invalid URL: {message}")]
    InvalidUrl {
        /// Error message describing the URL issue
        message: String,
    },

    /// Serialization error
    #[error("Serialization error: {message}")]
    SerializationError {
        /// Error message describing the serialization issue
        message: String,
    },

    /// Invalid response error
    #[error("Invalid response: {message}")]
    InvalidResponse {
        /// Error message describing the response issue
        message: String,
    },

    /// Deserialization error
    #[error("Deserialization error: {message}")]
    DeserializationError {
        /// Error message describing the deserialization issue
        message: String,
    },

    /// URL parse error
    #[error("URL parse error: {message}")]
    UrlParseError {
        /// Error message describing the URL parsing issue
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

    /// Generic error variant for compatibility
    #[error("Generic error: {message}")]
    Generic {
        /// Error message describing the generic issue
        message: String,
    },
}

impl HttpError {
    /// Create a new network error
    pub fn network<S: Into<String>>(message: S) -> Self {
        HttpError::NetworkError {
            message: message.into(),
        }
    }

    /// Create a new client error
    pub fn client<S: Into<String>>(message: S) -> Self {
        HttpError::ClientError {
            message: message.into(),
        }
    }

    /// Create a new configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        HttpError::Configuration(message.into())
    }

    /// Create a new HTTP status error
    pub fn http_status<S: Into<String>>(status: u16, message: S) -> Self {
        HttpError::HttpStatus {
            status,
            message: message.into(),
            body: String::new(),
        }
    }

    /// Create a new timeout error
    pub fn timeout<S: Into<String>>(message: S) -> Self {
        HttpError::Timeout {
            message: message.into(),
        }
    }

    /// Create a new TLS error
    pub fn tls<S: Into<String>>(message: S) -> Self {
        HttpError::Tls {
            message: message.into(),
        }
    }

    /// Create a new DNS error
    pub fn dns<S: Into<String>>(message: S) -> Self {
        HttpError::Dns {
            message: message.into(),
        }
    }

    /// Create a new connection error
    pub fn connection<S: Into<String>>(message: S) -> Self {
        HttpError::Connection {
            message: message.into(),
        }
    }

    /// Create a new request error
    pub fn request<S: Into<String>>(message: S) -> Self {
        HttpError::Request {
            message: message.into(),
        }
    }

    /// Create a new response error
    pub fn response<S: Into<String>>(message: S) -> Self {
        HttpError::Response {
            message: message.into(),
        }
    }

    /// Create a new builder error
    pub fn builder<S: Into<String>>(message: S) -> Self {
        HttpError::Generic {
            message: message.into(),
        }
    }

    /// Create a new JSON error
    pub fn json<S: Into<String>>(message: S) -> Self {
        HttpError::Json {
            message: message.into(),
        }
    }

    /// Create a new URL error
    pub fn url<S: Into<String>>(message: S) -> Self {
        HttpError::Url {
            message: message.into(),
        }
    }

    /// Create a new header error
    pub fn invalid_header<S: Into<String>, T: Into<String>>(name: S, message: T) -> Self {
        HttpError::InvalidHeader {
            name: name.into(),
            message: message.into(),
            value: None,
            error_source: None,
        }
    }

    /// Create a new body error
    pub fn body<S: Into<String>>(message: S) -> Self {
        HttpError::Body {
            message: message.into(),
        }
    }

    /// Create a new redirect error
    pub fn redirect<S: Into<String>>(message: S) -> Self {
        HttpError::Redirect {
            message: message.into(),
        }
    }

    /// Create a new compression error
    pub fn compression<S: Into<String>>(message: S) -> Self {
        HttpError::Compression {
            message: message.into(),
        }
    }

    /// Create a new proxy error
    pub fn proxy<S: Into<String>>(message: S) -> Self {
        HttpError::Proxy {
            message: message.into(),
        }
    }

    /// Create a new status code error
    pub fn status_code(code: u16) -> Self {
        HttpError::StatusCode { code }
    }

    /// Create a new authentication error
    pub fn authentication<S: Into<String>>(message: S) -> Self {
        HttpError::Authentication {
            message: message.into(),
        }
    }

    /// Create a new rate limit error
    pub fn rate_limit<S: Into<String>>(message: S) -> Self {
        HttpError::RateLimit {
            message: message.into(),
        }
    }

    /// Create a new server error
    pub fn server<S: Into<String>>(message: S) -> Self {
        HttpError::Server {
            message: message.into(),
        }
    }

    /// Create a new IO error
    pub fn io<S: Into<String>>(message: S) -> Self {
        HttpError::IoError(message.into())
    }

    /// Create a new generic error
    pub fn other<S: Into<String>>(message: S) -> Self {
        HttpError::Other(message.into())
    }

    /// Create a new stream error
    pub fn stream_error<S: Into<String>>(message: S) -> Self {
        HttpError::StreamError {
            message: message.into(),
        }
    }

    /// Create a new TLS error
    pub fn tls_error<S: Into<String>>(message: S) -> Self {
        HttpError::TlsError {
            message: message.into(),
        }
    }

    /// Create a new invalid URL error
    pub fn invalid_url<S: Into<String>>(message: S) -> Self {
        HttpError::InvalidUrl {
            message: message.into(),
        }
    }

    /// Create a new serialization error
    pub fn serialization_error<S: Into<String>>(message: S) -> Self {
        HttpError::SerializationError {
            message: message.into(),
        }
    }

    /// Create a new invalid response error
    pub fn invalid_response<S: Into<String>>(message: S) -> Self {
        HttpError::InvalidResponse {
            message: message.into(),
        }
    }

    /// Create a new deserialization error
    pub fn deserialization_error<S: Into<String>>(message: S) -> Self {
        HttpError::DeserializationError {
            message: message.into(),
        }
    }

    /// Create a new URL parse error
    pub fn url_parse_error<S: Into<String>>(message: S) -> Self {
        HttpError::UrlParseError {
            message: message.into(),
        }
    }

    /// Create a new download interrupted error
    pub fn download_interrupted<S: Into<String>>(message: S) -> Self {
        HttpError::DownloadInterrupted {
            message: message.into(),
        }
    }

    /// Create a new invalid content length error
    pub fn invalid_content_length<S: Into<String>>(message: S) -> Self {
        HttpError::InvalidContentLength {
            message: message.into(),
        }
    }

    /// Create a new chunk processing error
    pub fn chunk_processing_error<S: Into<String>>(message: S) -> Self {
        HttpError::ChunkProcessingError {
            message: message.into(),
        }
    }

    /// Create a new generic error
    pub fn generic<S: Into<String>>(message: S) -> Self {
        HttpError::Generic {
            message: message.into(),
        }
    }
}

impl From<InvalidHeaderName> for HttpError {
    fn from(error: InvalidHeaderName) -> Self {
        HttpError::InvalidHeader {
            message: error.to_string(),
            name: "unknown".to_string(),
            value: None,
            error_source: Some(error.to_string()),
        }
    }
}

impl From<InvalidHeaderValue> for HttpError {
    fn from(error: InvalidHeaderValue) -> Self {
        HttpError::InvalidHeader {
            message: error.to_string(),
            name: "unknown".to_string(),
            value: None,
            error_source: Some(error.to_string()),
        }
    }
}

impl From<crate::hyper::Error> for HttpError {
    fn from(error: crate::hyper::Error) -> Self {
        if error.is_timeout() {
            HttpError::Timeout {
                message: error.to_string(),
            }
        } else if error.is_status() {
            HttpError::HttpStatus {
                status: error.status().map_or(0, |s| s.as_u16()),
                message: error.to_string(),
                body: "".to_string(),
            }
        } else if error.is_connect() {
            HttpError::Connection {
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

#[cfg(feature = "__rustls")]
impl From<rustls::Error> for HttpError {
    fn from(error: rustls::Error) -> Self {
        HttpError::Tls {
            message: error.to_string(),
        }
    }
}

impl From<std::io::Error> for HttpError {
    fn from(error: std::io::Error) -> Self {
        HttpError::IoError(error.to_string())
    }
}

impl MessageChunk for HttpError {
    fn bad_chunk(error: String) -> Self {
        HttpError::NetworkError { message: error }
    }

    fn error(&self) -> Option<&str> {
        None // Return None to avoid lifetime issues with temporary string
    }
}

impl Default for HttpError {
    fn default() -> Self {
        HttpError::NetworkError {
            message: "Unknown error".to_string(),
        }
    }
}

/// HTTP result type for zero-allocation error handling
pub type HttpResult<T> = Result<T, HttpError>;

// Re-export internal hyper error types and helpers for legacy paths expecting `crate::error::*`
#[cfg(any(
    feature = "gzip",
    feature = "zstd",
    feature = "brotli",
    feature = "deflate",
))]
pub(crate) use crate::hyper::error::{BadScheme, TimedOut, body, builder, request};
