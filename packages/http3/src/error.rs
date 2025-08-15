//! HTTP error types and utilities

use cyrup_sugars::prelude::MessageChunk;
use fluent_ai_async::prelude::MessageChunk as FluentMessageChunk;
use http::StatusCode;
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
        /// Error message describing the status error
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

    /// Runtime error
    #[error("Runtime error: {message}")]
    RuntimeError {
        /// Error message describing the runtime issue
        message: String,
    },

    /// Stream error
    #[error("Stream error: {message}")]
    StreamError {
        /// Error message describing the stream issue
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
        /// The invalid URL that caused the error
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
        /// Error message describing the TLS issue
        message: String,
    },

    /// Connection error
    #[error("Connection error: {message}")]
    ConnectionError {
        /// Error message describing the connection issue
        message: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Invalid header error
    #[error("Invalid header: {message}")]
    InvalidHeader {
        /// Error message describing the header issue
        message: String,
    },

    /// Custom error for middleware and other uses
    #[error("Custom error: {message}")]
    Custom {
        /// Custom error message
        message: String,
    },

    /// Error processing a response chunk during collection
    #[error("Chunk processing error: {source}")]
    ChunkProcessingError {
        /// The underlying JSON error that occurred
        source: std::sync::Arc<serde_json::Error>,
        /// The response body that failed to process
        body: Vec<u8>,
    },

    /// Download was interrupted
    #[error("Download interrupted: {message}")]
    DownloadInterrupted {
        /// Error message describing the interruption
        message: String,
    },

    /// Invalid content length
    #[error("Invalid content length: {message}")]
    InvalidContentLength {
        /// Error message describing the content length issue
        message: String,
    },

    /// A retryable error
    #[error("Retryable error: {message}")]
    Retryable {
        /// Error message for the retryable error
        message: String,
    },

    /// A non-retryable error
    #[error("Non-retryable error: {message}")]
    NonRetryable {
        /// Error message for the non-retryable error
        message: String,
    },

    /// Generic error with a simple message
    #[error("Error: {0}")]
    Generic(String),
}

impl HttpError {
    /// Returns the status code if the error is `HttpStatus`.
    pub fn status(&self) -> Option<StatusCode> {
        match self {
            HttpError::HttpStatus { status, .. } => StatusCode::from_u16(*status).ok(),
            _ => None,
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
                body: "".to_string(), // crate::hyper::Error doesn't expose the body
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

impl From<serde_urlencoded::ser::Error> for HttpError {
    fn from(error: serde_urlencoded::ser::Error) -> Self {
        HttpError::SerializationError {
            message: error.to_string(),
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

impl From<InvalidHeaderName> for HttpError {
    fn from(error: InvalidHeaderName) -> Self {
        HttpError::InvalidHeader {
            message: error.to_string(),
        }
    }
}

impl From<InvalidHeaderValue> for HttpError {
    fn from(error: InvalidHeaderValue) -> Self {
        HttpError::InvalidHeader {
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

impl FluentMessageChunk for HttpError {
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

/// Wrapper for HTTP results that implements required traits
#[derive(Debug, Clone)]
pub enum HttpResult<T> {
    Ok(T),
    Err(HttpError),
}

impl<T> MessageChunk for HttpResult<T>
where
    T: MessageChunk + Send + Default + 'static,
{
    fn bad_chunk(error: String) -> Self {
        HttpResult::Err(HttpError::bad_chunk(error))
    }

    fn is_error(&self) -> bool {
        matches!(self, HttpResult::Err(_))
    }

    fn error(&self) -> Option<&str> {
        match self {
            HttpResult::Ok(_) => None,
            HttpResult::Err(e) => e.error(),
        }
    }
}

impl<T> FluentMessageChunk for HttpResult<T>
where
    T: FluentMessageChunk + Send + Default + 'static,
{
    fn bad_chunk(error: String) -> Self {
        HttpResult::Err(HttpError::bad_chunk(error))
    }

    fn is_error(&self) -> bool {
        matches!(self, HttpResult::Err(_))
    }

    fn error(&self) -> Option<&str> {
        match self {
            HttpResult::Ok(_) => None,
            HttpResult::Err(e) => e.error(),
        }
    }
}

impl<T> Default for HttpResult<T>
where
    T: Default,
{
    fn default() -> Self {
        HttpResult::Ok(T::default())
    }
}

impl<T> From<Result<T, HttpError>> for HttpResult<T> {
    fn from(result: Result<T, HttpError>) -> Self {
        match result {
            Ok(val) => HttpResult::Ok(val),
            Err(err) => HttpResult::Err(err),
        }
    }
}

impl<T> From<HttpResult<T>> for Result<T, HttpError> {
    fn from(result: HttpResult<T>) -> Self {
        match result {
            HttpResult::Ok(val) => Ok(val),
            HttpResult::Err(err) => Err(err),
        }
    }
}

// Re-export internal hyper error types and helpers for legacy paths expecting `crate::error::*`
#[cfg(any(
    feature = "gzip",
    feature = "zstd",
    feature = "brotli",
    feature = "deflate",
))]
pub(crate) use crate::hyper::error::into_io;
pub(crate) use crate::hyper::error::{
    BadScheme, BoxError, TimedOut, body, builder, decode, decode_io, redirect, request,
    status_code, upgrade, url_bad_scheme,
};

// Additional MessageChunk implementations for external types used in AsyncStream
// (MessageChunk already imported at top of file)

// NOTE: Removed bytes::Bytes MessageChunk impl due to orphan rule violation
// Users can create wrapper types if needed

// NOTE: Removed http_body::Frame<T> MessageChunk impl due to orphan rule violation
// Users should implement MessageChunk on their own wrapper types

// NOTE: Removed std::result::Result<T,E> and std::vec::IntoIter<T> MessageChunk impls
// due to orphan rule violations. Users should create wrapper types if needed.

// NOTE: Removed Box<dyn std::error::Error + Send + Sync + 'static> MessageChunk impl
// due to orphan rule violation. Users should create wrapper types if needed.

// NOTE: Removed SocketAddr MessageChunk impl due to orphan rule violation
// Users can create wrapper types if needed

// NOTE: Removed hyper::Error MessageChunk impl due to orphan rule violation
// Users can create wrapper types if needed

// NOTE: Removed String MessageChunk impl due to orphan rule violation
// String already has Default impl in std

// HttpError already has Default and MessageChunk implementations via cyrup_sugars
