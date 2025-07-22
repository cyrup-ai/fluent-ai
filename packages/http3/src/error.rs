//! HTTP error types and utilities

use http::StatusCode;
use http::header::{InvalidHeaderName, InvalidHeaderValue};
use thiserror::Error;

/// HTTP error types
#[derive(Error, Debug, Clone)]
pub enum HttpError {
    /// Network error
    #[error("Network error: {message}")]
    NetworkError { message: String },

    /// Client configuration error
    #[error("Client error: {message}")]
    ClientError { message: String },

    /// HTTP status error
    #[error("HTTP {status}: {message}")]
    HttpStatus {
        status: u16,
        message: String,
        body: String,
    },

    /// Timeout error
    #[error("Request timeout: {message}")]
    Timeout { message: String },

    /// Serialization error
    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    /// Deserialization error
    #[error("Deserialization error: {message}")]
    DeserializationError { message: String },

    /// URL parsing error
    #[error("URL parsing error: {message}")]
    UrlParseError { message: String },

    /// Invalid URL error
    #[error("Invalid URL '{url}': {message}")]
    InvalidUrl { url: String, message: String },

    /// Invalid response error
    #[error("Invalid response: {message}")]
    InvalidResponse { message: String },

    /// TLS error
    #[error("TLS error: {message}")]
    TlsError { message: String },

    /// Connection error
    #[error("Connection error: {message}")]
    ConnectionError { message: String },

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Invalid header error
    #[error("Invalid header: {message}")]
    InvalidHeader { message: String },

    /// Custom error for middleware and other uses
    #[error("Custom error: {message}")]
    Custom { message: String },

    /// Error processing a response chunk during collection
    #[error("Chunk processing error: {source}")]
    ChunkProcessingError {
        source: std::sync::Arc<serde_json::Error>,
        body: Vec<u8>,
    },

    /// Download was interrupted
    #[error("Download interrupted: {message}")]
    DownloadInterrupted { message: String },

    /// Invalid content length
    #[error("Invalid content length: {message}")]
    InvalidContentLength { message: String },

    /// A retryable error
    #[error("Retryable error: {message}")]
    Retryable { message: String },

    /// A non-retryable error
    #[error("Non-retryable error: {message}")]
    NonRetryable { message: String },
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

impl From<reqwest::Error> for HttpError {
    fn from(error: reqwest::Error) -> Self {
        if error.is_timeout() {
            HttpError::Timeout {
                message: error.to_string(),
            }
        } else if error.is_status() {
            HttpError::HttpStatus {
                status: error.status().map_or(0, |s| s.as_u16()),
                message: error.to_string(),
                body: "".to_string(), // reqwest::Error doesn't expose the body
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

/// Result type for HTTP operations
pub type HttpResult<T> = Result<T, HttpError>;
