//! Zero-allocation error handling for the build system
//!
//! This module provides error types and result aliases for build operations
//! with zero-allocation and maximum performance in mind.

use std::borrow::Cow;
use std::convert::Infallible;
use std::error::Error as StdError;
use std::fmt;

use crate::image_processing::generation::models::ModelError;

/// Type alias for build results with our custom error type
pub type BuildResult<T> = Result<T, BuildError>;

/// Type alias for cache results
pub type CacheResult<T> = Result<T, CacheError>;

// Removed unused type aliases YamlResult and HttpClientResult

/// Main error type for build operations
#[derive(Debug)]
pub enum BuildError {
    /// I/O operation failed
    IoError(std::io::Error),

    /// Cache operation failed (kept as it might be used by cache_manager)
    CacheError(Cow<'static, str>),

    /// YAML processing failed (kept as it might be used by yaml_processor)
    YamlError(YamlError),

    /// HTTP client error (kept as it might be used by http_client)
    HttpClientError(HttpClientError),

    /// Network operation failed
    Network(Cow<'static, str>),

    /// Template processing failed
    Template(Cow<'static, str>),

    /// Generic error for other cases
    Other(Cow<'static, str>),

    /// HTTP operation error
    HttpError(String),

    /// Validation error
    ValidationError(String),

    /// Bincode serialization/deserialization error
    BincodeError(bincode::error::EncodeError),

    /// Model operation error
    ModelError(String),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::IoError(e) => write!(f, "I/O error: {}", e),
            BuildError::CacheError(msg) => write!(f, "Cache error: {}", msg),
            BuildError::YamlError(e) => write!(f, "YAML error: {}", e),
            BuildError::HttpClientError(e) => write!(f, "HTTP client error: {}", e),
            BuildError::Network(msg) => write!(f, "Network error: {}", msg),
            BuildError::Template(msg) => write!(f, "Template error: {}", msg),
            BuildError::Other(msg) => write!(f, "Other error: {}", msg),
            BuildError::HttpError(msg) => write!(f, "HTTP error: {}", msg),
            BuildError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            BuildError::BincodeError(e) => write!(f, "Bincode error: {}", e),
            BuildError::ModelError(msg) => write!(f, "Model error: {}", msg),
        }
    }
}

impl StdError for BuildError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            BuildError::IoError(e) => Some(e),
            BuildError::YamlError(e) => Some(e),
            BuildError::HttpClientError(e) => Some(e),
            BuildError::BincodeError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<bincode::error::EncodeError> for BuildError {
    fn from(e: bincode::error::EncodeError) -> Self {
        BuildError::BincodeError(e)
    }
}

impl From<std::io::Error> for BuildError {
    fn from(e: std::io::Error) -> Self {
        BuildError::IoError(e)
    }
}

impl From<std::fmt::Error> for BuildError {
    fn from(e: std::fmt::Error) -> Self {
        BuildError::Other(e.to_string().into())
    }
}

impl From<YamlError> for BuildError {
    fn from(e: YamlError) -> Self {
        BuildError::YamlError(e)
    }
}

impl From<CacheError> for BuildError {
    fn from(e: CacheError) -> Self {
        BuildError::CacheError(e.to_string().into())
    }
}

impl From<HttpClientError> for BuildError {
    fn from(e: HttpClientError) -> Self {
        BuildError::HttpClientError(e)
    }
}

impl From<ModelError> for BuildError {
    fn from(e: ModelError) -> Self {
        BuildError::ModelError(e.to_string())
    }
}

/// Error type for cache operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// Cache entry not found
    NotFound(String),
    /// Cache entry expired
    Expired(String),
    /// I/O error during cache operation
    Io(String),
    /// Serialization or deserialization error
    Serialization(String),
    /// Other cache error
    Other(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::NotFound(s) => write!(f, "Cache entry not found: {}", s),
            CacheError::Expired(s) => write!(f, "Cache entry expired: {}", s),
            CacheError::Io(s) => write!(f, "Cache I/O error: {}", s),
            CacheError::Serialization(s) => write!(f, "Cache serialization error: {}", s),
            CacheError::Other(s) => write!(f, "Cache error: {}", s),
        }
    }
}

impl StdError for CacheError {}

/// Error type for YAML processing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YamlError {
    message: Cow<'static, str>,
    line: Option<usize>,
    column: Option<usize>,
}

impl YamlError {
    /// Create a new YAML error
    pub fn new(message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            message: message.into(),
            line: None,
            column: None,
        }
    }

    // Removed unused methods: with_location, set_location

    /// Get the line number if available
    pub fn line(&self) -> Option<usize> {
        self.line
    }

    /// Get the column number if available
    pub fn column(&self) -> Option<usize> {
        self.column
    }
}

impl fmt::Display for YamlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let (Some(line), Some(column)) = (self.line, self.column) {
            write!(f, "{} at line {}, column {}", self.message, line, column)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl StdError for YamlError {}

/// Error type for HTTP client operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpClientError {
    message: Cow<'static, str>,
    status: Option<u16>,
    retryable: bool,
}

impl HttpClientError {
    /// Create a new HTTP client error
    pub fn new(message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            message: message.into(),
            status: None,
            retryable: false,
        }
    }

    // Removed unused methods: with_status, set_status, set_retryable, is_retryable, status, is_status_retryable
}

impl fmt::Display for HttpClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(status) = self.status {
            write!(f, "HTTP Error {}: {}", status, self.message)
        } else {
            write!(f, "HTTP Client Error: {}", self.message)
        }
    }
}

impl StdError for HttpClientError {}

impl From<Infallible> for BuildError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

/// Helper macro for creating generic errors
#[macro_export]
macro_rules! format_err {
    ($variant:ident, $($arg:tt)*) => {
        $crate::build_modules::errors::BuildError::$variant(
            format!($($arg)*).into()
        )
    };
}

/// Helper macro for creating validation errors
#[macro_export]
macro_rules! validation_err {
    ($($arg:tt)*) => {
        $crate::build_modules::errors::BuildError::ValidationError(
            format!($($arg)*).into()
        )
    };
}

/// Helper macro for creating configuration errors
#[macro_export]
macro_rules! config_err {
    ($($arg:tt)*) => {
        $crate::build_modules::errors::BuildError::ConfigError(
            format!($($arg)*).into()
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_error_display() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let build_error = BuildError::IoError(io_error);
        assert!(format!("{}", build_error).contains("I/O error"));

        let yaml_error = YamlError::new("invalid YAML");
        let build_error = BuildError::YamlError(yaml_error);
        assert!(format!("{}", build_error).contains("YAML error"));
    }

    #[test]
    fn test_cache_error_display() {
        let cache_error = CacheError::NotFound;
        assert_eq!(format!("{}", cache_error), "Cache entry not found");
    }

    #[test]
    fn test_yaml_error_with_location() {
        let yaml_error = YamlError::with_location("invalid syntax", 42, 10);
        assert_eq!(
            format!("{}", yaml_error),
            "invalid syntax at line 42, column 10"
        );
    }

    #[test]
    fn test_http_client_error_retryable() {
        let error = HttpClientError::with_status(429, "Too Many Requests");
        assert!(error.is_retryable());

        let error = HttpClientError::with_status(500, "Internal Server Error");
        assert!(error.is_retryable());

        let error = HttpClientError::with_status(404, "Not Found");
        assert!(!error.is_retryable());
    }
}
