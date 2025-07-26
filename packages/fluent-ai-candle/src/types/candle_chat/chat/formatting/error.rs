//! Format error types and result handling
//!
//! Comprehensive error handling for formatting operations with detailed error
//! context and zero-allocation error propagation patterns.

use thiserror::Error;

/// Format error enumeration for comprehensive error handling
#[derive(Error, Debug, Clone)]
pub enum FormatError {
    #[error("Invalid markdown syntax: {detail}")]
    InvalidMarkdown { detail: String },
    #[error("Unsupported language: {language}")]
    UnsupportedLanguage { language: String },
    #[error("Parse error: {detail}")]
    ParseError { detail: String },
    #[error("Render error: {detail}")]
    RenderError { detail: String },
    #[error("Invalid content: {detail}")]
    InvalidContent { detail: String },
    #[error("Configuration error: {detail}")]
    ConfigurationError { detail: String },
    #[error("IO error: {detail}")]
    IoError { detail: String },
    #[error("Timeout error")]
    Timeout,
    #[error("Resource not found: {resource}")]
    ResourceNotFound { resource: String },
    #[error("Internal error: {detail}")]
    InternalError { detail: String }}

/// Result type for formatting operations
pub type FormatResult<T> = Result<T, FormatError>;

impl FormatError {
    /// Create an invalid markdown error
    pub fn invalid_markdown(detail: impl Into<String>) -> Self {
        Self::InvalidMarkdown {
            detail: detail.into()}
    }

    /// Create an unsupported language error
    pub fn unsupported_language(language: impl Into<String>) -> Self {
        Self::UnsupportedLanguage {
            language: language.into()}
    }

    /// Create a parse error
    pub fn parse_error(detail: impl Into<String>) -> Self {
        Self::ParseError {
            detail: detail.into()}
    }

    /// Create a render error
    pub fn render_error(detail: impl Into<String>) -> Self {
        Self::RenderError {
            detail: detail.into()}
    }

    /// Create an invalid content error
    pub fn invalid_content(detail: impl Into<String>) -> Self {
        Self::InvalidContent {
            detail: detail.into()}
    }

    /// Create a configuration error
    pub fn configuration_error(detail: impl Into<String>) -> Self {
        Self::ConfigurationError {
            detail: detail.into()}
    }

    /// Create an IO error
    pub fn io_error(detail: impl Into<String>) -> Self {
        Self::IoError {
            detail: detail.into()}
    }

    /// Create a timeout error
    pub fn timeout() -> Self {
        Self::Timeout
    }

    /// Create a resource not found error
    pub fn resource_not_found(resource: impl Into<String>) -> Self {
        Self::ResourceNotFound {
            resource: resource.into()}
    }

    /// Create an internal error
    pub fn internal_error(detail: impl Into<String>) -> Self {
        Self::InternalError {
            detail: detail.into()}
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::Timeout | Self::IoError { .. } | Self::ResourceNotFound { .. }
        )
    }

    /// Check if error indicates a user input problem
    pub fn is_user_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidMarkdown { .. }
                | Self::UnsupportedLanguage { .. }
                | Self::ParseError { .. }
                | Self::InvalidContent { .. }
                | Self::ConfigurationError { .. }
        )
    }

    /// Get error category for logging/metrics
    pub fn category(&self) -> &'static str {
        match self {
            Self::InvalidMarkdown { .. } => "markdown",
            Self::UnsupportedLanguage { .. } => "language",
            Self::ParseError { .. } => "parse",
            Self::RenderError { .. } => "render",
            Self::InvalidContent { .. } => "content",
            Self::ConfigurationError { .. } => "config",
            Self::IoError { .. } => "io",
            Self::Timeout => "timeout",
            Self::ResourceNotFound { .. } => "resource",
            Self::InternalError { .. } => "internal"}
    }
}

// Convert from std::io::Error
impl From<std::io::Error> for FormatError {
    fn from(error: std::io::Error) -> Self {
        Self::IoError {
            detail: error.to_string()}
    }
}

// Convert from fmt::Error
impl From<std::fmt::Error> for FormatError {
    fn from(error: std::fmt::Error) -> Self {
        Self::RenderError {
            detail: error.to_string()}
    }
}
