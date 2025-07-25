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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_error_creation() {
        let error = FormatError::invalid_markdown("Missing closing tag");
        assert_eq!(error.category(), "markdown");
        assert!(error.is_user_error());
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_timeout_error() {
        let error = FormatError::timeout();
        assert_eq!(error.category(), "timeout");
        assert!(!error.is_user_error());
        assert!(error.is_recoverable());
    }

    #[test]
    fn test_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let format_error: FormatError = io_error.into();
        assert_eq!(format_error.category(), "io");
    }

    #[test]
    fn test_error_display() {
        let error = FormatError::unsupported_language("cobol");
        let error_string = format!("{}", error);
        assert!(error_string.contains("Unsupported language: cobol"));
    }

    #[test]
    fn test_error_classification() {
        let user_errors = vec![
            FormatError::invalid_markdown("test"),
            FormatError::parse_error("test"),
            FormatError::configuration_error("test"),
        ];

        for error in user_errors {
            assert!(error.is_user_error());
            assert!(!error.is_recoverable());
        }

        let recoverable_errors = vec![
            FormatError::timeout(),
            FormatError::io_error("test"),
            FormatError::resource_not_found("test"),
        ];

        for error in recoverable_errors {
            assert!(error.is_recoverable());
        }
    }
}