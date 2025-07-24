//! Search system error types
//!
//! This module defines all error types used throughout the search system,
//! providing comprehensive error handling with context information.

use std::sync::Arc;
use thiserror::Error;

/// Search system errors
#[derive(Debug, Error, Clone)]
pub enum SearchError {
    /// Index-related errors
    #[error("Index error: {reason}")]
    IndexError { 
        /// Error reason description
        reason: Arc<str> 
    },
    
    /// Search query processing errors
    #[error("Search error: {reason}")]
    SearchQueryError { 
        /// Error reason description
        reason: Arc<str> 
    },
    
    /// Tag management errors
    #[error("Tag error: {reason}")]
    TagError { 
        /// Error reason description
        reason: Arc<str> 
    },
    
    /// Export functionality errors
    #[error("Export error: {reason}")]
    ExportError { 
        /// Error reason description
        reason: Arc<str> 
    },
    
    /// Invalid query format or parameters
    #[error("Invalid query: {details}")]
    InvalidQuery { 
        /// Error details
        details: Arc<str> 
    },
    
    /// System resource exhaustion
    #[error("System overload: {resource}")]
    SystemOverload { 
        /// Overloaded resource name
        resource: Arc<str> 
    },
    
    /// I/O related errors
    #[error("I/O error: {message}")]
    IoError { 
        /// Error message
        message: Arc<str> 
    },
    
    /// Serialization/deserialization errors
    #[error("Serialization error: {context}")]
    SerializationError { 
        /// Error context
        context: Arc<str> 
    },
    
    /// Concurrency-related errors
    #[error("Concurrency error: {operation}")]
    ConcurrencyError { 
        /// Failed operation
        operation: Arc<str> 
    },
    
    /// Configuration errors
    #[error("Configuration error: {setting}")]
    ConfigurationError { 
        /// Invalid setting
        setting: Arc<str> 
    },
}

impl SearchError {
    /// Create an index error
    pub fn index_error(reason: impl Into<Arc<str>>) -> Self {
        Self::IndexError { reason: reason.into() }
    }
    
    /// Create a search query error
    pub fn search_query_error(reason: impl Into<Arc<str>>) -> Self {
        Self::SearchQueryError { reason: reason.into() }
    }
    
    /// Create a tag error
    pub fn tag_error(reason: impl Into<Arc<str>>) -> Self {
        Self::TagError { reason: reason.into() }
    }
    
    /// Create an export error
    pub fn export_error(reason: impl Into<Arc<str>>) -> Self {
        Self::ExportError { reason: reason.into() }
    }
    
    /// Create an invalid query error
    pub fn invalid_query(details: impl Into<Arc<str>>) -> Self {
        Self::InvalidQuery { details: details.into() }
    }
    
    /// Create a system overload error
    pub fn system_overload(resource: impl Into<Arc<str>>) -> Self {
        Self::SystemOverload { resource: resource.into() }
    }
    
    /// Create an I/O error
    pub fn io_error(message: impl Into<Arc<str>>) -> Self {
        Self::IoError { message: message.into() }
    }
    
    /// Create a serialization error
    pub fn serialization_error(context: impl Into<Arc<str>>) -> Self {
        Self::SerializationError { context: context.into() }
    }
    
    /// Create a concurrency error
    pub fn concurrency_error(operation: impl Into<Arc<str>>) -> Self {
        Self::ConcurrencyError { operation: operation.into() }
    }
    
    /// Create a configuration error
    pub fn configuration_error(setting: impl Into<Arc<str>>) -> Self {
        Self::ConfigurationError { setting: setting.into() }
    }
    
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::IndexError { .. } => false,
            Self::SearchQueryError { .. } => true,
            Self::TagError { .. } => true,
            Self::ExportError { .. } => true,
            Self::InvalidQuery { .. } => true,
            Self::SystemOverload { .. } => true,
            Self::IoError { .. } => false,
            Self::SerializationError { .. } => false,
            Self::ConcurrencyError { .. } => true,
            Self::ConfigurationError { .. } => false,
        }
    }
    
    /// Get error category
    pub fn category(&self) -> &'static str {
        match self {
            Self::IndexError { .. } => "index",
            Self::SearchQueryError { .. } => "search",
            Self::TagError { .. } => "tag",
            Self::ExportError { .. } => "export",
            Self::InvalidQuery { .. } => "query",
            Self::SystemOverload { .. } => "system",
            Self::IoError { .. } => "io",
            Self::SerializationError { .. } => "serialization",
            Self::ConcurrencyError { .. } => "concurrency",
            Self::ConfigurationError { .. } => "configuration",
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::IndexError { .. } => ErrorSeverity::Critical,
            Self::SearchQueryError { .. } => ErrorSeverity::Warning,
            Self::TagError { .. } => ErrorSeverity::Warning,
            Self::ExportError { .. } => ErrorSeverity::Warning,
            Self::InvalidQuery { .. } => ErrorSeverity::Info,
            Self::SystemOverload { .. } => ErrorSeverity::Critical,
            Self::IoError { .. } => ErrorSeverity::Error,
            Self::SerializationError { .. } => ErrorSeverity::Error,
            Self::ConcurrencyError { .. } => ErrorSeverity::Warning,
            Self::ConfigurationError { .. } => ErrorSeverity::Critical,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Informational messages
    Info,
    /// Warning conditions
    Warning,
    /// Error conditions
    Error,
    /// Critical system failures
    Critical,
}

impl ErrorSeverity {
    /// Get severity as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Critical => "critical",
        }
    }
    
    /// Get severity as numeric level (0-3)
    pub fn as_level(&self) -> u8 {
        match self {
            Self::Info => 0,
            Self::Warning => 1,
            Self::Error => 2,
            Self::Critical => 3,
        }
    }
}

/// Result type alias for search operations
pub type SearchResult<T> = Result<T, SearchError>;

/// Error context for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed
    pub operation: Arc<str>,
    /// Additional context information
    pub context: Vec<(Arc<str>, Arc<str>)>,
    /// Timestamp when error occurred
    pub timestamp: u64,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(operation: impl Into<Arc<str>>) -> Self {
        Self {
            operation: operation.into(),
            context: Vec::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
    
    /// Add context information
    pub fn with_context(mut self, key: impl Into<Arc<str>>, value: impl Into<Arc<str>>) -> Self {
        self.context.push((key.into(), value.into()));
        self
    }
    
    /// Get context value by key
    pub fn get_context(&self, key: &str) -> Option<&Arc<str>> {
        self.context
            .iter()
            .find(|(k, _)| k.as_ref() == key)
            .map(|(_, v)| v)
    }
}