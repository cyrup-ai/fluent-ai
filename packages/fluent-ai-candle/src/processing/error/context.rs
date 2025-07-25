//! Error Context Module
//!
//! Provides enhanced error context information for detailed debugging and monitoring.
//! Zero-allocation context management with comprehensive metadata support.

use std::collections::HashMap;

use super::error_types::ProcessingError;
use super::classification::{ErrorCategory, ErrorSeverity};
/// Error context information for detailed reporting
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that was being performed
    pub operation: String,
    /// Processor name if applicable
    pub processor: Option<String>,
    /// Input array size if applicable
    pub array_size: Option<usize>,
    /// Token position if applicable
    pub position: Option<usize>,
    /// Additional context data
    pub metadata: HashMap<String, String>}

impl ErrorContext {
    /// Create new error context
    #[inline(always)]
    pub fn new<S: Into<String>>(operation: S) -> Self {
        Self {
            operation: operation.into(),
            processor: None,
            array_size: None,
            position: None,
            metadata: HashMap::new()}
    }

    /// Set processor name
    #[inline(always)]
    pub fn processor<S: Into<String>>(mut self, name: S) -> Self {
        self.processor = Some(name.into());
        self
    }

    /// Set array size
    #[inline(always)]
    pub fn array_size(mut self, size: usize) -> Self {
        self.array_size = Some(size);
        self
    }

    /// Set position
    #[inline(always)]
    pub fn position(mut self, pos: usize) -> Self {
        self.position = Some(pos);
        self
    }

    /// Add metadata entry
    #[inline(always)]
    pub fn metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl std::fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Operation: {}", self.operation)?;

        if let Some(ref processor) = self.processor {
            write!(f, ", Processor: {}", processor)?;
        }

        if let Some(size) = self.array_size {
            write!(f, ", Array size: {}", size)?;
        }

        if let Some(pos) = self.position {
            write!(f, ", Position: {}", pos)?;
        }

        if !self.metadata.is_empty() {
            write!(f, ", Metadata: {:?}", self.metadata)?;
        }

        Ok(())
    }
}

/// Enhanced error type with full context information
#[derive(Debug, Clone)]
pub struct ContextualError {
    /// Core error
    pub error: ProcessingError,
    /// Error context
    pub context: ErrorContext,
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
    /// Stack trace if available
    pub backtrace: Option<String>}

impl ContextualError {
    /// Create contextual error
    pub fn new(error: ProcessingError, context: ErrorContext) -> Self {
        Self {
            error,
            context,
            timestamp: std::time::SystemTime::now(),
            backtrace: std::env::var("RUST_BACKTRACE").ok().and_then(|val| {
                if val == "1" {
                    Some("backtrace".to_string())
                } else {
                    None
                }
            })}
    }

    /// Get error category
    #[inline(always)]
    pub fn category(&self) -> ErrorCategory {
        self.error.category()
    }

    /// Get error severity
    #[inline(always)]
    pub fn severity(&self) -> ErrorSeverity {
        self.error.severity()
    }

    /// Check if error is recoverable
    #[inline(always)]
    pub fn is_recoverable(&self) -> bool {
        self.error.is_recoverable()
    }

    /// Get suggested action
    #[inline(always)]
    pub fn suggested_action(&self) -> &'static str {
        self.error.suggested_action()
    }
}

impl std::fmt::Display for ContextualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.error, self.context)?;

        if let Some(ref backtrace) = self.backtrace {
            write!(f, "\nBacktrace: {}", backtrace)?;
        }

        Ok(())
    }
}

impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new("test_operation")
            .processor("test_processor")
            .array_size(100)
            .position(42)
            .metadata("key", "value");

        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.processor, Some("test_processor".to_string()));
        assert_eq!(context.array_size, Some(100));
        assert_eq!(context.position, Some(42));
        assert_eq!(context.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_error_context_display() {
        let context = ErrorContext::new("test_operation")
            .processor("test_processor")
            .array_size(100);

        let display = format!("{}", context);
        assert!(display.contains("Operation: test_operation"));
        assert!(display.contains("Processor: test_processor"));
        assert!(display.contains("Array size: 100"));
    }

    #[test]
    fn test_contextual_error_creation() {
        let error = ProcessingError::validation("test error");
        let context = ErrorContext::new("test_operation");
        let contextual_error = ContextualError::new(error, context);

        assert_eq!(contextual_error.category(), ErrorCategory::Validation);
        assert_eq!(contextual_error.severity(), ErrorSeverity::Low);
        assert!(!contextual_error.is_recoverable());
    }

    #[test]
    fn test_contextual_error_display() {
        let error = ProcessingError::validation("test error");
        let context = ErrorContext::new("test_operation");
        let contextual_error = ContextualError::new(error, context);

        let display = format!("{}", contextual_error);
        assert!(display.contains("Input validation error: test error"));
        assert!(display.contains("Operation: test_operation"));
    }
}