//! Error context for better debugging and error handling

use std::fmt;
use super::error_types::CandleError;

/// Error context for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed
    pub operation: &'static str,
    /// Model name if applicable
    pub model_name: Option<String>,
    /// Device information
    pub device: Option<String>,
    /// Additional context
    pub context: Option<String>}

impl ErrorContext {
    /// Create new error context
    #[inline(always)]
    pub fn new(operation: &'static str) -> Self {
        Self {
            operation,
            model_name: None,
            device: None,
            context: None}
    }

    /// Add model name to context
    #[inline(always)]
    pub fn with_model_name<S: Into<String>>(mut self, name: S) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Add device information to context
    #[inline(always)]
    pub fn with_device<S: Into<String>>(mut self, device: S) -> Self {
        self.device = Some(device.into());
        self
    }

    /// Add additional context
    #[inline(always)]
    pub fn with_context<S: Into<String>>(mut self, context: S) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// Enhanced error with context
#[derive(Debug, Clone)]
pub struct CandleErrorWithContext {
    /// The underlying error
    pub error: CandleError,
    /// Error context
    pub context: ErrorContext}

impl fmt::Display for CandleErrorWithContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} during {}", self.error, self.context.operation)?;

        if let Some(model) = &self.context.model_name {
            write!(f, " (model: {})", model)?;
        }

        if let Some(device) = &self.context.device {
            write!(f, " (device: {})", device)?;
        }

        if let Some(context) = &self.context.context {
            write!(f, " ({})", context)?;
        }

        Ok(())
    }
}

impl std::error::Error for CandleErrorWithContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}