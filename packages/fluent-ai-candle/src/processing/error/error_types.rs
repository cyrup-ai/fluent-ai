//! Core Processing Error Types Module
//!
//! Defines the main ProcessingError enum and constructor methods for semantic error handling.
//! Provides zero-allocation error classification with comprehensive context information.

use super::classification::{ErrorCategory, ErrorSeverity};

/// Comprehensive error type for all processing operations
///
/// Provides semantic error classification with rich context information.
/// All processing operations return this error type for consistent
/// error handling throughout the system.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProcessingError {
    /// Invalid configuration parameter or setting
    ///
    /// Covers configuration validation failures such as:
    /// - Invalid temperature values (≤ 0, NaN, infinite)
    /// - Invalid top-k values (0 or excessive)
    /// - Invalid top-p values (< 0 or > 1)
    /// - Mismatched array sizes
    /// - Resource limit violations
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Processing context error
    ///
    /// Covers issues with the processing context such as:
    /// - Context initialization failures
    /// - Token history management errors
    /// - Context state inconsistencies
    /// - Context capacity violations
    #[error("Processing context error: {0}")]
    ContextError(String),

    /// Numerical stability or computational error
    ///
    /// Covers mathematical and numerical computation issues such as:
    /// - NaN or infinite values in calculations
    /// - Overflow/underflow in operations
    /// - Division by zero
    /// - Invalid probability distributions
    /// - Numerical precision loss
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Resource exhaustion or allocation failure
    ///
    /// Covers resource management issues such as:
    /// - Memory allocation failures
    /// - Array capacity violations
    /// - Stack overflow in deep recursion
    /// - Timeout in processing operations
    #[error("Resource exhaustion: {0}")]
    ResourceError(String),

    /// External dependency or integration error
    ///
    /// Covers errors from external systems such as:
    /// - SIMD operation failures
    /// - Hardware acceleration errors
    /// - External library errors
    /// - Platform-specific failures
    #[error("External system error: {0}")]
    ExternalError(String),

    /// Tensor operation failure
    ///
    /// Covers tensor computation and manipulation errors such as:
    /// - Tensor creation from slice failures
    /// - Shape mismatch in operations
    /// - Device compatibility issues
    /// - Memory layout conversion errors
    #[error("Tensor operation failed: {0}")]
    TensorOperationFailed(String),

    /// Processor chain composition or execution error
    ///
    /// Covers errors in processor chain management such as:
    /// - Chain construction failures
    /// - Processor compatibility issues
    /// - Chain optimization failures
    /// - Circular dependency detection
    #[error("Processor chain error: {0}")]
    ProcessorChainError(String),

    /// Validation failure for input data
    ///
    /// Covers input validation errors such as:
    /// - Empty or malformed input arrays
    /// - Invalid token sequences
    /// - Boundary condition violations
    /// - Data integrity failures
    #[error("Input validation error: {0}")]
    ValidationError(String),

    /// Buffer overflow or capacity violation
    ///
    /// Covers buffer management issues such as:
    /// - Array capacity violations during processing
    /// - Stack overflow conditions
    /// - Fixed-size buffer limitations exceeded
    /// - Memory buffer overflow protection
    #[error("Buffer overflow: {0}")]
    BufferOverflow(String),

    /// Invalid input data or parameters
    ///
    /// Covers input validation failures such as:
    /// - Malformed input data
    /// - Invalid parameter combinations
    /// - Out-of-range input values
    /// - Incompatible data formats
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Internal processing logic error
    ///
    /// Covers unexpected internal errors that should not occur
    /// in normal operation but need handling for robustness:
    /// - Assertion failures in debug builds
    /// - Internal state inconsistencies
    /// - Algorithm implementation bugs
    /// - Unreachable code paths
    #[error("Internal processing error: {0}")]
    InternalError(String),
}

impl ProcessingError {
    /// Create configuration error with context
    #[inline(always)]
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::InvalidConfiguration(message.into())
    }

    /// Create context error with details
    #[inline(always)]
    pub fn context<S: Into<String>>(message: S) -> Self {
        Self::ContextError(message.into())
    }

    /// Create numerical error with computation details
    #[inline(always)]
    pub fn numerical<S: Into<String>>(message: S) -> Self {
        Self::NumericalError(message.into())
    }

    /// Create resource error with resource details
    #[inline(always)]
    pub fn resource<S: Into<String>>(message: S) -> Self {
        Self::ResourceError(message.into())
    }

    /// Create external system error
    #[inline(always)]
    pub fn external<S: Into<String>>(message: S) -> Self {
        Self::ExternalError(message.into())
    }

    /// Create processor chain error
    #[inline(always)]
    pub fn processor_chain<S: Into<String>>(message: S) -> Self {
        Self::ProcessorChainError(message.into())
    }

    /// Create validation error
    #[inline(always)]
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::ValidationError(message.into())
    }

    /// Create buffer overflow error
    #[inline(always)]
    pub fn buffer_overflow<S: Into<String>>(message: S) -> Self {
        Self::BufferOverflow(message.into())
    }

    /// Create invalid input error
    #[inline(always)]
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInput(message.into())
    }

    /// Create tensor operation error
    #[inline(always)]
    pub fn tensor_operation<S: Into<String>>(message: S) -> Self {
        Self::TensorOperationFailed(message.into())
    }

    /// Create internal error (should be rare)
    #[inline(always)]
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::InternalError(message.into())
    }

    /// Get error category for monitoring and metrics
    #[inline(always)]
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::InvalidConfiguration(_) => ErrorCategory::Configuration,
            Self::ContextError(_) => ErrorCategory::Context,
            Self::NumericalError(_) => ErrorCategory::Numerical,
            Self::ResourceError(_) => ErrorCategory::Resource,
            Self::ExternalError(_) => ErrorCategory::External,
            Self::ProcessorChainError(_) => ErrorCategory::ProcessorChain,
            Self::ValidationError(_) => ErrorCategory::Validation,
            Self::BufferOverflow(_) => ErrorCategory::Resource,
            Self::InvalidInput(_) => ErrorCategory::Validation,
            Self::TensorOperationFailed(_) => ErrorCategory::External,
            Self::InternalError(_) => ErrorCategory::Internal,
        }
    }

    /// Check if error is recoverable
    #[inline(always)]
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::InvalidConfiguration(_) => false, // Config must be fixed
            Self::ContextError(_) => true,          // Context can be reset
            Self::NumericalError(_) => true,        // Different values might work
            Self::ResourceError(_) => true,         // Might work with retry
            Self::ExternalError(_) => true,         // External issue might resolve
            Self::ProcessorChainError(_) => false,  // Chain must be rebuilt
            Self::ValidationError(_) => false,      // Input must be corrected
            Self::BufferOverflow(_) => true,        // Might work with smaller input
            Self::InvalidInput(_) => false,         // Input must be corrected
            Self::TensorOperationFailed(_) => true, // Different tensor ops might work
            Self::InternalError(_) => false,        // Internal bug, not recoverable
        }
    }

    /// Get error severity level
    #[inline(always)]
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::InvalidConfiguration(_) => ErrorSeverity::High,
            Self::ContextError(_) => ErrorSeverity::Medium,
            Self::NumericalError(_) => ErrorSeverity::Medium,
            Self::ResourceError(_) => ErrorSeverity::High,
            Self::ExternalError(_) => ErrorSeverity::Medium,
            Self::ProcessorChainError(_) => ErrorSeverity::High,
            Self::ValidationError(_) => ErrorSeverity::Low,
            Self::BufferOverflow(_) => ErrorSeverity::High,
            Self::InvalidInput(_) => ErrorSeverity::Low,
            Self::TensorOperationFailed(_) => ErrorSeverity::Medium,
            Self::InternalError(_) => ErrorSeverity::Critical,
        }
    }

    /// Get suggested action for error resolution
    #[inline(always)]
    pub fn suggested_action(&self) -> &'static str {
        match self {
            Self::InvalidConfiguration(_) => "Review and correct configuration parameters",
            Self::ContextError(_) => "Reset processing context and retry",
            Self::NumericalError(_) => "Check input values and try different parameters",
            Self::ResourceError(_) => "Reduce resource usage or increase limits",
            Self::ExternalError(_) => "Check external system status and retry",
            Self::ProcessorChainError(_) => "Rebuild processor chain with valid processors",
            Self::ValidationError(_) => "Validate and correct input data",
            Self::BufferOverflow(_) => "Reduce input size or increase buffer capacity",
            Self::InvalidInput(_) => "Validate and correct input data format",
            Self::TensorOperationFailed(_) => {
                "Check tensor shapes and try different tensor operations"
            }
            Self::InternalError(_) => "Report as bug, restart system if necessary",
        }
    }

    /// Create error with additional context
    pub fn with_context<S: Into<String>>(self, context: S) -> Self {
        let context_str = context.into();
        match self {
            Self::InvalidConfiguration(msg) => {
                Self::InvalidConfiguration(format!("{}: {}", context_str, msg))
            }
            Self::ContextError(msg) => Self::ContextError(format!("{}: {}", context_str, msg)),
            Self::NumericalError(msg) => Self::NumericalError(format!("{}: {}", context_str, msg)),
            Self::ResourceError(msg) => Self::ResourceError(format!("{}: {}", context_str, msg)),
            Self::ExternalError(msg) => Self::ExternalError(format!("{}: {}", context_str, msg)),
            Self::ProcessorChainError(msg) => {
                Self::ProcessorChainError(format!("{}: {}", context_str, msg))
            }
            Self::ValidationError(msg) => {
                Self::ValidationError(format!("{}: {}", context_str, msg))
            }
            Self::BufferOverflow(msg) => Self::BufferOverflow(format!("{}: {}", context_str, msg)),
            Self::InvalidInput(msg) => Self::InvalidInput(format!("{}: {}", context_str, msg)),
            Self::TensorOperationFailed(msg) => {
                Self::TensorOperationFailed(format!("{}: {}", context_str, msg))
            }
            Self::InternalError(msg) => Self::InternalError(format!("{}: {}", context_str, msg)),
        }
    }
}