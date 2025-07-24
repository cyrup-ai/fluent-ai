//! Comprehensive error handling system for logits processing
//!
//! This module provides semantic error types for all processing operations with:
//! - Hierarchical error classification for different failure modes
//! - Rich error context for debugging and monitoring  
//! - Integration with existing error systems
//! - Production-ready error handling without unwrap/expect

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
            Self::TensorOperationFailed(_) => "Check tensor shapes and try different tensor operations",
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
            Self::TensorOperationFailed(msg) => Self::TensorOperationFailed(format!("{}: {}", context_str, msg)),
            Self::InternalError(msg) => Self::InternalError(format!("{}: {}", context_str, msg)),
        }
    }
}

/// Error category for monitoring and metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    Configuration,
    Context,
    Numerical,
    Resource,
    External,
    ProcessorChain,
    Validation,
    Internal,
}

impl ErrorCategory {
    /// Get category name as string
    #[inline(always)]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Configuration => "configuration",
            Self::Context => "context",
            Self::Numerical => "numerical",
            Self::Resource => "resource",
            Self::External => "external",
            Self::ProcessorChain => "processor_chain",
            Self::Validation => "validation",
            Self::Internal => "internal",
        }
    }
}

impl std::fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Error severity levels for prioritization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl ErrorSeverity {
    /// Get severity name as string
    #[inline(always)]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }

    /// Get severity level as number
    #[inline(always)]
    pub fn level(&self) -> u8 {
        *self as u8
    }
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Result type alias for processing operations
pub type ProcessingResult<T> = Result<T, ProcessingError>;

/// Integration with existing Candle error system
impl From<candle_core::Error> for ProcessingError {
    fn from(err: candle_core::Error) -> Self {
        Self::ExternalError(format!("Candle error: {}", err))
    }
}

/// Integration with sampling error system (for migration)
impl From<crate::sampling::SamplingError> for ProcessingError {
    fn from(err: crate::sampling::SamplingError) -> Self {
        match &err {
            crate::sampling::SamplingError::InvalidTemperature(temp) => {
                Self::InvalidConfiguration(format!("Invalid temperature: {}", temp))
            }
            crate::sampling::SamplingError::InvalidTopP(p) => {
                Self::InvalidConfiguration(format!("Invalid top-p: {}", p))
            }
            crate::sampling::SamplingError::InvalidTopK(k) => {
                Self::InvalidConfiguration(format!("Invalid top-k: {}", k))
            }
            crate::sampling::SamplingError::InvalidRepetitionPenalty(penalty) => {
                Self::InvalidConfiguration(format!("Invalid repetition penalty: {}", penalty))
            }
            crate::sampling::SamplingError::TensorError(msg) => {
                Self::ExternalError(format!("Tensor error: {}", msg))
            }
            crate::sampling::SamplingError::NumericalInstability(msg) => {
                Self::NumericalError(msg.clone())
            }
            crate::sampling::SamplingError::EmptyVocabulary => {
                Self::ValidationError("Empty vocabulary".to_string())
            }
            crate::sampling::SamplingError::EmptyLogits => {
                Self::ValidationError("Empty logits".to_string())
            }
            crate::sampling::SamplingError::ProcessingFailed(msg) => {
                Self::InternalError(msg.clone())
            }
            crate::sampling::SamplingError::ProcessorChainError(msg) => {
                Self::ProcessorChainError(msg.clone())
            }
        }
    }
}

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
    pub metadata: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create new error context
    #[inline(always)]
    pub fn new<S: Into<String>>(operation: S) -> Self {
        Self {
            operation: operation.into(),
            processor: None,
            array_size: None,
            position: None,
            metadata: std::collections::HashMap::new(),
        }
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
    pub backtrace: Option<String>,
}

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
            }),
        }
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

/// Utility functions for error handling
pub mod utils {
    use super::*;

    /// Validate configuration parameter range
    #[inline(always)]
    pub fn validate_range<T>(value: T, min: T, max: T, param_name: &str) -> ProcessingResult<()>
    where
        T: PartialOrd + std::fmt::Display,
    {
        if value < min || value > max {
            return Err(ProcessingError::configuration(format!(
                "{} value {} is outside valid range [{}, {}]",
                param_name, value, min, max
            )));
        }
        Ok(())
    }

    /// Validate that a value is finite
    #[inline(always)]
    pub fn validate_finite(value: f32, param_name: &str) -> ProcessingResult<()> {
        if !value.is_finite() {
            return Err(ProcessingError::numerical(format!(
                "{} value {} is not finite",
                param_name, value
            )));
        }
        Ok(())
    }

    /// Validate array is not empty
    #[inline(always)]
    pub fn validate_not_empty<T>(array: &[T], array_name: &str) -> ProcessingResult<()> {
        if array.is_empty() {
            return Err(ProcessingError::validation(format!(
                "{} array is empty",
                array_name
            )));
        }
        Ok(())
    }

    /// Validate array sizes match
    #[inline(always)]
    pub fn validate_array_sizes<T, U>(
        array1: &[T],
        array2: &[U],
        name1: &str,
        name2: &str,
    ) -> ProcessingResult<()> {
        if array1.len() != array2.len() {
            return Err(ProcessingError::validation(format!(
                "{} size {} does not match {} size {}",
                name1,
                array1.len(),
                name2,
                array2.len()
            )));
        }
        Ok(())
    }

    /// Convert error to contextual error with operation info
    #[inline(always)]
    pub fn with_operation_context(error: ProcessingError, operation: &str) -> ContextualError {
        let context = ErrorContext::new(operation);
        ContextualError::new(error, context)
    }

    /// Convert error to contextual error with processor info
    #[inline(always)]
    pub fn with_processor_context(
        error: ProcessingError,
        operation: &str,
        processor: &str,
    ) -> ContextualError {
        let context = ErrorContext::new(operation).processor(processor);
        ContextualError::new(error, context)
    }

    /// Check if error should trigger system shutdown
    #[inline(always)]
    pub fn should_shutdown(error: &ProcessingError) -> bool {
        matches!(error, ProcessingError::InternalError(_))
            && error.severity() == ErrorSeverity::Critical
    }

    /// Get retry delay based on error type
    #[inline(always)]
    pub fn retry_delay(error: &ProcessingError) -> Option<std::time::Duration> {
        if !error.is_recoverable() {
            return None;
        }

        match error.category() {
            ErrorCategory::Resource => Some(std::time::Duration::from_millis(100)),
            ErrorCategory::External => Some(std::time::Duration::from_millis(50)),
            ErrorCategory::Numerical => Some(std::time::Duration::from_millis(10)),
            ErrorCategory::Context => Some(std::time::Duration::from_millis(1)),
            _ => None,
        }
    }
}
