//! Zero-allocation error handling for candle integration

use fluent_ai_core::completion::error::{CompletionError, CandleError as CoreCandleError};
use std::fmt;

/// Result type alias for candle operations
pub type CandleResult<T> = Result<T, CandleError>;

/// Candle-specific error types with zero allocation
#[derive(Debug, Clone, PartialEq)]
pub enum CandleError {
    /// Model file not found or inaccessible
    ModelNotFound(String),
    /// Invalid model format or corrupted file
    InvalidModelFormat(&'static str),
    /// Tensor operation failed
    TensorOperation(&'static str),
    /// Device allocation or operation failed
    DeviceAllocation(&'static str),
    /// Quantization error
    Quantization(&'static str),
    /// Tokenizer error
    Tokenizer(&'static str),
    /// Memory mapping failed
    MemoryMapping(&'static str),
    /// Model loading timeout
    LoadingTimeout,
    /// Unsupported model architecture
    UnsupportedArchitecture(&'static str),
    /// Configuration error
    Configuration(&'static str),
    /// HuggingFace Hub error
    HuggingFaceHub(String),
    /// SafeTensors format error
    SafeTensors(&'static str),
    /// Context length exceeded
    ContextLengthExceeded {
        current: u32,
        max: u32,
    },
    /// Vocabulary size mismatch
    VocabularyMismatch {
        expected: u32,
        actual: u32,
    },
    /// Generation failed
    GenerationFailed(&'static str),
    /// Cache overflow
    CacheOverflow,
}

impl fmt::Display for CandleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound(path) => write!(f, "Model not found: {}", path),
            Self::InvalidModelFormat(msg) => write!(f, "Invalid model format: {}", msg),
            Self::TensorOperation(msg) => write!(f, "Tensor operation failed: {}", msg),
            Self::DeviceAllocation(msg) => write!(f, "Device allocation failed: {}", msg),
            Self::Quantization(msg) => write!(f, "Quantization error: {}", msg),
            Self::Tokenizer(msg) => write!(f, "Tokenizer error: {}", msg),
            Self::MemoryMapping(msg) => write!(f, "Memory mapping failed: {}", msg),
            Self::LoadingTimeout => write!(f, "Model loading timeout"),
            Self::UnsupportedArchitecture(arch) => write!(f, "Unsupported architecture: {}", arch),
            Self::Configuration(msg) => write!(f, "Configuration error: {}", msg),
            Self::HuggingFaceHub(msg) => write!(f, "HuggingFace Hub error: {}", msg),
            Self::SafeTensors(msg) => write!(f, "SafeTensors error: {}", msg),
            Self::ContextLengthExceeded { current, max } => {
                write!(f, "Context length exceeded: {} > {}", current, max)
            }
            Self::VocabularyMismatch { expected, actual } => {
                write!(f, "Vocabulary size mismatch: expected {}, got {}", expected, actual)
            }
            Self::GenerationFailed(msg) => write!(f, "Generation failed: {}", msg),
            Self::CacheOverflow => write!(f, "Cache overflow"),
        }
    }
}

impl std::error::Error for CandleError {}

// Conversion to core completion error
impl From<CandleError> for CompletionError {
    #[inline(always)]
    fn from(err: CandleError) -> Self {
        let core_err = match err {
            CandleError::ModelNotFound(_) => CoreCandleError::ModelNotFound("Model file not found"),
            CandleError::InvalidModelFormat(msg) => CoreCandleError::InvalidModelFormat(msg),
            CandleError::TensorOperation(msg) => CoreCandleError::TensorOperation(msg),
            CandleError::DeviceAllocation(msg) => CoreCandleError::DeviceAllocation(msg),
            CandleError::Quantization(msg) => CoreCandleError::Quantization(msg),
            CandleError::Tokenizer(msg) => CoreCandleError::Tokenizer(msg),
            _ => CoreCandleError::TensorOperation("Candle operation failed"),
        };
        CompletionError::from(core_err)
    }
}

// Conversion from candle-core errors
impl From<candle_core::Error> for CandleError {
    #[inline(always)]
    fn from(err: candle_core::Error) -> Self {
        match err {
            candle_core::Error::Msg(msg) => {
                if msg.contains("device") || msg.contains("cuda") || msg.contains("metal") {
                    Self::DeviceAllocation("Device operation failed")
                } else if msg.contains("tensor") || msg.contains("shape") {
                    Self::TensorOperation("Tensor operation failed")
                } else {
                    Self::TensorOperation("Candle operation failed")
                }
            }
            candle_core::Error::Io(_) => Self::ModelNotFound("IO error during model loading".to_string()),
            candle_core::Error::Zip(_) => Self::InvalidModelFormat("ZIP archive error"),
            candle_core::Error::SafeTensorError(_) => Self::SafeTensors("SafeTensors format error"),
            _ => Self::TensorOperation("Unknown candle error"),
        }
    }
}

// Conversion from tokenizers errors
impl From<tokenizers::Error> for CandleError {
    #[inline(always)]
    fn from(_err: tokenizers::Error) -> Self {
        Self::Tokenizer("Tokenizer operation failed")
    }
}

// Conversion from HuggingFace Hub errors
impl From<hf_hub::api::tokio::ApiError> for CandleError {
    #[inline(always)]
    fn from(err: hf_hub::api::tokio::ApiError) -> Self {
        Self::HuggingFaceHub(format!("HF Hub API error: {}", err))
    }
}

// Helper functions for creating common errors
impl CandleError {
    /// Create a model not found error
    #[inline(always)]
    pub fn model_not_found<S: Into<String>>(path: S) -> Self {
        Self::ModelNotFound(path.into())
    }

    /// Create an invalid model format error
    #[inline(always)]
    pub fn invalid_model_format(msg: &'static str) -> Self {
        Self::InvalidModelFormat(msg)
    }

    /// Create a tensor operation error
    #[inline(always)]
    pub fn tensor_operation(msg: &'static str) -> Self {
        Self::TensorOperation(msg)
    }

    /// Create a device allocation error
    #[inline(always)]
    pub fn device_allocation(msg: &'static str) -> Self {
        Self::DeviceAllocation(msg)
    }

    /// Create a quantization error
    #[inline(always)]
    pub fn quantization(msg: &'static str) -> Self {
        Self::Quantization(msg)
    }

    /// Create a tokenizer error
    #[inline(always)]
    pub fn tokenizer(msg: &'static str) -> Self {
        Self::Tokenizer(msg)
    }

    /// Create a memory mapping error
    #[inline(always)]
    pub fn memory_mapping(msg: &'static str) -> Self {
        Self::MemoryMapping(msg)
    }

    /// Create a loading timeout error
    #[inline(always)]
    pub fn loading_timeout() -> Self {
        Self::LoadingTimeout
    }

    /// Create an unsupported architecture error
    #[inline(always)]
    pub fn unsupported_architecture(arch: &'static str) -> Self {
        Self::UnsupportedArchitecture(arch)
    }

    /// Create a configuration error
    #[inline(always)]
    pub fn configuration(msg: &'static str) -> Self {
        Self::Configuration(msg)
    }

    /// Create a SafeTensors error
    #[inline(always)]
    pub fn safetensors(msg: &'static str) -> Self {
        Self::SafeTensors(msg)
    }

    /// Create a context length exceeded error
    #[inline(always)]
    pub fn context_length_exceeded(current: u32, max: u32) -> Self {
        Self::ContextLengthExceeded { current, max }
    }

    /// Create a vocabulary mismatch error
    #[inline(always)]
    pub fn vocabulary_mismatch(expected: u32, actual: u32) -> Self {
        Self::VocabularyMismatch { expected, actual }
    }

    /// Create a generation failed error
    #[inline(always)]
    pub fn generation_failed(msg: &'static str) -> Self {
        Self::GenerationFailed(msg)
    }

    /// Create a cache overflow error
    #[inline(always)]
    pub fn cache_overflow() -> Self {
        Self::CacheOverflow
    }

    /// Check if this error is retryable
    #[inline(always)]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::LoadingTimeout => true,
            Self::DeviceAllocation(_) => true,
            Self::HuggingFaceHub(_) => true,
            Self::CacheOverflow => true,
            _ => false,
        }
    }

    /// Get suggested retry delay in seconds
    #[inline(always)]
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            Self::LoadingTimeout => Some(5),
            Self::DeviceAllocation(_) => Some(1),
            Self::HuggingFaceHub(_) => Some(2),
            Self::CacheOverflow => Some(1),
            _ => None,
        }
    }
}

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
    pub context: Option<String>,
}

impl ErrorContext {
    /// Create new error context
    #[inline(always)]
    pub fn new(operation: &'static str) -> Self {
        Self {
            operation,
            model_name: None,
            device: None,
            context: None,
        }
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
    pub context: ErrorContext,
}

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

/// Macro for creating errors with context
#[macro_export]
macro_rules! candle_error {
    ($error:expr, $operation:expr) => {
        CandleErrorWithContext {
            error: $error,
            context: ErrorContext::new($operation),
        }
    };
    ($error:expr, $operation:expr, model = $model:expr) => {
        CandleErrorWithContext {
            error: $error,
            context: ErrorContext::new($operation).with_model_name($model),
        }
    };
    ($error:expr, $operation:expr, device = $device:expr) => {
        CandleErrorWithContext {
            error: $error,
            context: ErrorContext::new($operation).with_device($device),
        }
    };
    ($error:expr, $operation:expr, context = $context:expr) => {
        CandleErrorWithContext {
            error: $error,
            context: ErrorContext::new($operation).with_context($context),
        }
    };
}