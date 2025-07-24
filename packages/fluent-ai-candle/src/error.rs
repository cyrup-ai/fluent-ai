//! Zero-allocation error handling for candle integration

use std::fmt;

use crate::types::{
    CandleCompletionError,
};

// Type aliases for missing error types
type ExtractionError = CandleCompletionError;
type CompletionRequestError = CandleCompletionError;
type CompletionCoreError = CandleCompletionError;

/// Result type alias for candle operations
pub type CandleResult<T> = Result<T, CandleError>;

/// Candle-specific error types with zero allocation
#[derive(Debug, Clone, PartialEq)]
pub enum CandleError {
    /// Model file not found or inaccessible
    ModelNotFound(String),
    /// Model loading failed
    ModelLoadError(String),
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
    /// Tokenization error with dynamic message
    TokenizationError(String),
    /// SafeTensors format error
    SafeTensors(&'static str),
    /// Context length exceeded
    ContextLengthExceeded { current: u32, max: u32 },
    /// Vocabulary size mismatch
    VocabularyMismatch { expected: u32, actual: u32 },
    /// Generation failed
    GenerationFailed(&'static str),
    /// Cache overflow
    CacheOverflow,
    /// Invalid input data
    InvalidInput(&'static str),
    /// Device operation failed
    DeviceOperation(&'static str),
    /// Processing error
    ProcessingError(&'static str),
    /// Invalid configuration  
    InvalidConfiguration(&'static str),
    /// Tokenization error
    Tokenization(&'static str),
    /// Memory allocation failed
    MemoryAllocation(&'static str),
    /// Tokenizer error with dynamic message
    TokenizerError(&'static str),
    /// Configuration error with dynamic message
    ConfigurationError(&'static str),
    /// Unsupported operation
    UnsupportedOperation(&'static str),
    /// Incompatible tokenizer
    IncompatibleTokenizer(&'static str),
    /// Progress tracking error
    Progress(String),
    /// Cache-related error
    Cache(String),
    /// ProgressHub client initialization failed
    InitializationError(String),
    /// ProgressHub download error
    ProgressHubError(String),
    /// Backend selection error
    BackendError(String),
    /// Network error during download
    NetworkError(String),
    /// Validation error
    ValidationError(String),
    /// I/O operation failed
    Io(String),
    /// Model not loaded error (alias for ModelLoadError)
    ModelNotLoaded(String),
    /// Tensor error (alias for TensorOperation)
    TensorError(String),
    /// Model inference error
    ModelInferenceError(String),
    /// Cache error (alias for Cache)
    CacheError(String),
}

impl fmt::Display for CandleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound(path) => write!(f, "Model not found: {}", path),
            Self::ModelLoadError(msg) => write!(f, "Model loading failed: {}", msg),
            Self::InvalidModelFormat(msg) => write!(f, "Invalid model format: {}", msg),
            Self::TensorOperation(msg) => write!(f, "Tensor operation failed: {}", msg),
            Self::DeviceAllocation(msg) => write!(f, "Device allocation failed: {}", msg),
            Self::Quantization(msg) => write!(f, "Quantization error: {}", msg),
            Self::Tokenizer(msg) => write!(f, "Tokenizer error: {}", msg),
            Self::MemoryMapping(msg) => write!(f, "Memory mapping failed: {}", msg),
            Self::LoadingTimeout => write!(f, "Model loading timeout"),
            Self::UnsupportedArchitecture(arch) => write!(f, "Unsupported architecture: {}", arch),
            Self::Configuration(msg) => write!(f, "Configuration error: {}", msg),
            Self::TokenizationError(msg) => write!(f, "Tokenization error: {}", msg),
            Self::SafeTensors(msg) => write!(f, "SafeTensors error: {}", msg),
            Self::ContextLengthExceeded { current, max } => {
                write!(f, "Context length exceeded: {} > {}", current, max)
            }
            Self::VocabularyMismatch { expected, actual } => {
                write!(
                    f,
                    "Vocabulary size mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::GenerationFailed(msg) => write!(f, "Generation failed: {}", msg),
            Self::CacheOverflow => write!(f, "Cache overflow"),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::DeviceOperation(msg) => write!(f, "Device operation failed: {}", msg),
            Self::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
            Self::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::Tokenization(msg) => write!(f, "Tokenization error: {}", msg),
            Self::MemoryAllocation(msg) => write!(f, "Memory allocation failed: {}", msg),
            Self::TokenizerError(msg) => write!(f, "Tokenizer error: {}", msg),
            Self::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            Self::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            Self::IncompatibleTokenizer(msg) => write!(f, "Incompatible tokenizer: {}", msg),
            Self::Progress(msg) => write!(f, "Progress tracking error: {}", msg),
            Self::Cache(msg) => write!(f, "Cache error: {}", msg),
            Self::InitializationError(msg) => {
                write!(f, "ProgressHub initialization error: {}", msg)
            }
            Self::ProgressHubError(msg) => write!(f, "ProgressHub error: {}", msg),
            Self::BackendError(msg) => write!(f, "Backend selection error: {}", msg),
            Self::NetworkError(msg) => write!(f, "Network error: {}", msg),
            Self::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            Self::Io(msg) => write!(f, "I/O error: {}", msg),
            Self::ModelNotLoaded(msg) => write!(f, "Model not loaded: {}", msg),
            Self::TensorError(msg) => write!(f, "Tensor error: {}", msg),
            Self::ModelInferenceError(msg) => write!(f, "Model inference error: {}", msg),
            Self::CacheError(msg) => write!(f, "Cache error: {}", msg),
        }
    }
}

impl std::error::Error for CandleError {}

// Note: ExtractionError is a type alias for CandleCompletionError
// The From<CandleError> for CandleCompletionError implementation below handles this conversion

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
            candle_core::Error::Io(_) => {
                Self::ModelNotFound("IO error during model loading".to_string())
            }
            candle_core::Error::Zip(_) => Self::InvalidModelFormat("ZIP archive error"),
            candle_core::Error::SafeTensor(_) => Self::SafeTensors("SafeTensors format error"),
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

// HuggingFace Hub errors are no longer used - replaced with direct HTTP3 implementation

// Conversion from domain completion request errors
impl From<CompletionRequestError> for CandleError {
    #[inline(always)]
    fn from(err: CompletionRequestError) -> Self {
        match err {
            CompletionRequestError::InvalidRequest { message: _ } => {
                Self::InvalidInput("Invalid completion parameter")
            }
            CompletionRequestError::TokenizationFailed { message: _ } => {
                Self::Configuration("Completion request validation failed")
            }
            CompletionRequestError::GenerationFailed { reason: _ } => {
                Self::ProcessingError("Generation failed during completion request")
            }
            _ => {
                Self::Configuration("Completion request processing failed")
            }
        }
    }
}

// Note: CompletionCoreError is a type alias for CandleCompletionError
// The From<CandleCompletionError> implementation below handles this conversion

// Helper functions for creating common errors
impl CandleError {
    /// Create a model not found error
    #[inline(always)]
    pub fn model_not_found<S: Into<String>>(path: S) -> Self {
        Self::ModelNotFound(path.into())
    }

    /// Create a model loading error
    #[inline(always)]
    pub fn model_load_error<S: Into<String>>(msg: S) -> Self {
        Self::ModelLoadError(msg.into())
    }

    /// Create a model loading error (alias for model_load_error)
    #[inline(always)]
    pub fn model_loading<S: Into<String>>(msg: S) -> Self {
        Self::ModelLoadError(msg.into())
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

    /// Create a tokenization error
    #[inline(always)]
    pub fn tokenization<S: Into<String>>(msg: S) -> Self {
        Self::TokenizationError(msg.into())
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

    /// Create an invalid input error
    #[inline(always)]
    pub fn invalid_input(msg: &'static str) -> Self {
        Self::InvalidInput(msg)
    }

    /// Create a streaming error
    #[inline(always)]
    pub fn streaming_error(msg: &'static str) -> Self {
        Self::GenerationFailed(msg)
    }

    /// Create a progress tracking error
    #[inline(always)]
    pub fn progress_error<S: Into<String>>(msg: S) -> Self {
        Self::Progress(msg.into())
    }

    /// Create a cache error
    #[inline(always)]
    pub fn cache_error<S: Into<String>>(msg: S) -> Self {
        Self::Cache(msg.into())
    }

    /// Check if this error is retryable
    #[inline(always)]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::LoadingTimeout => true,
            Self::DeviceAllocation(_) => true,
            Self::TokenizationError(_) => true,
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
            Self::TokenizationError(_) => Some(2),
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

/// Convert CandleError to CandleCompletionError for internal use
impl From<CandleError> for CandleCompletionError {
    fn from(err: CandleError) -> Self {
        match err {
            CandleError::ModelNotFound(_msg) => CandleCompletionError::ModelNotLoaded,
            CandleError::ModelLoadError(_msg) => CandleCompletionError::ModelNotLoaded,
            CandleError::ProcessingError(msg) => CandleCompletionError::GenerationFailed { reason: msg.to_string() },
            CandleError::InvalidConfiguration(msg) => CandleCompletionError::InvalidRequest { message: msg.to_string() },
            CandleError::Tokenization(msg) => CandleCompletionError::TokenizationFailed { message: msg.to_string() },
            CandleError::Io(msg) => CandleCompletionError::Internal { message: msg.clone() },
            CandleError::InvalidModelFormat(_msg) => {
                CandleCompletionError::ModelNotLoaded
            }
            CandleError::TensorOperation(msg) => {
                CandleCompletionError::GenerationFailed { reason: msg.to_string() }
            }
            CandleError::DeviceOperation(msg) => {
                CandleCompletionError::GenerationFailed { reason: msg.to_string() }
            }
            CandleError::MemoryAllocation(msg) => {
                CandleCompletionError::GenerationFailed { reason: msg.to_string() }
            }
            CandleError::GenerationFailed(msg) => {
                CandleCompletionError::GenerationFailed { reason: msg.to_string() }
            }
            CandleError::TokenizerError(msg) => {
                CandleCompletionError::GenerationFailed { reason: msg.to_string() }
            }
            CandleError::ConfigurationError(msg) => {
                CandleCompletionError::InvalidRequest { message: msg.to_string() }
            }
            CandleError::UnsupportedOperation(msg) => {
                CandleCompletionError::InvalidRequest { message: msg.to_string() }
            }
            CandleError::ContextLengthExceeded { current, max } => {
                CandleCompletionError::ContextLengthExceeded { current, max }
            }
            CandleError::VocabularyMismatch { expected, actual } => {
                CandleCompletionError::GenerationFailed { 
                    reason: format!(
                        "Vocabulary mismatch: expected {}, got {}",
                        expected, actual
                    )
                }
            }
            CandleError::IncompatibleTokenizer(msg) => {
                CandleCompletionError::GenerationFailed { reason: msg.to_string() }
            }
            CandleError::SafeTensors(msg) => {
                CandleCompletionError::ModelLoadingFailed { message: msg.to_string() }
            }
            CandleError::Quantization(msg) => {
                CandleCompletionError::ModelLoadingFailed { message: msg.to_string() }
            }
            CandleError::Tokenizer(msg) => CandleCompletionError::GenerationFailed { reason: msg.to_string() },
            CandleError::MemoryMapping(msg) => {
                CandleCompletionError::ModelLoadingFailed { message: msg.to_string() }
            }
            CandleError::LoadingTimeout => {
                CandleCompletionError::ModelLoadingFailed { message: "Model loading timeout".to_string() }
            }
            CandleError::UnsupportedArchitecture(arch) => {
                CandleCompletionError::InvalidRequest { message: format!("Unsupported architecture: {}", arch) }
            }
            CandleError::Configuration(msg) => CandleCompletionError::InvalidRequest { message: msg.to_string() },
            // Handle missing patterns
            CandleError::DeviceAllocation(msg) => {
                CandleCompletionError::GenerationFailed { reason: msg.to_string() }
            }
            CandleError::TokenizationError(msg) => CandleCompletionError::TokenizationFailed { message: msg.clone() },
            CandleError::CacheOverflow => {
                CandleCompletionError::GenerationFailed { reason: "Cache overflow".to_string() }
            }
            CandleError::InvalidInput(msg) => CandleCompletionError::InvalidRequest { message: msg.to_string() },
            CandleError::Progress(msg) => CandleCompletionError::Internal { message: msg.to_string() },
            CandleError::Cache(msg) => CandleCompletionError::GenerationFailed { reason: msg.to_string() },
            CandleError::InitializationError(msg) => CandleCompletionError::Internal { message: msg.to_string() },
            CandleError::ProgressHubError(msg) => CandleCompletionError::GenerationFailed { reason: msg.to_string() },
            CandleError::BackendError(msg) => CandleCompletionError::Internal { message: msg.to_string() },
            CandleError::NetworkError(msg) => CandleCompletionError::GenerationFailed { reason: msg.to_string() },
            CandleError::ValidationError(msg) => CandleCompletionError::GenerationFailed { reason: msg.to_string() },
            CandleError::ModelNotLoaded(msg) => CandleCompletionError::ModelLoadingFailed { message: msg },
            CandleError::TensorError(msg) => CandleCompletionError::GenerationFailed { reason: msg },
            CandleError::ModelInferenceError(msg) => CandleCompletionError::GenerationFailed { reason: msg },
            CandleError::CacheError(msg) => CandleCompletionError::GenerationFailed { reason: msg },
        }
    }
}

// ProgressHub error mappings
impl From<anyhow::Error> for CandleError {
    fn from(err: anyhow::Error) -> Self {
        CandleError::ProgressHubError(format!("ProgressHub error: {}", err))
    }
}
