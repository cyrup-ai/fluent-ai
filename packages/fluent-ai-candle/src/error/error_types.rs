//! Core error types for candle integration

use std::fmt;

/// Result type alias for candle operations
pub type CandleResult<T> = Result<T, CandleError>;

/// Candle-specific error types with zero allocation
#[derive(Debug, Clone, PartialEq)]
pub enum CandleError {
    /// Generic message error (for compatibility with candle-core)
    Msg(String),
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
    /// Device incompatibility error
    IncompatibleDevice { msg: String },
    /// Model incompatibility error
    IncompatibleModel { msg: String },
    /// Version incompatibility error
    IncompatibleVersion { msg: String },
    /// Shape mismatch error
    ShapeMismatch { expected: String, actual: String },
    /// Device error
    DeviceError(String),
    /// IO error
    IoError(String),
    /// JSON parsing error
    JsonError(String)}

impl fmt::Display for CandleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Msg(msg) => write!(f, "{}", msg),
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
            Self::IncompatibleDevice { msg } => write!(f, "Incompatible device: {}", msg),
            Self::IncompatibleModel { msg } => write!(f, "Incompatible model: {}", msg),
            Self::IncompatibleVersion { msg } => write!(f, "Incompatible version: {}", msg),
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {}, got {}", expected, actual)
            },
            Self::DeviceError(msg) => write!(f, "Device error: {}", msg),
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
            Self::JsonError(msg) => write!(f, "JSON error: {}", msg)}
    }
}

impl std::error::Error for CandleError {}

/// Convert CandleError to candle_core::Error for seamless interoperability
impl From<CandleError> for candle_core::Error {
    fn from(err: CandleError) -> Self {
        match err {
            CandleError::Msg(msg) => candle_core::Error::Msg(msg),
            CandleError::ModelNotFound(path) => candle_core::Error::Msg(format!("Model not found: {}", path)),
            CandleError::ModelLoadError(msg) => candle_core::Error::Msg(format!("Model loading failed: {}", msg)),
            CandleError::InvalidModelFormat(msg) => candle_core::Error::Msg(format!("Invalid model format: {}", msg)),
            CandleError::TensorOperation(msg) => candle_core::Error::Msg(format!("Tensor operation failed: {}", msg)),
            CandleError::DeviceAllocation(msg) => candle_core::Error::Msg(format!("Device allocation failed: {}", msg)),
            CandleError::Quantization(msg) => candle_core::Error::Msg(format!("Quantization error: {}", msg)),
            CandleError::Tokenizer(msg) => candle_core::Error::Msg(format!("Tokenizer error: {}", msg)),
            CandleError::MemoryMapping(msg) => candle_core::Error::Msg(format!("Memory mapping failed: {}", msg)),
            CandleError::LoadingTimeout => candle_core::Error::Msg("Model loading timeout".to_string()),
            CandleError::UnsupportedArchitecture(arch) => candle_core::Error::Msg(format!("Unsupported architecture: {}", arch)),
            CandleError::Configuration(msg) => candle_core::Error::Msg(format!("Configuration error: {}", msg)),
            CandleError::TokenizationError(msg) => candle_core::Error::Msg(format!("Tokenization error: {}", msg)),
            CandleError::SafeTensors(msg) => candle_core::Error::Msg(format!("SafeTensors error: {}", msg)),
            CandleError::ContextLengthExceeded { current, max } => {
                candle_core::Error::Msg(format!("Context length exceeded: {} > {}", current, max))
            }
            CandleError::VocabularyMismatch { expected, actual } => {
                candle_core::Error::Msg(format!("Vocabulary size mismatch: expected {}, got {}", expected, actual))
            }
            CandleError::GenerationFailed(msg) => candle_core::Error::Msg(format!("Generation failed: {}", msg)),
            CandleError::CacheOverflow => candle_core::Error::Msg("Cache overflow".to_string()),
            CandleError::InvalidInput(msg) => candle_core::Error::Msg(format!("Invalid input: {}", msg)),
            CandleError::DeviceOperation(msg) => candle_core::Error::Msg(format!("Device operation failed: {}", msg)),
            CandleError::ProcessingError(msg) => candle_core::Error::Msg(format!("Processing error: {}", msg)),
            CandleError::InvalidConfiguration(msg) => candle_core::Error::Msg(format!("Invalid configuration: {}", msg)),
            CandleError::Tokenization(msg) => candle_core::Error::Msg(format!("Tokenization error: {}", msg)),
            CandleError::MemoryAllocation(msg) => candle_core::Error::Msg(format!("Memory allocation failed: {}", msg)),
            CandleError::TokenizerError(msg) => candle_core::Error::Msg(format!("Tokenizer error: {}", msg)),
            CandleError::ConfigurationError(msg) => candle_core::Error::Msg(format!("Configuration error: {}", msg)),
            CandleError::UnsupportedOperation(msg) => candle_core::Error::Msg(format!("Unsupported operation: {}", msg)),
            CandleError::IncompatibleTokenizer(msg) => candle_core::Error::Msg(format!("Incompatible tokenizer: {}", msg)),
            CandleError::Progress(msg) => candle_core::Error::Msg(format!("Progress tracking error: {}", msg)),
            CandleError::Cache(msg) => candle_core::Error::Msg(format!("Cache error: {}", msg)),
            CandleError::InitializationError(msg) => candle_core::Error::Msg(format!("ProgressHub initialization error: {}", msg)),
            CandleError::ProgressHubError(msg) => candle_core::Error::Msg(format!("ProgressHub error: {}", msg)),
            CandleError::BackendError(msg) => candle_core::Error::Msg(format!("Backend selection error: {}", msg)),
            CandleError::NetworkError(msg) => candle_core::Error::Msg(format!("Network error: {}", msg)),
            CandleError::ValidationError(msg) => candle_core::Error::Msg(format!("Validation error: {}", msg)),
            CandleError::Io(msg) => candle_core::Error::Msg(format!("I/O error: {}", msg)),
            CandleError::ModelNotLoaded(msg) => candle_core::Error::Msg(format!("Model not loaded: {}", msg)),
            CandleError::TensorError(msg) => candle_core::Error::Msg(format!("Tensor error: {}", msg)),
            CandleError::ModelInferenceError(msg) => candle_core::Error::Msg(format!("Model inference error: {}", msg)),
            CandleError::CacheError(msg) => candle_core::Error::Msg(format!("Cache error: {}", msg)),
            CandleError::IncompatibleDevice { msg } => candle_core::Error::Msg(format!("Incompatible device: {}", msg)),
            CandleError::IncompatibleModel { msg } => candle_core::Error::Msg(format!("Incompatible model: {}", msg)),
            CandleError::IncompatibleVersion { msg } => {
                candle_core::Error::Msg(format!("Incompatible version: {}", msg))
            }
            CandleError::ShapeMismatch { expected, actual } => {
                candle_core::Error::Msg(format!("Shape mismatch: expected {}, got {}", expected, actual))
            }
            CandleError::DeviceError(msg) => candle_core::Error::Msg(format!("Device error: {}", msg)),
            CandleError::IoError(msg) => candle_core::Error::Msg(format!("IO error: {}", msg)),
            CandleError::JsonError(msg) => candle_core::Error::Msg(format!("JSON error: {}", msg))}
    }
}