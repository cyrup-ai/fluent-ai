//! Type conversions from external error types to CandleError

use super::error_types::CandleError;
use crate::types::CandleCompletionError;

// Type aliases for missing error types
#[allow(dead_code)] // Used in multiple modules but flagged incorrectly by compiler
type ExtractionError = CandleCompletionError;
#[allow(dead_code)] // Used for completion request compatibility
type CompletionRequestError = CandleCompletionError;
#[allow(dead_code)] // Used in generator.rs and error conversion logic
type CompletionCoreError = CandleCompletionError;

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
            _ => Self::TensorOperation("Unknown candle error")}
    }
}

// Conversion from tokenizers errors
impl From<tokenizers::Error> for CandleError {
    #[inline(always)]
    fn from(_err: tokenizers::Error) -> Self {
        Self::Tokenizer("Tokenizer operation failed")
    }
}

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
            _ => Self::Configuration("Completion request processing failed")}
    }
}

/// Convert CandleError to CandleCompletionError for internal use
impl From<CandleError> for CandleCompletionError {
    fn from(err: CandleError) -> Self {
        match err {
            CandleError::Msg(msg) => CandleCompletionError::Internal { message: msg },
            CandleError::ModelNotFound(_msg) => CandleCompletionError::ModelNotLoaded,
            CandleError::ModelLoadError(_msg) => CandleCompletionError::ModelNotLoaded,
            CandleError::ProcessingError(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::InvalidConfiguration(msg) => CandleCompletionError::InvalidRequest {
                message: msg.to_string()},
            CandleError::Tokenization(msg) => CandleCompletionError::TokenizationFailed {
                message: msg.to_string()},
            CandleError::Io(msg) => CandleCompletionError::Internal {
                message: msg.clone()},
            CandleError::InvalidModelFormat(_msg) => CandleCompletionError::ModelNotLoaded,
            CandleError::TensorOperation(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::DeviceOperation(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::MemoryAllocation(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::GenerationFailed(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::TokenizerError(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::ConfigurationError(msg) => CandleCompletionError::InvalidRequest {
                message: msg.to_string()},
            CandleError::UnsupportedOperation(msg) => CandleCompletionError::InvalidRequest {
                message: msg.to_string()},
            CandleError::ContextLengthExceeded { current, max } => {
                CandleCompletionError::ContextLengthExceeded { current, max }
            }
            CandleError::VocabularyMismatch { expected, actual } => {
                CandleCompletionError::GenerationFailed {
                    reason: format!("Vocabulary mismatch: expected {}, got {}", expected, actual)}
            }
            CandleError::IncompatibleTokenizer(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::SafeTensors(msg) => CandleCompletionError::ModelLoadingFailed {
                message: msg.to_string()},
            CandleError::Quantization(msg) => CandleCompletionError::ModelLoadingFailed {
                message: msg.to_string()},
            CandleError::Tokenizer(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::MemoryMapping(msg) => CandleCompletionError::ModelLoadingFailed {
                message: msg.to_string()},
            CandleError::LoadingTimeout => CandleCompletionError::ModelLoadingFailed {
                message: "Model loading timeout".to_string()},
            CandleError::UnsupportedArchitecture(arch) => CandleCompletionError::InvalidRequest {
                message: format!("Unsupported architecture: {}", arch)},
            CandleError::Configuration(msg) => CandleCompletionError::InvalidRequest {
                message: msg.to_string()},
            // Handle missing patterns
            CandleError::DeviceAllocation(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::TokenizationError(msg) => CandleCompletionError::TokenizationFailed {
                message: msg.clone()},
            CandleError::CacheOverflow => CandleCompletionError::GenerationFailed {
                reason: "Cache overflow".to_string()},
            CandleError::InvalidInput(msg) => CandleCompletionError::InvalidRequest {
                message: msg.to_string()},
            CandleError::Progress(msg) => CandleCompletionError::Internal {
                message: msg.to_string()},
            CandleError::Cache(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::InitializationError(msg) => CandleCompletionError::Internal {
                message: msg.to_string()},
            CandleError::ProgressHubError(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::BackendError(msg) => CandleCompletionError::Internal {
                message: msg.to_string()},
            CandleError::NetworkError(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::ValidationError(msg) => CandleCompletionError::GenerationFailed {
                reason: msg.to_string()},
            CandleError::ModelNotLoaded(msg) => {
                CandleCompletionError::ModelLoadingFailed { message: msg }
            }
            CandleError::TensorError(msg) => {
                CandleCompletionError::GenerationFailed { reason: msg }
            }
            CandleError::ModelInferenceError(msg) => {
                CandleCompletionError::GenerationFailed { reason: msg }
            }
            CandleError::CacheError(msg) => CandleCompletionError::GenerationFailed { reason: msg },
            CandleError::IncompatibleDevice { msg } => CandleCompletionError::InvalidRequest {
                message: format!("Incompatible device: {}", msg)},
            CandleError::IncompatibleModel { msg } => CandleCompletionError::InvalidRequest {
                message: format!("Incompatible model: {}", msg)},
            CandleError::IncompatibleVersion { msg } => CandleCompletionError::InvalidRequest {
                message: format!("Incompatible version: {}", msg)},
            CandleError::ShapeMismatch { expected, actual } => {
                CandleCompletionError::GenerationFailed {
                    reason: format!("Shape mismatch: expected {}, got {}", expected, actual)}
            }
            CandleError::DeviceError(msg) => CandleCompletionError::DeviceError { message: msg },
            CandleError::IoError(msg) => CandleCompletionError::Internal {
                message: format!("I/O error: {}", msg)},
            CandleError::JsonError(msg) => CandleCompletionError::Internal {
                message: format!("JSON error: {}", msg)}}
    }
}

// Conversion from ProcessingError to CandleError
impl From<crate::processing::error::error_types::ProcessingError> for CandleError {
    #[inline(always)]
    fn from(err: crate::processing::error::error_types::ProcessingError) -> Self {
        match err {
            crate::processing::error::error_types::ProcessingError::InvalidConfiguration(msg) => {
                CandleError::Msg(msg)
            }
            crate::processing::error::error_types::ProcessingError::ContextError(msg) => {
                CandleError::Msg(msg)
            }
            crate::processing::error::error_types::ProcessingError::NumericalError(msg) => {
                CandleError::Msg(msg)
            }
            crate::processing::error::error_types::ProcessingError::ResourceError(msg) => {
                CandleError::Msg(msg)
            }
            crate::processing::error::error_types::ProcessingError::ExternalError(msg) => {
                CandleError::Msg(msg)
            }
            crate::processing::error::error_types::ProcessingError::TensorOperationFailed(msg) => {
                CandleError::Msg(msg)
            }
            crate::processing::error::error_types::ProcessingError::ProcessorChainError(msg) => {
                CandleError::Msg(msg)
            }
            crate::processing::error::error_types::ProcessingError::ValidationError(msg) => {
                CandleError::ValidationError(msg)
            }
            crate::processing::error::error_types::ProcessingError::BufferOverflow(msg) => {
                CandleError::Msg(msg)
            }
            crate::processing::error::error_types::ProcessingError::InvalidInput(msg) => {
                CandleError::Msg(msg)
            }
            crate::processing::error::error_types::ProcessingError::InternalError(msg) => {
                CandleError::Msg(msg)
            }
        }
    }
}

// ProgressHub error mappings
impl From<anyhow::Error> for CandleError {
    fn from(err: anyhow::Error) -> Self {
        CandleError::ProgressHubError(format!("ProgressHub error: {}", err))
    }
}
