//! Helper functions for creating common error types

use super::error_types::CandleError;

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

    /// Create a generic message error
    #[inline(always)]
    pub fn msg<S: Into<String>>(msg: S) -> Self {
        Self::Msg(msg.into())
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