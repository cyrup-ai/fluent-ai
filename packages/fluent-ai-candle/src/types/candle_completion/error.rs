//! Comprehensive error handling for completion operations
//!
//! This module provides specialized error types for text completion operations,
//! data extraction, and model interaction. It includes detailed error variants
//! with contextual information for debugging and error recovery in production
//! ML inference systems.
//!
//! # Error Categories
//!
//! - **Model Errors**: Model loading, inference, and configuration issues
//! - **Request Errors**: Invalid parameters, validation failures, rate limiting
//! - **Resource Errors**: Memory, device, and timeout issues
//! - **Extraction Errors**: Response parsing and schema validation failures
//!
//! # Error Handling Patterns
//!
//! ```rust
//! use fluent_ai_candle::types::candle_completion::error::{CandleCompletionError, CandleCompletionResult};
//!
//! fn handle_completion_error(result: CandleCompletionResult<String>) {
//!     match result {
//!         Ok(response) => println!("Success: {}", response),
//!         Err(CandleCompletionError::ModelNotLoaded) => {
//!             // Handle model loading
//!         }
//!         Err(CandleCompletionError::ContextLengthExceeded { current, max }) => {
//!             // Handle context length issues
//!         }
//!         Err(CandleCompletionError::RateLimited { retry_after, .. }) => {
//!             // Handle rate limiting with backoff
//!         }
//!         Err(err) => eprintln!("Other error: {}", err),
//!     }
//! }
//! ```

use thiserror::Error;

/// Comprehensive error enumeration for completion operations
///
/// Provides detailed error variants covering all aspects of text completion
/// operations including model loading, inference, parameter validation,
/// resource management, and operational failures. Each variant includes
/// contextual information to aid in debugging and error recovery.
///
/// # Design Principles
///
/// - **Semantic clarity**: Each error variant has a specific meaning
/// - **Contextual information**: Errors include relevant details for debugging
/// - **Recovery guidance**: Error messages suggest potential solutions
/// - **Performance conscious**: Uses efficient string handling and cloning
#[derive(Debug, Error, Clone)]
pub enum CandleCompletionError {
    /// Model not loaded or available for inference
    ///
    /// Indicates that no model has been loaded into memory for completion operations.
    /// This typically occurs when attempting to generate completions before model
    /// initialization or after model unloading.
    #[error("Model not loaded")]
    ModelNotLoaded,

    /// Invalid request parameters provided to completion operation
    ///
    /// Indicates that the completion request contains invalid or malformed parameters.
    /// Common causes include negative values, missing required fields, or incompatible
    /// parameter combinations.
    #[error("Invalid request parameters: {message}")]
    InvalidRequest {
        /// Detailed description of the parameter validation failure
        message: String,
    },

    /// Text generation process failed during execution
    ///
    /// Indicates that the model encountered an error during the generation process,
    /// such as numerical instability, sampling failures, or convergence issues.
    #[error("Generation failed: {reason}")]
    GenerationFailed {
        /// Specific reason for the generation failure
        reason: String,
    },

    /// Input context exceeds maximum supported length
    ///
    /// Indicates that the input context length exceeds the model's maximum supported
    /// context window. This is a hard limit that cannot be bypassed without truncating
    /// the input or using a model with a larger context window.
    #[error("Context length exceeded: {current} > {max}")]
    ContextLengthExceeded {
        /// Current context length in tokens
        current: u32,
        /// Maximum supported context length in tokens
        max: u32,
    },

    /// Text tokenization process failed
    ///
    /// Indicates that the input text could not be converted to tokens, typically
    /// due to unsupported characters, encoding issues, or tokenizer failures.
    #[error("Tokenization failed: {message}")]
    TokenizationFailed {
        /// Details about the tokenization failure
        message: String,
    },

    /// Model inference operation encountered an error
    ///
    /// Indicates that the neural network forward pass or related inference operations
    /// failed. This can include tensor operations, activation functions, or numerical
    /// computation errors.
    #[error("Model inference error: {message}")]
    InferenceError {
        /// Detailed description of the inference failure
        message: String,
    },

    /// Device-related error (GPU/CPU operations)
    ///
    /// Indicates problems with hardware device operations, such as GPU memory issues,
    /// CUDA errors, device synchronization failures, or device unavailability.
    #[error("Device error: {message}")]
    DeviceError {
        /// Specific device error description
        message: String,
    },

    /// Memory allocation or management error
    ///
    /// Indicates insufficient memory for completion operations, memory allocation
    /// failures, or memory corruption issues that prevent successful completion.
    #[error("Memory allocation error: {message}")]
    MemoryError {
        /// Details about the memory error
        message: String,
    },

    /// Operation exceeded configured timeout duration
    ///
    /// Indicates that a completion operation took longer than the configured timeout
    /// period. This helps prevent hung operations and ensures responsive behavior.
    #[error("Operation timed out after {seconds} seconds")]
    Timeout {
        /// Number of seconds after which the operation timed out
        seconds: u64,
    },

    /// Requested operation is not supported by the current model
    ///
    /// Indicates that the requested functionality is not available for the current
    /// model configuration, such as streaming with a non-streaming model or tool
    /// calling with an incompatible model.
    #[error("Unsupported operation: {operation}")]
    UnsupportedOperation {
        /// Name of the unsupported operation
        operation: String,
    },

    /// Invalid parameter value provided to the operation
    ///
    /// Indicates that a specific parameter has an invalid value, such as out-of-range
    /// numbers, malformed strings, or incompatible parameter combinations.
    #[error("Invalid parameter: {message}")]
    InvalidParameter {
        /// Description of the invalid parameter and expected values
        message: String,
    },

    /// Input validation failed according to specified rules
    ///
    /// Indicates that the input failed validation checks, such as schema validation,
    /// content filtering, or business rule validation.
    #[error("Validation error: {message}")]
    Validation {
        /// Detailed validation failure message
        message: String,
    },

    /// Internal system error occurred during processing
    ///
    /// Indicates an unexpected internal error that is not user-recoverable, such as
    /// system state corruption, unexpected code paths, or unhandled edge cases.
    #[error("Internal error: {message}")]
    Internal {
        /// Internal error details for debugging
        message: String,
    },

    /// Model loading process failed
    ///
    /// Indicates that the model could not be loaded from storage, typically due to
    /// file corruption, missing files, incompatible formats, or insufficient resources.
    #[error("Model loading failed: {message}")]
    ModelLoadingFailed {
        /// Specific reason for the model loading failure
        message: String,
    },

    /// Operation was rate limited by the system
    ///
    /// Indicates that the request was rejected due to rate limiting policies.
    /// The client should wait for the specified retry period before attempting again.
    #[error("Rate limited: {message}, retry after {retry_after} seconds")]
    RateLimited {
        /// Rate limiting message with details
        message: String,
        /// Number of seconds to wait before retrying
        retry_after: u64,
    },
}

/// Specialized error enumeration for data extraction operations
///
/// Provides error variants specific to extracting structured data from
/// completion responses, including JSON parsing, schema validation,
/// and field extraction failures.
///
/// These errors typically occur when processing model responses that are
/// expected to contain structured data in specific formats.
#[derive(Debug, Error, Clone)]
pub enum CandleExtractionError {
    /// Failed to parse completion response into expected format
    ///
    /// Indicates that the model response could not be parsed as the expected
    /// data format (JSON, XML, etc.). This can occur when the model generates
    /// malformed output or unexpected content.
    #[error("Failed to parse response: {message}")]
    ParseError {
        /// Detailed parsing error message
        message: String,
    },

    /// Required field missing from extracted data
    ///
    /// Indicates that a required field is missing from the parsed response.
    /// This occurs when the model generates valid format but omits expected fields.
    #[error("Missing required field: {field}")]
    MissingField {
        /// Name of the missing required field
        field: String,
    },

    /// Data format does not match expected format
    ///
    /// Indicates that the parsed data has a different format than expected,
    /// such as receiving a string when an object was expected.
    #[error("Invalid format: expected {expected}, got {actual}")]
    InvalidFormat {
        /// Expected data format description
        expected: String,
        /// Actual format that was received
        actual: String,
    },

    /// Schema validation failed for extracted data
    ///
    /// Indicates that the extracted data does not conform to the expected
    /// schema or validation rules, such as invalid field types or constraint violations.
    #[error("Schema validation failed: {message}")]
    SchemaValidation {
        /// Schema validation error details
        message: String,
    },
}

/// Result type for completion operations with comprehensive error handling
///
/// Standard Result type for all completion operations, providing either
/// successful results or detailed error information via CandleCompletionError.
/// Used throughout the completion API for consistent error handling.
pub type CandleCompletionResult<T> = Result<T, CandleCompletionError>;

/// Result type for data extraction operations with specialized error handling
///
/// Standard Result type for data extraction operations, providing either
/// successfully extracted data or detailed error information via CandleExtractionError.
/// Used for parsing and validating structured data from model responses.
pub type CandleExtractionResult<T> = Result<T, CandleExtractionError>;

impl CandleCompletionError {
    /// Create a new invalid request error with custom message
    ///
    /// Convenience constructor for invalid request errors with descriptive messaging.
    /// Use this when request parameters fail validation or are malformed.
    ///
    /// # Arguments
    ///
    /// * `message` - Descriptive error message explaining the validation failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::candle_completion::error::CandleCompletionError;
    ///
    /// let error = CandleCompletionError::invalid_request("Temperature must be between 0.0 and 2.0");
    /// ```
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    /// Create a new generation failed error with reason
    ///
    /// Convenience constructor for generation failures during text completion.
    /// Use this when the model encounters errors during the generation process.
    ///
    /// # Arguments
    ///
    /// * `reason` - Specific reason for the generation failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::candle_completion::error::CandleCompletionError;
    ///
    /// let error = CandleCompletionError::generation_failed("Sampling process diverged");
    /// ```
    pub fn generation_failed(reason: impl Into<String>) -> Self {
        Self::GenerationFailed {
            reason: reason.into(),
        }
    }

    /// Create a new inference error with detailed message
    ///
    /// Convenience constructor for model inference failures during computation.
    /// Use this when neural network operations or tensor computations fail.
    ///
    /// # Arguments
    ///
    /// * `message` - Detailed description of the inference failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::candle_completion::error::CandleCompletionError;
    ///
    /// let error = CandleCompletionError::inference_error("Forward pass failed on layer 12");
    /// ```
    pub fn inference_error(message: impl Into<String>) -> Self {
        Self::InferenceError {
            message: message.into(),
        }
    }

    /// Create a new invalid parameter error with description
    ///
    /// Convenience constructor for parameter validation failures.
    /// Use this when specific parameters have invalid values or ranges.
    ///
    /// # Arguments
    ///
    /// * `message` - Description of the invalid parameter and expected values
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::candle_completion::error::CandleCompletionError;
    ///
    /// let error = CandleCompletionError::invalid_parameter("max_tokens must be positive");
    /// ```
    pub fn invalid_parameter(message: impl Into<String>) -> Self {
        Self::InvalidParameter {
            message: message.into(),
        }
    }

    /// Create a new validation error with failure details
    ///
    /// Convenience constructor for input validation failures.
    /// Use this when input fails validation checks or business rules.
    ///
    /// # Arguments
    ///
    /// * `message` - Detailed validation failure message
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::candle_completion::error::CandleCompletionError;
    ///
    /// let error = CandleCompletionError::validation_error("Content contains prohibited terms");
    /// ```
    pub fn validation_error(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create a new internal error with debugging information
    ///
    /// Convenience constructor for unexpected internal errors.
    /// Use this for system errors that are not user-recoverable.
    ///
    /// # Arguments
    ///
    /// * `message` - Internal error details for debugging
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::candle_completion::error::CandleCompletionError;
    ///
    /// let error = CandleCompletionError::internal_error("Unexpected state in processor chain");
    /// ```
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Create a new model loading failed error with failure reason
    ///
    /// Convenience constructor for model loading failures.
    /// Use this when models cannot be loaded from storage or initialized.
    ///
    /// # Arguments
    ///
    /// * `message` - Specific reason for the model loading failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::candle_completion::error::CandleCompletionError;
    ///
    /// let error = CandleCompletionError::model_loading_failed("Model file corrupted");
    /// ```
    pub fn model_loading_failed(message: impl Into<String>) -> Self {
        Self::ModelLoadingFailed {
            message: message.into(),
        }
    }

    /// Create a new rate limited error with retry timing
    ///
    /// Convenience constructor for rate limiting scenarios.
    /// Use this when requests are rejected due to rate limiting policies.
    ///
    /// # Arguments
    ///
    /// * `message` - Rate limiting message with details
    /// * `retry_after` - Number of seconds to wait before retrying
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::candle_completion::error::CandleCompletionError;
    ///
    /// let error = CandleCompletionError::rate_limited("Too many requests", 60);
    /// ```
    pub fn rate_limited(message: impl Into<String>, retry_after: u64) -> Self {
        Self::RateLimited {
            message: message.into(),
            retry_after,
        }
    }
}

impl CandleExtractionError {
    /// Create a new parse error with detailed message
    ///
    /// Convenience constructor for response parsing failures.
    /// Use this when model responses cannot be parsed into expected formats.
    ///
    /// # Arguments
    ///
    /// * `message` - Detailed parsing error message
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::candle_completion::error::CandleExtractionError;
    ///
    /// let error = CandleExtractionError::parse_error("Invalid JSON at line 5");
    /// ```
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::ParseError {
            message: message.into(),
        }
    }

    /// Create a new missing field error with field name
    ///
    /// Convenience constructor for missing required fields in extracted data.
    /// Use this when expected fields are absent from parsed responses.
    ///
    /// # Arguments
    ///
    /// * `field` - Name of the missing required field
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::candle_completion::error::CandleExtractionError;
    ///
    /// let error = CandleExtractionError::missing_field("user_id");
    /// ```
    pub fn missing_field(field: impl Into<String>) -> Self {
        Self::MissingField {
            field: field.into(),
        }
    }
}

// From trait implementations for error conversion

impl From<crate::model::error::ModelError> for CandleCompletionError {
    fn from(err: crate::model::error::ModelError) -> Self {
        match err {
            crate::model::error::ModelError::Validation(validation_err) => Self::Validation {
                message: validation_err.to_string()},
            crate::model::error::ModelError::Io(io_err) => Self::Internal {
                message: format!("IO error: {}", io_err)},
            crate::model::error::ModelError::Serialization(serde_err) => Self::Internal {
                message: format!("Serialization error: {}", serde_err)},
            crate::model::error::ModelError::Candle { message } => Self::InferenceError { message },
            crate::model::error::ModelError::InitializationFailed { reason } => {
                Self::ModelLoadingFailed { message: reason }
            }
            crate::model::error::ModelError::Inference { message } => {
                Self::InferenceError { message }
            }
            crate::model::error::ModelError::Memory { message } => Self::MemoryError { message },
            crate::model::error::ModelError::Device { message } => Self::DeviceError { message },
            crate::model::error::ModelError::InvalidConfiguration(msg) => Self::InvalidRequest {
                message: msg.to_string()},
            crate::model::error::ModelError::OperationNotSupported(msg) => {
                Self::UnsupportedOperation {
                    operation: msg.to_string()}
            }
            crate::model::error::ModelError::ModelNotFound { provider, name } => {
                Self::ModelLoadingFailed {
                    message: format!("Model not found: {}::{}", provider, name)}
            }
            crate::model::error::ModelError::ModelAlreadyExists { name } => Self::InvalidRequest {
                message: format!("Model already exists: {}", name)}}
    }
}

/// Convert ModelError to CandleCompletionError

impl From<crate::types::candle_model::error::ModelError> for CandleCompletionError {
    fn from(error: crate::types::candle_model::error::ModelError) -> Self {
        match error {
            crate::types::candle_model::error::ModelError::ModelNotFound { provider, name } => {
                Self::InvalidRequest {
                    message: format!("Model not found: {}:{}", provider, name)}
            }
            crate::types::candle_model::error::ModelError::ModelAlreadyExists {
                provider,
                name} => Self::InvalidRequest {
                message: format!("Model already exists: {}:{}", provider, name)},
            crate::types::candle_model::error::ModelError::ProviderNotFound(provider) => {
                Self::InvalidRequest {
                    message: format!("Provider not found: {}", provider)}
            }
            crate::types::candle_model::error::ModelError::InvalidConfiguration(msg) => {
                Self::InvalidParameter {
                    message: msg.to_string()}
            }
            crate::types::candle_model::error::ModelError::OperationNotSupported(msg) => {
                Self::UnsupportedOperation {
                    operation: msg.to_string()}
            }
            crate::types::candle_model::error::ModelError::InvalidInput(msg) => {
                Self::InvalidParameter {
                    message: msg.to_string()}
            }
            crate::types::candle_model::error::ModelError::Internal(msg) => Self::Internal {
                message: msg.to_string()}}
    }
}
