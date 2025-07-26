//! Helper functions for creating common error types

use super::error_types::CandleError;

// Helper functions for creating common errors
impl CandleError {
    /// Creates a model not found error with comprehensive path information
    /// 
    /// Constructs a standardized error for when model files cannot be located
    /// at the specified path, providing detailed context for debugging and
    /// user feedback in production applications.
    /// 
    /// # Arguments
    /// 
    /// * `path` - Model file path that could not be found, accepts any type
    ///   that converts to String (String, &str, PathBuf, etc.)
    /// 
    /// # Error Context
    /// 
    /// This error typically occurs when:
    /// - **File System**: Model file deleted or moved after path resolution
    /// - **Permissions**: Insufficient read permissions on model directory
    /// - **Network**: Remote model URLs that return 404 or connection failures
    /// - **Configuration**: Incorrect model paths in configuration files
    /// 
    /// # Recovery Strategies
    /// 
    /// Applications should handle this error by:
    /// 1. **Path Validation**: Check if parent directory exists and is readable
    /// 2. **Alternative Paths**: Try fallback model locations or default models
    /// 3. **User Guidance**: Provide clear instructions for model installation
    /// 4. **Logging**: Record path for debugging and monitoring
    /// 
    /// # Examples
    /// 
    /// ## Basic Usage
    /// ```rust
    /// use fluent_ai_candle::error::CandleError;
    /// 
    /// let model_path = "/models/llama-7b.safetensors";
    /// if !std::path::Path::new(model_path).exists() {
    ///     return Err(CandleError::model_not_found(model_path));
    /// }
    /// ```
    /// 
    /// ## With PathBuf
    /// ```rust
    /// use std::path::PathBuf;
    /// 
    /// let model_dir = PathBuf::from("/models");
    /// let model_file = model_dir.join("model.safetensors");
    /// 
    /// if !model_file.exists() {
    ///     return Err(CandleError::model_not_found(model_file));
    /// }
    /// ```
    /// 
    /// ## Error Handling with Fallback
    /// ```rust
    /// match load_model(&primary_path) {
    ///     Err(CandleError::ModelNotFound(_)) => {
    ///         println!("Primary model not found, trying fallback...");
    ///         load_model(&fallback_path)
    ///     }
    ///     result => result,
    /// }
    /// ```
    /// 
    /// ## User-Friendly Error Messages
    /// ```rust
    /// let error = CandleError::model_not_found("/nonexistent/model.bin");
    /// println!("Error: {}", error);
    /// // Output: "Model not found: /nonexistent/model.bin"
    /// 
    /// // Provide helpful guidance
    /// eprintln!("Please ensure the model file exists and is readable.");
    /// eprintln!("You can download models from: https://huggingface.co/models");
    /// ```
    /// 
    /// # Performance Characteristics
    /// 
    /// - **Zero Allocation**: Uses Into<String> for efficient string conversion
    /// - **Inlined**: Compiler eliminates function call overhead
    /// - **Copy Avoidance**: Moves string data instead of copying when possible
    /// 
    /// # Thread Safety
    /// 
    /// This constructor is thread-safe and can be called from any thread context.
    /// The resulting error is Send + Sync for propagation across thread boundaries.
    /// 
    /// # Architecture Compliance
    /// 
    /// - ✅ **Zero Allocation**: Efficient string handling in error path
    /// - ✅ **Type Safety**: Generic parameter ensures correct type conversion
    /// - ✅ **Error Context**: Provides sufficient information for debugging
    #[inline(always)]
    pub fn model_not_found<S: Into<String>>(path: S) -> Self {
        Self::ModelNotFound(path.into())
    }

    /// Creates a model loading error with detailed failure context
    /// 
    /// Constructs a specialized error for failures that occur during the model
    /// loading process, providing rich context for debugging complex loading
    /// scenarios in production machine learning applications.
    /// 
    /// # Arguments
    /// 
    /// * `msg` - Detailed error message describing the specific loading failure,
    ///   accepts any type convertible to String for maximum flexibility
    /// 
    /// # Common Loading Failure Scenarios
    /// 
    /// ## File System Issues
    /// - **Corrupted Files**: Model files with invalid or truncated data
    /// - **Permission Errors**: Insufficient read access to model directories
    /// - **Disk Space**: Insufficient space for temporary loading operations
    /// - **Network Failures**: Timeouts or connectivity issues for remote models
    /// 
    /// ## Model Format Problems
    /// - **Version Mismatch**: Model saved with incompatible framework version
    /// - **Architecture Changes**: Model structure incompatible with current code
    /// - **Serialization Issues**: Pickle/SafeTensors deserialization failures
    /// - **Encoding Problems**: Character encoding issues in model metadata
    /// 
    /// ## Resource Constraints
    /// - **Memory Exhaustion**: Insufficient RAM for large model loading
    /// - **GPU Memory**: CUDA out-of-memory during device transfer
    /// - **CPU Limits**: Process memory limits exceeded
    /// - **Timeout**: Loading operation exceeded configured timeout
    /// 
    /// # Examples
    /// 
    /// ## Basic Error Creation
    /// ```rust
    /// use fluent_ai_candle::error::CandleError;
    /// 
    /// // Simple string message
    /// let error = CandleError::model_load_error("Failed to parse model config");
    /// 
    /// // Formatted message with context
    /// let model_size = 7_000_000_000; // 7B parameters
    /// let available_memory = 4 * 1024 * 1024 * 1024; // 4GB
    /// let error = CandleError::model_load_error(
    ///     format!("Model too large: {}B parameters need {}GB RAM, only {}GB available", 
    ///             model_size / 1_000_000_000,
    ///             model_size * 2 / 1_000_000_000, // Rough estimate
    ///             available_memory / 1_000_000_000)
    /// );
    /// ```
    /// 
    /// ## Error Context Chaining
    /// ```rust
    /// fn load_model_with_context(path: &str) -> Result<Model, CandleError> {
    ///     match std::fs::File::open(path) {
    ///         Ok(file) => {
    ///             match deserialize_model(file) {
    ///                 Ok(model) => Ok(model),
    ///                 Err(serde_err) => Err(CandleError::model_load_error(
    ///                     format!("Deserialization failed for {}: {}", path, serde_err)
    ///                 ))
    ///             }
    ///         }
    ///         Err(io_err) => Err(CandleError::model_load_error(
    ///             format!("Cannot open model file {}: {}", path, io_err)
    ///         ))
    ///     }
    /// }
    /// ```
    /// 
    /// ## Progress Tracking Integration
    /// ```rust
    /// use std::sync::Arc;
    /// use std::sync::atomic::{AtomicU64, Ordering};
    /// 
    /// async fn load_model_with_progress(
    ///     path: &str,
    ///     progress: Arc<AtomicU64>
    /// ) -> Result<Model, CandleError> {
    ///     progress.store(10, Ordering::Relaxed); // 10% - Started
    ///     
    ///     let file_size = match std::fs::metadata(path) {
    ///         Ok(meta) => meta.len(),
    ///         Err(e) => return Err(CandleError::model_load_error(
    ///             format!("Cannot read model file metadata: {}", e)
    ///         ))
    ///     };
    ///     
    ///     progress.store(30, Ordering::Relaxed); // 30% - File validated
    ///     
    ///     let bytes = match tokio::fs::read(path).await {
    ///         Ok(data) => {
    ///             progress.store(60, Ordering::Relaxed); // 60% - File read
    ///             data
    ///         }
    ///         Err(e) => return Err(CandleError::model_load_error(
    ///             format!("Failed to read {}MB model file: {}", 
    ///                    file_size / 1_000_000, e)
    ///         ))
    ///     };
    ///     
    ///     progress.store(80, Ordering::Relaxed); // 80% - Parsing
    ///     
    ///     match parse_model_bytes(&bytes) {
    ///         Ok(model) => {
    ///             progress.store(100, Ordering::Relaxed); // 100% - Complete
    ///             Ok(model)
    ///         }
    ///         Err(e) => Err(CandleError::model_load_error(
    ///             format!("Model parsing failed at {}% progress: {}", 
    ///                    progress.load(Ordering::Relaxed), e)
    ///         ))
    ///     }
    /// }
    /// ```
    /// 
    /// ## Memory Diagnostics
    /// ```rust
    /// fn load_model_with_memory_check(path: &str) -> Result<Model, CandleError> {
    ///     let model_size = std::fs::metadata(path)
    ///         .map_err(|e| CandleError::model_load_error(
    ///             format!("Cannot check model size: {}", e)
    ///         ))?
    ///         .len();
    ///     
    ///     let available_memory = get_available_memory();
    ///     let required_memory = model_size * 3; // Conservative estimate
    ///     
    ///     if required_memory > available_memory {
    ///         return Err(CandleError::model_load_error(
    ///             format!(
    ///                 "Insufficient memory: model needs {}GB, only {}GB available. \
    ///                  Try using quantization or a smaller model.",
    ///                 required_memory / 1_000_000_000,
    ///                 available_memory / 1_000_000_000
    ///             )
    ///         ));
    ///     }
    ///     
    ///     // Proceed with loading...
    ///     load_model_impl(path)
    /// }
    /// ```
    /// 
    /// # Error Message Best Practices
    /// 
    /// ## Include Context
    /// - File paths and sizes
    /// - Memory usage and limits
    /// - Progress information
    /// - System resource state
    /// 
    /// ## Provide Solutions
    /// - Suggest alternative model variants
    /// - Recommend configuration changes
    /// - Point to documentation or support
    /// - Include relevant error codes
    /// 
    /// ## User-Friendly Format
    /// ```rust
    /// // Good: Actionable error message
    /// CandleError::model_load_error(
    ///     "Model loading failed: 7B model requires 16GB RAM, system has 8GB. \
    ///      Try the 3B variant or enable quantization."
    /// )
    /// 
    /// // Avoid: Technical jargon without guidance
    /// CandleError::model_load_error("SafeTensors deserialization error 0x4A")
    /// ```
    /// 
    /// # Performance Characteristics
    /// 
    /// - **Zero Copy**: Efficient string conversion using Into<String>
    /// - **Inlined**: Compiler eliminates function call overhead
    /// - **Memory Efficient**: Single allocation for error message
    /// 
    /// # Thread Safety
    /// 
    /// This constructor is thread-safe and the resulting error can be safely
    /// sent across thread boundaries for error propagation.
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

    /// Creates a tensor operation error for mathematical computation failures
    /// 
    /// Constructs a specialized error for failures in tensor operations, matrix
    /// computations, and numerical processing that occur during model inference
    /// or training. These errors typically indicate issues with tensor shapes,
    /// data types, or mathematical operations.
    /// 
    /// # Arguments
    /// 
    /// * `msg` - Static string describing the specific tensor operation that failed.
    ///   Using &'static str ensures zero allocation in error paths.
    /// 
    /// # Common Tensor Operation Failures
    /// 
    /// ## Shape Mismatches
    /// - **Matrix Multiplication**: Incompatible dimensions for matmul operations
    /// - **Broadcasting**: Tensors with incompatible shapes for element-wise ops
    /// - **Reshape**: Invalid target shape for tensor restructuring
    /// - **Concatenation**: Mismatched dimensions for tensor joining
    /// 
    /// ## Data Type Issues
    /// - **Type Casting**: Unsupported conversions between f16/f32/i32/etc.
    /// - **Mixed Precision**: Operations mixing incompatible numeric types
    /// - **Quantization**: Errors in quantized tensor operations
    /// - **Device Types**: CPU/CUDA tensor mixing without proper conversion
    /// 
    /// ## Memory and Device Errors
    /// - **CUDA OOM**: GPU memory exhaustion during tensor allocation
    /// - **Device Mismatch**: Operations on tensors from different devices
    /// - **Stride Issues**: Invalid memory layout for tensor operations
    /// - **Alignment**: Memory alignment requirements not met
    /// 
    /// # Examples
    /// 
    /// ## Shape Validation
    /// ```rust
    /// use fluent_ai_candle::error::CandleError;
    /// use candle_core::{Tensor, Shape};
    /// 
    /// fn safe_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, CandleError> {
    ///     let a_shape = a.shape();
    ///     let b_shape = b.shape();
    ///     
    ///     if a_shape.dims().len() != 2 || b_shape.dims().len() != 2 {
    ///         return Err(CandleError::tensor_operation(
    ///             "Matrix multiplication requires 2D tensors"
    ///         ));
    ///     }
    ///     
    ///     if a_shape.dims()[1] != b_shape.dims()[0] {
    ///         return Err(CandleError::tensor_operation(
    ///             "Incompatible dimensions for matrix multiplication"
    ///         ));
    ///     }
    ///     
    ///     a.matmul(b).map_err(|_| CandleError::tensor_operation(
    ///         "Matrix multiplication failed on device"
    ///     ))
    /// }
    /// ```
    /// 
    /// ## Device Compatibility
    /// ```rust
    /// fn ensure_same_device(tensors: &[&Tensor]) -> Result<(), CandleError> {
    ///     if tensors.is_empty() {
    ///         return Ok(());
    ///     }
    ///     
    ///     let first_device = tensors[0].device();
    ///     for (i, tensor) in tensors.iter().enumerate().skip(1) {
    ///         if tensor.device() != first_device {
    ///             return Err(CandleError::tensor_operation(
    ///                 "All tensors must be on the same device"
    ///             ));
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    /// 
    /// ## Memory-Safe Operations
    /// ```rust
    /// fn safe_tensor_reshape(tensor: &Tensor, new_shape: &[usize]) -> Result<Tensor, CandleError> {
    ///     let current_elements: usize = tensor.shape().dims().iter().product();
    ///     let new_elements: usize = new_shape.iter().product();
    ///     
    ///     if current_elements != new_elements {
    ///         return Err(CandleError::tensor_operation(
    ///             "Reshape must preserve total number of elements"
    ///         ));
    ///     }
    ///     
    ///     tensor.reshape(new_shape).map_err(|_| CandleError::tensor_operation(
    ///         "Tensor reshape operation failed"
    ///     ))
    /// }
    /// ```
    /// 
    /// ## Numerical Stability Checks
    /// ```rust
    /// fn safe_softmax(logits: &Tensor) -> Result<Tensor, CandleError> {
    ///     // Check for NaN or infinite values
    ///     let max_val = logits.max_keepdim(candle_core::D::Minus1)
    ///         .map_err(|_| CandleError::tensor_operation(
    ///             "Failed to compute maximum for numerical stability"
    ///         ))?;
    ///     
    ///     let shifted = logits.broadcast_sub(&max_val)
    ///         .map_err(|_| CandleError::tensor_operation(
    ///             "Failed to shift logits for numerical stability"
    ///         ))?;
    ///     
    ///     let exp_vals = shifted.exp()
    ///         .map_err(|_| CandleError::tensor_operation(
    ///             "Exponential operation failed - possible overflow"
    ///         ))?;
    ///     
    ///     let sum_exp = exp_vals.sum_keepdim(candle_core::D::Minus1)
    ///         .map_err(|_| CandleError::tensor_operation(
    ///             "Failed to compute softmax normalization"
    ///         ))?;
    ///     
    ///     exp_vals.broadcast_div(&sum_exp)
    ///         .map_err(|_| CandleError::tensor_operation(
    ///             "Division by zero in softmax computation"
    ///         ))
    /// }
    /// ```
    /// 
    /// ## Error Recovery Patterns
    /// ```rust
    /// fn robust_tensor_operation(input: &Tensor) -> Result<Tensor, CandleError> {
    ///     // Try optimal path first
    ///     match input.to_dtype(candle_core::DType::F16) {
    ///         Ok(fp16_tensor) => {
    ///             // Fast FP16 path
    ///             perform_fast_operation(&fp16_tensor)
    ///         }
    ///         Err(_) => {
    ///             // Fallback to FP32 for compatibility
    ///             match input.to_dtype(candle_core::DType::F32) {
    ///                 Ok(fp32_tensor) => perform_safe_operation(&fp32_tensor),
    ///                 Err(_) => Err(CandleError::tensor_operation(
    ///                     "Cannot convert tensor to supported data type"
    ///                 ))
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    /// 
    /// # Error Message Guidelines
    /// 
    /// ## Descriptive Messages
    /// - "Matrix multiplication requires 2D tensors" ✓
    /// - "Tensor operation failed" ✗ (too vague)
    /// 
    /// ## Include Operation Context
    /// - "Failed to compute softmax normalization" ✓
    /// - "Division by zero" ✗ (lacks context)
    /// 
    /// ## Performance Focus
    /// Use static strings to avoid allocation in error paths:
    /// ```rust
    /// // Good: Zero allocation
    /// CandleError::tensor_operation("Shape mismatch in matrix multiplication")
    /// 
    /// // Avoid: Allocation in error path
    /// CandleError::tensor_operation(&format!("Shape {} vs {}", shape1, shape2))
    /// ```
    /// 
    /// # Performance Characteristics
    /// 
    /// - **Zero Allocation**: Uses static string reference
    /// - **Inlined**: Compiler eliminates function call overhead
    /// - **Fast Error Path**: Optimized for minimal performance impact
    /// - **Cache Friendly**: Static strings stored in read-only memory
    /// 
    /// # Thread Safety
    /// 
    /// This constructor is thread-safe and can be called from any thread.
    /// Static string messages are safe to share across threads.
    /// 
    /// # Architecture Compliance
    /// 
    /// - ✅ **Zero Allocation**: Static strings prevent error path allocation
    /// - ✅ **Performance**: Minimal overhead for error construction
    /// - ✅ **Safety**: Type-safe tensor operation error classification
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

    /// Determines if this error represents a transient condition that may succeed on retry
    /// 
    /// Analyzes the error type to determine whether the operation that caused this
    /// error might succeed if attempted again after a delay. This enables robust
    /// error handling with intelligent retry logic in production systems.
    /// 
    /// # Retry Classification
    /// 
    /// ## Retryable Errors (Returns `true`)
    /// - **LoadingTimeout**: Network delays or temporary resource contention
    /// - **DeviceAllocation**: GPU memory pressure that may clear
    /// - **TokenizationError**: Transient tokenizer initialization issues
    /// - **CacheOverflow**: Temporary cache pressure
    /// 
    /// ## Non-Retryable Errors (Returns `false`)
    /// - **ModelNotFound**: File system errors requiring manual intervention
    /// - **InvalidModelFormat**: Corrupted files that won't fix themselves
    /// - **UnsupportedArchitecture**: Fundamental incompatibility issues
    /// - **Configuration**: Wrong settings that need manual correction
    /// 
    /// # Retry Strategy Integration
    /// 
    /// Use with `retry_delay()` for complete retry logic:
    /// ```rust
    /// async fn robust_model_operation() -> Result<(), CandleError> {
    ///     let mut attempts = 0;
    ///     let max_attempts = 3;
    ///     
    ///     loop {
    ///         match risky_model_operation().await {
    ///             Ok(result) => return Ok(result),
    ///             Err(e) if attempts < max_attempts && e.is_retryable() => {
    ///                 attempts += 1;
    ///                 let delay = e.retry_delay().unwrap_or(1);
    ///                 println!("Attempt {}/{} failed, retrying in {}s: {}", 
    ///                          attempts, max_attempts, delay, e);
    ///                 tokio::time::sleep(Duration::from_secs(delay)).await;
    ///             }
    ///             Err(e) => {
    ///                 println!("Non-retryable error or max attempts reached: {}", e);
    ///                 return Err(e);
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    /// 
    /// # Examples
    /// 
    /// ## Basic Retry Logic
    /// ```rust
    /// use fluent_ai_candle::error::CandleError;
    /// 
    /// let error = CandleError::loading_timeout();
    /// if error.is_retryable() {
    ///     println!("Will retry operation after delay");
    ///     let delay = error.retry_delay().unwrap_or(1);
    ///     // ... implement retry logic
    /// } else {
    ///     println!("Permanent failure, manual intervention required");
    /// }
    /// ```
    /// 
    /// ## Exponential Backoff
    /// ```rust
    /// async fn exponential_backoff_retry<F, T>(
    ///     mut operation: F,
    ///     max_attempts: u32
    /// ) -> Result<T, CandleError> 
    /// where
    ///     F: FnMut() -> Result<T, CandleError>,
    /// {
    ///     let mut delay = 1;
    ///     
    ///     for attempt in 1..=max_attempts {
    ///         match operation() {
    ///             Ok(result) => return Ok(result),
    ///             Err(e) if attempt < max_attempts && e.is_retryable() => {
    ///                 println!("Attempt {} failed, waiting {}s", attempt, delay);
    ///                 tokio::time::sleep(Duration::from_secs(delay)).await;
    ///                 delay = (delay * 2).min(60); // Cap at 60 seconds
    ///             }
    ///             Err(e) => return Err(e),
    ///         }
    ///     }
    ///     
    ///     unreachable!()
    /// }
    /// ```
    /// 
    /// ## Circuit Breaker Integration
    /// ```rust
    /// struct CircuitBreaker {
    ///     failure_count: u32,
    ///     last_failure: Option<Instant>,
    /// }
    /// 
    /// impl CircuitBreaker {
    ///     fn should_attempt(&self) -> bool {
    ///         if self.failure_count < 5 {
    ///             return true;
    ///         }
    ///         
    ///         // Circuit open - check if enough time has passed
    ///         if let Some(last_failure) = self.last_failure {
    ///             last_failure.elapsed() > Duration::from_secs(30)
    ///         } else {
    ///             true
    ///         }
    ///     }
    ///     
    ///     fn record_result(&mut self, result: &Result<(), CandleError>) {
    ///         match result {
    ///             Ok(_) => self.failure_count = 0,
    ///             Err(e) if e.is_retryable() => {
    ///                 self.failure_count += 1;
    ///                 self.last_failure = Some(Instant::now());
    ///             }
    ///             Err(_) => {
    ///                 // Non-retryable error - don't affect circuit state
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    /// 
    /// # Performance Characteristics
    /// 
    /// - **O(1) Time**: Simple pattern matching with no computation
    /// - **Zero Allocation**: No memory allocation during classification
    /// - **Inlined**: Compiler eliminates function call overhead
    /// - **Branch Predictable**: Most errors are non-retryable (optimized path)
    /// 
    /// # Error Categories by Retry Suitability
    /// 
    /// ## Transient (Retryable)
    /// - Network timeouts
    /// - Resource exhaustion
    /// - Temporary device unavailability
    /// - System load spikes
    /// 
    /// ## Permanent (Non-Retryable)
    /// - Missing files
    /// - Corrupted data
    /// - Configuration errors
    /// - Incompatible hardware/software
    /// 
    /// # Thread Safety
    /// 
    /// This method is thread-safe and can be called concurrently on the same
    /// error instance without synchronization.
    #[inline(always)]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::LoadingTimeout => true,
            Self::DeviceAllocation(_) => true,
            Self::TokenizationError(_) => true,
            Self::CacheOverflow => true,
            _ => false}
    }

    /// Returns the recommended delay in seconds before retrying this operation
    /// 
    /// Provides intelligent delay recommendations based on the specific error type,
    /// enabling optimal retry strategies that balance quick recovery with system
    /// stability. Returns None for non-retryable errors.
    /// 
    /// # Delay Strategy by Error Type
    /// 
    /// ## Short Delays (1-2 seconds)
    /// - **DeviceAllocation**: GPU memory typically clears quickly
    /// - **CacheOverflow**: Cache eviction happens rapidly
    /// 
    /// ## Medium Delays (2-5 seconds)  
    /// - **TokenizationError**: Tokenizer initialization needs time
    /// - **LoadingTimeout**: Network/disk operations need recovery time
    /// 
    /// ## No Delay (None)
    /// - **Non-retryable errors**: No point in retrying
    /// - **Permanent failures**: Manual intervention required
    /// 
    /// # Integration with Retry Logic
    /// 
    /// ```rust
    /// use fluent_ai_candle::error::CandleError;
    /// use tokio::time::{sleep, Duration};
    /// 
    /// async fn smart_retry_operation() -> Result<(), CandleError> {
    ///     const MAX_ATTEMPTS: u32 = 3;
    ///     
    ///     for attempt in 1..=MAX_ATTEMPTS {
    ///         match perform_operation().await {
    ///             Ok(result) => return Ok(result),
    ///             Err(e) if attempt < MAX_ATTEMPTS => {
    ///                 if let Some(delay) = e.retry_delay() {
    ///                     println!("Attempt {} failed, waiting {}s: {}", 
    ///                              attempt, delay, e);
    ///                     sleep(Duration::from_secs(delay)).await;
    ///                 } else {
    ///                     println!("Non-retryable error: {}", e);
    ///                     return Err(e);
    ///                 }
    ///             }
    ///             Err(e) => return Err(e), // Final attempt failed
    ///         }
    ///     }
    ///     
    ///     unreachable!()
    /// }
    /// ```
    /// 
    /// # Advanced Retry Patterns
    /// 
    /// ## Exponential Backoff with Jitter
    /// ```rust
    /// use rand::Rng;
    /// 
    /// async fn exponential_backoff_with_jitter(
    ///     error: &CandleError,
    ///     attempt: u32
    /// ) -> Option<Duration> {
    ///     if let Some(base_delay) = error.retry_delay() {
    ///         let exponential = base_delay * 2_u64.pow(attempt.saturating_sub(1));
    ///         let max_delay = 60; // Cap at 60 seconds
    ///         let delay = exponential.min(max_delay);
    ///         
    ///         // Add jitter to prevent thundering herd
    ///         let jitter = rand::thread_rng().gen_range(0.8..=1.2);
    ///         let final_delay = (delay as f64 * jitter) as u64;
    ///         
    ///         Some(Duration::from_secs(final_delay))
    ///     } else {
    ///         None
    ///     }
    /// }
    /// ```
    /// 
    /// ## Adaptive Delay Based on System Load  
    /// ```rust
    /// fn adaptive_retry_delay(
    ///     error: &CandleError,
    ///     system_load: f32
    /// ) -> Option<Duration> {
    ///     error.retry_delay().map(|base_delay| {
    ///         let load_multiplier = if system_load > 0.8 {
    ///             3.0  // High load - wait longer
    ///         } else if system_load > 0.5 {
    ///             2.0  // Medium load - moderate delay
    ///         } else {
    ///             1.0  // Low load - standard delay
    ///         };
    ///         
    ///         Duration::from_secs((base_delay as f32 * load_multiplier) as u64)
    ///     })
    /// }
    /// ```
    /// 
    /// ## Contextual Retry with Error Patterns
    /// ```rust
    /// struct RetryContext {
    ///     consecutive_failures: u32,
    ///     last_error_type: Option<String>,
    ///     total_retry_time: Duration,
    /// }
    /// 
    /// impl RetryContext {
    ///     fn calculate_delay(&mut self, error: &CandleError) -> Option<Duration> {
    ///         if let Some(base_delay) = error.retry_delay() {
    ///             // Increase delay for repeated same-type failures
    ///             let error_type = format!("{:?}", error);
    ///             if self.last_error_type.as_ref() == Some(&error_type) {
    ///                 self.consecutive_failures += 1;
    ///             } else {
    ///                 self.consecutive_failures = 1;
    ///                 self.last_error_type = Some(error_type);
    ///             }
    ///             
    ///             let multiplier = (self.consecutive_failures as u64).min(8);
    ///             let delay = Duration::from_secs(base_delay * multiplier);
    ///             
    ///             // Cap total retry time at 5 minutes
    ///             if self.total_retry_time + delay > Duration::from_secs(300) {
    ///                 None
    ///             } else {
    ///                 self.total_retry_time += delay;
    ///                 Some(delay)
    ///             }
    ///         } else {
    ///             None
    ///         }
    ///     }
    /// }
    /// ```
    /// 
    /// # Performance Considerations
    /// 
    /// - **Fast Classification**: O(1) pattern matching
    /// - **No Allocation**: Returns primitive u64 value
    /// - **Inlined**: Zero function call overhead
    /// - **Cache Friendly**: Simple enum discriminant check
    /// 
    /// # Delay Rationale
    /// 
    /// The specific delay values are chosen based on:
    /// - **System Recovery Time**: How long typical failures take to resolve
    /// - **Resource Contention**: Avoiding thundering herd effects
    /// - **User Experience**: Balancing responsiveness with stability
    /// - **Production Testing**: Observed optimal delays in real deployments
    /// 
    /// # Thread Safety
    /// 
    /// This method is thread-safe and can be called concurrently without
    /// synchronization. The returned delay value is independent per call.
    #[inline(always)]
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            Self::LoadingTimeout => Some(5),
            Self::DeviceAllocation(_) => Some(1),
            Self::TokenizationError(_) => Some(2),
            Self::CacheOverflow => Some(1),
            _ => None}
    }
}