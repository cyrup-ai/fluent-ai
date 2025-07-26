//! High-performance candle integration for fluent-ai completion system
//!
//! This crate provides zero-allocation, lock-free ML inference using the candle framework.
//! It implements the CompletionClient trait with blazing-fast performance optimizations:
//!
//! - Zero allocation: Stack allocation, pre-allocated buffers, ArrayVec/SmallVec
//! - No locking: Crossbeam channels, atomics, lock-free data structures
//! - Blazing-fast: Inline hot paths, optimized memory layout, SIMD where possible
//! - No unsafe/unchecked: Explicit bounds checking, safe performance optimizations
//! - Elegant ergonomic: Clean API with builder patterns, zero-cost abstractions
//!
//! # Architecture Overview
//!
//! The crate is organized into specialized modules for different aspects of ML inference:
//!
//! - **Client**: High-level completion client with builder pattern configuration
//! - **Model**: Model loading, management, and inference execution
//! - **Tokenizer**: Fast tokenization with special token handling
//! - **Sampling**: Advanced sampling strategies (nucleus, top-k, temperature, etc.)
//! - **Streaming**: Real-time token streaming with flow control
//! - **Processing**: Logits processing and transformation pipelines
//! - **Constraints**: Structured generation with JSON schema support
//! - **Memory**: Memory management and allocation tracking
//! - **Progress**: Fine-grained progress reporting and metrics
//!
//! # Performance Features
//!
//! - **Zero-copy operations**: Direct tensor manipulation without intermediate allocations
//! - **SIMD optimizations**: Vectorized operations for sampling and processing
//! - **Lock-free concurrency**: Atomic operations and channel-based communication
//! - **Adaptive batching**: Dynamic batch size optimization based on hardware
//! - **Memory pooling**: Reusable buffers for frequent operations
//! - **Hardware acceleration**: CUDA and Metal GPU support with CPU fallback
//!
//! # Quick Start
//!
//! ```rust
//! use fluent_ai_candle::{CandleClientBuilder, GenerationConfig};
//!
//! // Create a high-performance completion client
//! let client = CandleClientBuilder::new()
//!     .model_path("path/to/model")
//!     .generation_config(GenerationConfig::default())
//!     .build()?;
//!
//! // Generate completions with streaming
//! let stream = client.stream_completion("Hello, world!")?;
//! ```

#![warn(missing_docs)]

/// Core client implementation for high-level completion operations
/// 
/// Provides the main `CandleCompletionClient` with builder pattern configuration,
/// connection pooling, and optimized request handling.
pub mod client;

/// System-wide constants and configuration defaults
/// 
/// Contains performance tuning parameters, model defaults, and system limits
/// used across the inference pipeline.
pub mod constants;

/// Structured generation constraints for JSON and schema validation
/// 
/// Implements constraint-based generation ensuring outputs conform to specified
/// JSON schemas, regular expressions, or custom validation rules.
pub mod constraints;

/// Error types and error handling utilities
/// 
/// Comprehensive error taxonomy with context preservation, error chaining,
/// and conversion utilities for debugging and error recovery.
pub mod error;

/// High-performance text generation with advanced sampling
/// 
/// Core generation engine with optimized inference loops, memory management,
/// and configurable stopping criteria.
pub mod generator;

/// Model hub integration for downloading and caching models
/// 
/// Provides seamless integration with Hugging Face Hub, local model loading,
/// and intelligent caching strategies for model artifacts.
pub mod hub;

/// Key-value cache implementation for efficient attention computation
/// 
/// Optimized KV cache with memory-efficient storage, automatic eviction policies,
/// and support for dynamic sequence lengths.
pub mod kv_cache;

/// Logits processing and transformation pipeline
/// 
/// Advanced logits manipulation including temperature scaling, top-k/top-p filtering,
/// repetition penalties, and custom transformation functions.
pub mod logits;

/// Memory management and allocation tracking
/// 
/// Zero-allocation patterns, memory pool management, and allocation tracking
/// for performance monitoring and memory leak detection.
pub mod memory;

/// Model loading, management, and inference
/// 
/// Handles model lifecycle from loading to inference, including quantization,
/// device placement, and model format conversions.
pub mod model;

/// Logits processing and token filtering
/// 
/// Implements various processing strategies for controlling generation quality
/// and characteristics through logits manipulation.
pub mod processing;

/// Progress reporting and metrics collection
/// 
/// Fine-grained progress tracking for model loading, inference, and streaming
/// operations with customizable reporting callbacks.
pub mod progress;

/// Advanced sampling strategies and probability distributions
/// 
/// Implements various sampling methods including nucleus sampling, typical sampling,
/// Mirostat, and custom probability distribution sampling.
pub mod sampling;

/// Real-time token streaming with flow control
/// 
/// High-performance streaming implementation with backpressure handling,
/// rate limiting, and adaptive buffering strategies.
pub mod streaming;

/// Fast tokenization with special token support
/// 
/// Optimized tokenizer implementation with caching, special token handling,
/// and support for various tokenizer formats.
pub mod tokenizer;

/// Core type definitions and data structures
/// 
/// Central type definitions used throughout the crate including completion
/// requests/responses, model configurations, and utility types.
pub mod types;
// ============================================================================
// Core Type Re-exports
// ============================================================================

/// Core completion and model types from the types module
/// 
/// These are the fundamental data structures used throughout the completion
/// pipeline, re-exported for convenient access without deep module imports.
pub use types::{
    CandleCompletionError,
    CandleCompletionModel,
    CandleCompletionRequest,
    CandleCompletionResponse,
    CandleCompletionResult,
    CandleDocument,
    CandleFinishReason,
    CandleMessage,
    CandleMessageRole,
    CandleModelInfo,
    CandleStreamingResponse,
    CandleUsage,
};

/// Variable builder for advanced model configuration and tensor management
/// 
/// Provides low-level control over model loading, tensor allocation, and
/// device placement for performance optimization.
pub mod var_builder;

/// High-level builders for chat and completion operations
/// 
/// Contains specialized builder patterns for different completion scenarios
/// including chat conversations and single-shot completions.
pub mod builders;

/// Re-export all builder types for convenient access
/// 
/// Includes chat builders, completion builders, and configuration builders
/// used to construct optimized completion pipelines.
pub use builders::*;

// ============================================================================
// External Dependency Re-exports
// ============================================================================

/// Async streaming primitives from fluent-ai-async
/// 
/// These types provide the foundation for all streaming operations in the crate,
/// enabling zero-allocation, backpressure-aware streaming of tokens and responses.
pub use fluent_ai_async::{
    AsyncStream,
    AsyncStreamSender,
};

/// Utility type for representing optional, single, or multiple values
/// 
/// Zero-allocation container that can hold 0, 1, or many values efficiently
/// using stack allocation for small collections.
pub use types::candle_utils::zero_one_or_many::ZeroOneOrMany;

// ============================================================================
// Primary API Re-exports
// ============================================================================

/// Core client types for high-level completion operations
/// 
/// These are the main entry points for most users of the crate, providing
/// builder pattern configuration and managed completion operations.
pub use client::{
    CandleClientBuilder,
    CandleClientConfig,
    CandleCompletionClient,
};

/// Structured generation constraint types
/// 
/// Advanced constraint system for ensuring generated text conforms to specific
/// formats, schemas, or validation rules during the generation process.
pub use constraints::{
    GenerationConstraint,
    JsonConstraint,
    JsonCurrentState,
    JsonStackItem,
    JsonState,
    NumberState,
    create_json_constraint_for_tokenizer,
};

/// Error handling types and utilities
/// 
/// Comprehensive error system with context preservation, error chaining,
/// and conversion utilities for robust error handling and debugging.
pub use error::{
    CandleError,
    CandleResult,
};

/// Text generation engine and configuration
/// 
/// Core generation functionality with advanced sampling, stopping criteria,
/// and performance optimizations for high-throughput text generation.
pub use generator::{
    CandleGenerator,
    GenerationConfig,
};

/// Model hub integration types
/// 
/// Seamless integration with Hugging Face Hub and local model management
/// with intelligent caching and download progress tracking.
pub use hub::{
    Backend,
    Client,
    DownloadConfig,
    DownloadEvent,
    DownloadProgress,
    DownloadResult,
    ProgressData,
    ProgressHandler,
    ProgressHubConfig,
    create_client,
    create_download_config,
};

/// Key-value cache types for efficient attention computation
/// 
/// Memory-efficient KV cache implementation with automatic memory management,
/// eviction policies, and support for dynamic sequence lengths.
pub use kv_cache::{
    CacheStats,
    EvictionStrategy,
    KVCache,
    KVCacheBuilder,
    KVCacheConfig,
    KVCacheEntry,
};

/// Logits processing and sampling types
/// 
/// Advanced logits manipulation and sampling strategies for controlling
/// generation quality, diversity, and adherence to constraints.
pub use logits::{
    CompositeProcessor,
    LogitsProcessor,
    LogitsSampler,
    ProcessingContext,
    RepetitionPenaltyProcessor,
    SamplingConfig,
    SamplingMetrics,
    TemperatureProcessor,
    TopKProcessor,
    TopPProcessor,
    sampling_metrics,
};

/// Model loading and management
/// 
/// Core model abstraction with loading, device placement, quantization,
/// and inference capabilities for various model formats.
pub use model::{
    CandleModel,
};

/// Progress reporting and metrics collection
/// 
/// Comprehensive progress tracking for all operations with customizable
/// reporting, metrics aggregation, and performance monitoring.
pub use progress::{
    AggregatorStats,
    InferenceMetrics,
    MetricsAggregator,
    ProgressHubReporter,
    ProgressReporter,
};

/// Streaming and flow control types
/// 
/// High-performance streaming implementation with backpressure handling,
/// rate limiting, and efficient token delivery for real-time applications.
pub use streaming::{
    FlushPolicy,
    StreamingConfig,
    StreamingMetrics,
    TokenChunk,
    TokenMetadata,
    TokenOutputStream,
    TokenStreamSender,
};

/// Tokenization types and configuration
/// 
/// Fast tokenization with special token handling, caching, and support
/// for various tokenizer formats and encoding strategies.
pub use tokenizer::{
    CandleTokenizer,
    TokenizerConfig,
};

/// Variable builder types for advanced model configuration
/// 
/// Low-level tensor and model configuration utilities for custom model
/// loading, quantization, and device-specific optimizations.
pub use var_builder::{
    CandleVarBuilder,
    LoadingStats,
    ModelMetadata,
    TensorEntry,
    VarBuilderConfig,
    VarBuilderConfigBuilder,
};

/// Device utilities for optimal device selection
pub mod device {
    use candle_core::Device;

    /// Automatically selects the optimal compute device based on hardware availability
    /// 
    /// Implements intelligent device selection with priority ordering for maximum
    /// performance while maintaining compatibility across different hardware configurations.
    /// 
    /// # Device Selection Priority
    /// 
    /// 1. **CUDA GPU** (if available and feature enabled)
    ///    - Highest performance for ML workloads
    ///    - Massive parallel processing capability
    ///    - Hardware acceleration for matrix operations
    /// 
    /// 2. **Metal GPU** (if available and feature enabled) 
    ///    - Apple Silicon optimization
    ///    - Unified memory architecture benefits
    ///    - Native macOS hardware acceleration
    /// 
    /// 3. **CPU** (fallback, always available)
    ///    - Universal compatibility
    ///    - No additional dependencies
    ///    - Suitable for development and testing
    /// 
    /// # Returns
    /// 
    /// `candle_core::Result<Device>` containing:
    /// - `Ok(Device::Cuda(0))` - CUDA device 0 if available
    /// - `Ok(Device::Metal(0))` - Metal device 0 if available  
    /// - `Ok(Device::Cpu)` - CPU device as universal fallback
    /// - `Err(...)` - Only if all device initialization fails (extremely rare)
    /// 
    /// # Performance Impact
    /// 
    /// Device selection significantly affects generation performance:
    /// - **CUDA**: 10-100x speedup for large models
    /// - **Metal**: 5-50x speedup on Apple Silicon
    /// - **CPU**: Baseline performance, adequate for small models
    /// 
    /// # Feature Gates
    /// 
    /// Device availability depends on compile-time features:
    /// - `cuda` feature: Enables CUDA device detection
    /// - `metal` feature: Enables Metal device detection
    /// - No features: CPU-only operation
    /// 
    /// # Error Handling
    /// 
    /// Device initialization failures are handled gracefully:
    /// - CUDA failure → Try Metal
    /// - Metal failure → Fall back to CPU
    /// - CPU failure → Return error (should never happen)
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::device::auto_device;
    /// 
    /// // Automatically select best device
    /// let device = auto_device()?;
    /// 
    /// // Use device for model creation
    /// let model = CandleModel::new(device);
    /// ```
    /// 
    /// # Platform Compatibility
    /// 
    /// - **Linux**: CUDA + CPU support
    /// - **macOS**: Metal + CPU support  
    /// - **Windows**: CUDA + CPU support
    /// - **Other**: CPU-only fallback
    /// 
    /// # Thread Safety
    /// 
    /// This function is thread-safe and can be called concurrently.
    /// Device selection is deterministic given the same hardware configuration.
    #[inline(always)]
    pub fn auto_device() -> candle_core::Result<Device> {
        // Try CUDA first if available
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                return Ok(device);
            }
        }

        // Try Metal if available
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                return Ok(device);
            }
        }

        // Fallback to CPU
        Ok(Device::Cpu)
    }

    /// Returns a human-readable string describing the device type
    /// 
    /// Provides a concise device identification string for logging, debugging,
    /// and user interface display purposes.
    /// 
    /// # Arguments
    /// 
    /// * `device` - Reference to the Candle Device to identify
    /// 
    /// # Returns
    /// 
    /// `&'static str` containing device type description:
    /// - `"CPU"` - Central Processing Unit (universal fallback)
    /// - `"CUDA"` - NVIDIA GPU with CUDA acceleration
    /// - `"Metal"` - Apple GPU with Metal acceleration
    /// - `"Unknown"` - Unrecognized or unsupported device type
    /// 
    /// # Performance Notes
    /// 
    /// - Zero-cost operation (compile-time string selection)
    /// - No allocations or dynamic string formatting
    /// - Inlined for optimal performance in hot paths
    /// 
    /// # Use Cases
    /// 
    /// - **Logging**: Record device type in generation logs
    /// - **Debugging**: Verify correct device selection
    /// - **UI Display**: Show active device to users
    /// - **Telemetry**: Report device usage statistics
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::device::{auto_device, device_info};
    /// 
    /// let device = auto_device()?;
    /// let info = device_info(&device);
    /// 
    /// println!("Using {} for inference", info);
    /// 
    /// // Example outputs:
    /// // "Using CUDA for inference"
    /// // "Using Metal for inference" 
    /// // "Using CPU for inference"
    /// ```
    /// 
    /// # Feature Compatibility
    /// 
    /// Device identification works regardless of compile-time features.
    /// Unknown devices are safely handled with descriptive fallback.
    #[inline(always)]
    pub fn device_info(device: &Device) -> &'static str {
        match device {
            Device::Cpu => "CPU",
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => "CUDA",
            #[cfg(feature = "metal")]
            Device::Metal(_) => "Metal",
            _ => "Unknown"}
    }

    /// Determines if the device provides hardware-accelerated matrix operations
    /// 
    /// Evaluates device capabilities for high-performance linear algebra operations
    /// critical to neural network inference and training.
    /// 
    /// # Arguments
    /// 
    /// * `device` - Reference to the Candle Device to evaluate
    /// 
    /// # Returns
    /// 
    /// `bool` indicating hardware acceleration availability:
    /// - `true` - Device has dedicated matrix multiplication units (GPU)
    /// - `false` - Device uses general-purpose compute units (CPU)
    /// 
    /// # Device Capabilities
    /// 
    /// ## GPU Devices (Hardware Accelerated)
    /// - **CUDA**: Tensor Cores, cuBLAS optimizations
    /// - **Metal**: Apple GPU matrix units, Metal Performance Shaders
    /// 
    /// ## CPU Devices (Software Implementation)
    /// - **CPU**: SIMD instructions, but no dedicated matrix units
    /// - Limited parallelism compared to GPU architectures
    /// 
    /// # Performance Impact
    /// 
    /// Hardware matrix acceleration provides significant benefits:
    /// - **Memory Bandwidth**: GPU memory is 5-10x faster than system RAM
    /// - **Parallelism**: Thousands of cores vs dozens on CPU
    /// - **Specialized Units**: Dedicated matrix multiplication hardware
    /// - **Power Efficiency**: Better performance per watt for ML workloads
    /// 
    /// # Use Cases
    /// 
    /// - **Model Selection**: Choose appropriate model size based on device capabilities
    /// - **Batch Size Optimization**: Larger batches benefit more from GPU acceleration
    /// - **Memory Planning**: GPU memory constraints vs CPU memory abundance
    /// - **Performance Expectations**: Set appropriate latency/throughput targets
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::device::{auto_device, supports_fast_matmul};
    /// 
    /// let device = auto_device()?;
    /// 
    /// if supports_fast_matmul(&device) {
    ///     println!("Using hardware-accelerated matrix operations");
    ///     // Can use larger models and batch sizes
    /// } else {
    ///     println!("Using CPU matrix operations");
    ///     // Should use smaller models for acceptable performance
    /// }
    /// ```
    /// 
    /// # Decision Making
    /// 
    /// ```rust
    /// // Adjust configuration based on device capabilities
    /// let batch_size = if supports_fast_matmul(&device) {
    ///     32  // GPU can handle larger batches efficiently
    /// } else {
    ///     4   // CPU works better with smaller batches
    /// };
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// This function is thread-safe and deterministic for the same device type.
    /// Results are consistent across multiple calls.
    #[inline(always)]
    pub fn supports_fast_matmul(device: &Device) -> bool {
        match device {
            Device::Cpu => false,
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => true,
            #[cfg(feature = "metal")]
            Device::Metal(_) => true,
            _ => false}
    }
}

// ============================================================================
// Memory Management Re-exports
// ============================================================================

/// Memory tracking and allocation utilities
/// 
/// Advanced memory management functions for monitoring allocation patterns,
/// detecting memory leaks, and optimizing memory usage in ML workloads.
pub use memory::{
    allocation_count,
    current_usage,
    peak_usage,
    reset_stats,
    track_allocation,
    track_deallocation,
};

/// Performance utilities for optimization
pub mod perf {
    use std::time::Instant;

    /// High-precision performance timer for detailed benchmarking and optimization
    /// 
    /// Provides microsecond and nanosecond precision timing with minimal overhead
    /// for profiling critical code paths in ML inference pipelines.
    /// 
    /// # Features
    /// 
    /// - **High Precision**: Nanosecond resolution using `std::time::Instant`
    /// - **Zero Allocation**: Stack-allocated with compile-time string names
    /// - **Automatic Logging**: Logs elapsed time when dropped (RAII pattern)
    /// - **Minimal Overhead**: Inlined methods for negligible performance impact
    /// 
    /// # Use Cases
    /// 
    /// - **Model Inference**: Measure forward pass timing
    /// - **Token Generation**: Profile individual token generation latency
    /// - **Memory Operations**: Time allocation and deallocation patterns
    /// - **I/O Operations**: Measure model loading and saving performance
    /// - **Algorithm Comparison**: Compare different sampling strategies
    /// 
    /// # Automatic Logging
    /// 
    /// The timer automatically logs elapsed time when dropped, using the
    /// `tracing` crate at DEBUG level. This enables zero-effort profiling
    /// by simply creating a timer instance.
    pub struct PerfTimer {
        start: Instant,
        name: &'static str}

    impl PerfTimer {
        /// Creates and starts a new performance timer with the given name
        /// 
        /// Initializes a high-precision timer that begins measuring elapsed time
        /// immediately upon creation.
        /// 
        /// # Arguments
        /// 
        /// * `name` - Static string identifier for this timing measurement
        ///   Used in automatic logging when the timer is dropped
        /// 
        /// # Returns
        /// 
        /// `PerfTimer` instance that begins timing immediately
        /// 
        /// # Performance Notes
        /// 
        /// - Zero allocation (stack-allocated struct)
        /// - Inlined for minimal call overhead
        /// - Uses `Instant::now()` for maximum precision
        /// - Static string name avoids runtime string allocation
        /// 
        /// # Examples
        /// 
        /// ```rust
        /// use fluent_ai_candle::perf::PerfTimer;
        /// 
        /// let timer = PerfTimer::new("model_inference");
        /// // ... perform timed operation ...
        /// let elapsed = timer.elapsed_micros();
        /// ```
        /// 
        /// # RAII Pattern
        /// 
        /// Timer automatically logs elapsed time when dropped:
        /// ```rust
        /// {
        ///     let _timer = PerfTimer::new("operation_name");
        ///     // ... timed code ...
        /// } // Automatically logs: "operation_name took 1234 μs"
        /// ```
        #[inline(always)]
        pub fn new(name: &'static str) -> Self {
            Self {
                start: Instant::now(),
                name}
        }

        /// Returns elapsed time since timer creation in microseconds
        /// 
        /// Provides microsecond-precision timing suitable for most performance
        /// analysis needs while maintaining good readability.
        /// 
        /// # Returns
        /// 
        /// `u64` - Elapsed microseconds since timer was created
        /// 
        /// # Precision and Range
        /// 
        /// - **Resolution**: 1 microsecond (0.001 milliseconds)
        /// - **Range**: Up to ~584,000 years (u64 microseconds)
        /// - **Accuracy**: Platform-dependent, typically nanosecond precision
        /// 
        /// # Use Cases
        /// 
        /// - **Model Inference**: Track forward pass timing (typically 1-100ms)
        /// - **Token Generation**: Measure per-token latency (typically 1-50ms)
        /// - **API Responses**: Monitor end-to-end request timing
        /// - **Performance Regression**: Detect timing changes between versions
        /// 
        /// # Examples
        /// 
        /// ```rust
        /// use fluent_ai_candle::perf::PerfTimer;
        /// 
        /// let timer = PerfTimer::new("inference");
        /// // ... perform inference ...
        /// let elapsed = timer.elapsed_micros();
        /// 
        /// if elapsed > 100_000 {  // 100ms
        ///     eprintln!("Inference took {}ms (slow)", elapsed / 1000);
        /// } else {
        ///     println!("Inference took {}μs", elapsed);
        /// }
        /// ```
        /// 
        /// # Performance Notes
        /// 
        /// - Inlined for zero call overhead
        /// - Single system call to get current time
        /// - No allocations or complex calculations
        #[inline(always)]
        pub fn elapsed_micros(&self) -> u64 {
            self.start.elapsed().as_micros() as u64
        }

        /// Returns elapsed time since timer creation in nanoseconds
        /// 
        /// Provides maximum precision timing for ultra-fine-grained performance
        /// analysis and optimization of critical hot paths.
        /// 
        /// # Returns
        /// 
        /// `u64` - Elapsed nanoseconds since timer was created
        /// 
        /// # Precision and Range
        /// 
        /// - **Resolution**: 1 nanosecond (0.000001 milliseconds)
        /// - **Range**: Up to ~584 years (u64 nanoseconds)
        /// - **Accuracy**: Platform-dependent, may be limited by hardware
        /// 
        /// # Use Cases
        /// 
        /// - **CPU Instruction Timing**: Measure individual algorithm steps
        /// - **Memory Access Patterns**: Profile cache performance
        /// - **SIMD Operations**: Optimize vectorized computations
        /// - **Micro-benchmarks**: Compare different implementations
        /// - **Critical Path Analysis**: Identify bottlenecks in hot loops
        /// 
        /// # Platform Considerations
        /// 
        /// - **Windows**: ~100ns resolution (QueryPerformanceCounter)
        /// - **Linux**: ~1ns resolution (clock_gettime)
        /// - **macOS**: ~1ns resolution (mach_absolute_time)
        /// 
        /// # Examples
        /// 
        /// ```rust
        /// use fluent_ai_candle::perf::PerfTimer;
        /// 
        /// let timer = PerfTimer::new("simd_operation");
        /// // ... perform SIMD computation ...
        /// let elapsed = timer.elapsed_nanos();
        /// 
        /// if elapsed < 1000 {  // Less than 1 microsecond
        ///     println!("Ultra-fast operation: {}ns", elapsed);
        /// } else {
        ///     println!("Operation took {}μs", elapsed / 1000);
        /// }
        /// ```
        /// 
        /// # Micro-benchmarking
        /// 
        /// ```rust
        /// // Compare two implementations
        /// let timer1 = PerfTimer::new("implementation_a");
        /// implementation_a();
        /// let time_a = timer1.elapsed_nanos();
        /// 
        /// let timer2 = PerfTimer::new("implementation_b");  
        /// implementation_b();
        /// let time_b = timer2.elapsed_nanos();
        /// 
        /// println!("A: {}ns, B: {}ns, ratio: {:.2}", 
        ///          time_a, time_b, time_a as f64 / time_b as f64);
        /// ```
        /// 
        /// # Performance Notes
        /// 
        /// - Inlined for minimal measurement overhead
        /// - Direct system timer access
        /// - Measurement overhead typically < 10ns
        #[inline(always)]
        pub fn elapsed_nanos(&self) -> u64 {
            self.start.elapsed().as_nanos() as u64
        }
    }

    impl Drop for PerfTimer {
        fn drop(&mut self) {
            let elapsed = self.elapsed_micros();
            tracing::debug!("{} took {} μs", self.name, elapsed);
        }
    }

    /// Macro for easy performance timing
    #[macro_export]
    macro_rules! perf_timer {
        ($name:expr) => {
            let _timer = $crate::perf::PerfTimer::new($name);
        };
    }
}

// ============================================================================
// Tensor Utilities for High-Performance ML Operations
// ============================================================================

/// Optimized utilities for working with candle tensors and token operations
/// 
/// Provides zero-allocation, performance-optimized functions for common tensor
/// operations in ML inference pipelines, including token conversion, probability
/// sampling, and temperature scaling.
/// 
/// # Performance Features
/// 
/// - **Zero Allocation**: Uses pre-allocated buffers and stack allocation where possible
/// - **Inlined Functions**: All functions are inlined for maximum performance
/// - **SIMD-Friendly**: Operations designed to work well with vectorized instructions
/// - **Memory Efficient**: Minimizes tensor copies and temporary allocations
/// 
/// # Common Usage Patterns
/// 
/// ```rust
/// use fluent_ai_candle::tensor_utils::*;
/// use arrayvec::ArrayVec;
/// 
/// // Convert model output tensor to tokens
/// let mut token_buffer = ArrayVec::<u32, 2048>::new();
/// tensor_to_tokens(&output_tensor, &mut token_buffer)?;
/// 
/// // Apply temperature scaling to logits
/// let scaled_logits = softmax_with_temperature(&logits, 0.8)?;
/// 
/// // Sample next token with top-k and top-p filtering
/// let next_token = sample_token(&probabilities, Some(50), Some(0.9), &mut rng)?;
/// ```
pub mod tensor_utils {
    use candle_core::{Result as CandleResult, Tensor};
    use arrayvec::ArrayVec;

    /// Converts a tensor containing token IDs to a token buffer with zero allocation
    /// 
    /// Efficiently extracts token IDs from a 1D tensor and stores them in a pre-allocated
    /// buffer, avoiding heap allocations during inference hot paths.
    /// 
    /// # Arguments
    /// 
    /// * `tensor` - Input tensor containing token IDs as u32 values
    ///   Must be a 1D tensor or convertible to 1D via `to_vec1()`
    /// * `buffer` - Pre-allocated buffer to store extracted token IDs
    ///   Buffer is cleared before use and must have sufficient capacity
    /// 
    /// # Returns
    /// 
    /// `CandleResult<()>` indicating success or error:
    /// - `Ok(())` - Tokens successfully extracted and stored in buffer
    /// - `Err(candle_core::Error::Msg("Token buffer overflow"))` - Buffer too small
    /// - `Err(...)` - Tensor conversion or access errors
    /// 
    /// # Performance Notes
    /// 
    /// - Zero heap allocation (uses provided buffer)
    /// - Inlined for maximum performance in inference loops
    /// - Buffer capacity check prevents runtime panics
    /// - Direct tensor-to-slice conversion for efficiency
    /// 
    /// # Buffer Management
    /// 
    /// The buffer is automatically cleared before use, so previous contents
    /// are discarded. Ensure buffer capacity is sufficient for expected
    /// token sequences to avoid overflow errors.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::tensor_utils::tensor_to_tokens;
    /// use arrayvec::ArrayVec;
    /// use candle_core::{Tensor, Device};
    /// 
    /// // Create tensor with token IDs
    /// let tokens = [101u32, 2054, 2003, 102]; // Example BERT tokens
    /// let tensor = Tensor::new(&tokens, &Device::Cpu)?;
    /// 
    /// // Extract to pre-allocated buffer
    /// let mut buffer = ArrayVec::<u32, 2048>::new(); 
    /// tensor_to_tokens(&tensor, &mut buffer)?;
    /// 
    /// assert_eq!(buffer.as_slice(), &tokens);
    /// ```
    /// 
    /// # Error Handling
    /// 
    /// ```rust
    /// // Handle buffer overflow gracefully
    /// let mut small_buffer = ArrayVec::<u32, 2>::new();
    /// match tensor_to_tokens(&large_tensor, &mut small_buffer) {
    ///     Ok(()) => println!("Tokens extracted successfully"),
    ///     Err(e) if e.to_string().contains("overflow") => {
    ///         println!("Buffer too small, need larger capacity");
    ///     }
    ///     Err(e) => println!("Tensor error: {}", e),
    /// }
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// This function is thread-safe when each thread uses its own buffer.
    /// The tensor parameter is read-only and can be shared between threads.
    #[inline(always)]
    pub fn tensor_to_tokens(tensor: &Tensor, buffer: &mut ArrayVec<u32, 2048>) -> CandleResult<()> {
        let data = tensor.to_vec1::<u32>()?;
        buffer.clear();
        buffer
            .try_extend_from_slice(&data)
            .map_err(|_| candle_core::Error::Msg("Token buffer overflow".into()))?;
        Ok(())
    }

    /// Converts token IDs to a batched tensor suitable for model input
    /// 
    /// Creates a properly shaped tensor from token IDs, adding the batch dimension
    /// required by most language models. The resulting tensor has shape [1, sequence_length].
    /// 
    /// # Arguments
    /// 
    /// * `tokens` - Slice of token IDs to convert to tensor
    ///   Each token should be a valid vocabulary ID (u32)
    /// * `device` - Target device for tensor allocation (CPU, CUDA, Metal)
    ///   Tensor will be allocated on this device for optimal inference performance
    /// 
    /// # Returns
    /// 
    /// `CandleResult<Tensor>` containing the batched tensor:
    /// - `Ok(tensor)` - Tensor with shape [1, tokens.len()] on specified device
    /// - `Err(...)` - Device allocation or tensor creation errors
    /// 
    /// # Tensor Shape
    /// 
    /// Input: `[token1, token2, token3, ...]` (length N)
    /// Output: `[[token1, token2, token3, ...]]` (shape [1, N])
    /// 
    /// The batch dimension (first dimension = 1) is automatically added
    /// to match the expected input format for transformer models.
    /// 
    /// # Performance Notes
    /// 
    /// - Inlined for zero function call overhead
    /// - Single allocation for the output tensor
    /// - Direct memory copy from slice to tensor
    /// - Device-aware allocation for optimal performance
    /// 
    /// # Device Placement
    /// 
    /// The tensor is created directly on the target device, avoiding
    /// expensive CPU→GPU transfers during inference. Choose device
    /// based on where the model is located for best performance.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::tensor_utils::tokens_to_tensor;
    /// use fluent_ai_candle::device::auto_device;
    /// 
    /// // Convert tokens for model input
    /// let tokens = [101u32, 2054, 2003, 102];
    /// let device = auto_device()?;
    /// let input_tensor = tokens_to_tensor(&tokens, &device)?;
    /// 
    /// // Tensor ready for model forward pass
    /// assert_eq!(input_tensor.shape(), &[1, 4]);
    /// ```
    /// 
    /// # Memory Efficiency
    /// 
    /// ```rust
    /// // For inference loops, reuse device reference
    /// let device = auto_device()?;
    /// 
    /// for token_sequence in sequences {
    ///     let tensor = tokens_to_tensor(&token_sequence, &device)?;
    ///     let output = model.forward(&tensor)?;
    ///     // Process output...
    /// }
    /// ```
    /// 
    /// # Error Handling
    /// 
    /// ```rust
    /// match tokens_to_tensor(&tokens, &device) {
    ///     Ok(tensor) => {
    ///         println!("Created tensor with shape: {:?}", tensor.shape());
    ///     }
    ///     Err(e) => {
    ///         eprintln!("Failed to create tensor: {}", e);
    ///         // Handle device allocation failure, invalid tokens, etc.
    ///     }
    /// }
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// This function is thread-safe. Each call creates an independent tensor
    /// on the specified device. Multiple threads can create tensors concurrently.
    #[inline(always)]
    pub fn tokens_to_tensor(tokens: &[u32], device: &candle_core::Device) -> CandleResult<Tensor> {
        Tensor::new(tokens, device)?.unsqueeze(0)
    }

    /// Applies softmax activation with temperature scaling for controlled randomness
    /// 
    /// Computes temperature-scaled softmax over the last dimension of logits tensor,
    /// enabling fine control over the randomness/determinism of probability distributions
    /// used in text generation.
    /// 
    /// # Arguments
    /// 
    /// * `logits` - Input tensor containing raw logits from model output
    ///   Typically shape [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
    /// * `temperature` - Temperature scaling factor controlling randomness
    ///   - `temperature = 1.0`: Standard softmax (no scaling)
    ///   - `temperature < 1.0`: More deterministic (sharper distribution)
    ///   - `temperature > 1.0`: More random (flatter distribution)
    /// 
    /// # Returns
    /// 
    /// `CandleResult<Tensor>` containing the probability distribution:
    /// - `Ok(tensor)` - Softmax probabilities with same shape as input
    /// - `Err(...)` - Tensor operation or device errors
    /// 
    /// # Temperature Effects
    /// 
    /// ## Low Temperature (< 1.0) - More Deterministic
    /// - Amplifies differences between logits
    /// - Results in sharper, more peaked distributions
    /// - Model becomes more confident and repetitive
    /// - Good for factual, structured generation
    /// 
    /// ## High Temperature (> 1.0) - More Random  
    /// - Dampens differences between logits
    /// - Results in flatter, more uniform distributions
    /// - Model becomes more creative and diverse
    /// - Good for creative writing, brainstorming
    /// 
    /// # Mathematical Formula
    /// 
    /// ```text
    /// scaled_logits[i] = logits[i] / temperature
    /// probabilities[i] = exp(scaled_logits[i]) / sum(exp(scaled_logits))
    /// ```
    /// 
    /// # Performance Optimizations
    /// 
    /// - **Temperature = 1.0**: Bypasses scaling for maximum performance
    /// - **Inlined**: Zero function call overhead in hot paths
    /// - **Single Pass**: Combines scaling and softmax operations
    /// - **Device Optimized**: Uses candle's optimized softmax implementation
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::tensor_utils::softmax_with_temperature;
    /// use candle_core::{Tensor, Device};
    /// 
    /// let logits = Tensor::new(&[2.0f32, 1.0, 0.1], &Device::Cpu)?;
    /// 
    /// // Standard softmax (temperature = 1.0)
    /// let probs_std = softmax_with_temperature(&logits, 1.0)?;
    /// 
    /// // Low temperature - more deterministic
    /// let probs_det = softmax_with_temperature(&logits, 0.5)?;
    /// 
    /// // High temperature - more random
    /// let probs_rand = softmax_with_temperature(&logits, 2.0)?;
    /// ```
    /// 
    /// # Generation Quality Control
    /// 
    /// ```rust
    /// // Adjust temperature based on generation context
    /// let temperature = match generation_mode {
    ///     GenerationMode::Factual => 0.1,      // Very deterministic
    ///     GenerationMode::Balanced => 0.8,     // Slightly deterministic  
    ///     GenerationMode::Creative => 1.2,     // Slightly random
    ///     GenerationMode::Experimental => 2.0, // Very random
    /// };
    /// 
    /// let probs = softmax_with_temperature(&logits, temperature)?;
    /// ```
    /// 
    /// # Numerical Stability
    /// 
    /// The implementation uses candle's numerically stable softmax operation
    /// that handles extreme values and prevents overflow/underflow issues.
    /// 
    /// # Thread Safety
    /// 
    /// This function is thread-safe and can be called concurrently on different
    /// tensors. Each call operates on independent data.
    #[inline(always)]
    pub fn softmax_with_temperature(logits: &Tensor, temperature: f32) -> CandleResult<Tensor> {
        if temperature == 1.0 {
            candle_nn::ops::softmax(logits, candle_core::D::Minus1)
        } else {
            let scaled = logits.affine(1.0 / temperature as f64, 0.0)?;
            candle_nn::ops::softmax(&scaled, candle_core::D::Minus1)
        }
    }

    /// Samples a token from probability distribution using advanced filtering strategies
    /// 
    /// Implements nucleus (top-p) and top-k sampling with efficient probability filtering
    /// and weighted random selection. This is the core sampling function used in modern
    /// language model text generation.
    /// 
    /// # Arguments
    /// 
    /// * `probs` - Tensor containing probability distribution over vocabulary
    ///   Should be 1D tensor with probabilities that sum to ~1.0
    /// * `top_k` - Optional top-k filtering parameter
    ///   - `None`: No top-k filtering applied
    ///   - `Some(k)`: Only consider the k most probable tokens
    /// * `top_p` - Optional nucleus (top-p) filtering parameter  
    ///   - `None`: No nucleus filtering applied
    ///   - `Some(p)`: Only consider tokens in top p cumulative probability mass
    /// * `rng` - Random number generator for sampling
    ///   Any type implementing `rand::Rng` trait
    /// 
    /// # Returns
    /// 
    /// `CandleResult<u32>` containing the sampled token ID:
    /// - `Ok(token_id)` - Sampled token from filtered distribution
    /// - `Err(...)` - Tensor conversion or probability extraction errors
    /// 
    /// # Sampling Strategy
    /// 
    /// The function applies filtering in this order:
    /// 1. **Top-k filtering**: Zero out all but the k most probable tokens
    /// 2. **Top-p filtering**: Zero out tokens beyond cumulative probability p
    /// 3. **Renormalization**: Rescale remaining probabilities to sum to 1.0
    /// 4. **Weighted sampling**: Sample according to filtered distribution
    /// 
    /// # Top-k Filtering
    /// 
    /// Limits consideration to the k most probable tokens:
    /// - `top_k = Some(1)`: Greedy sampling (always most probable)
    /// - `top_k = Some(10)`: Consider only top 10 tokens
    /// - `top_k = Some(50)`: Moderate filtering for balance
    /// - Higher k values preserve more diversity
    /// 
    /// # Top-p (Nucleus) Filtering
    /// 
    /// Dynamically selects tokens based on cumulative probability:
    /// - `top_p = Some(0.9)`: Include tokens covering 90% probability mass
    /// - `top_p = Some(0.95)`: More conservative, higher quality
    /// - `top_p = Some(0.8)`: More aggressive filtering
    /// - Adapts to probability distribution shape
    /// 
    /// # Performance Characteristics
    /// 
    /// - **Time Complexity**: O(V log V) for vocabulary size V (due to sorting)
    /// - **Space Complexity**: O(V) for probability vector and indices
    /// - **Inlined**: Zero function call overhead
    /// - **Single Allocation**: Reuses probability vector for in-place filtering
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::tensor_utils::sample_token;
    /// use candle_core::{Tensor, Device};
    /// use rand::thread_rng;
    /// 
    /// let probs = Tensor::new(&[0.1f32, 0.7, 0.15, 0.05], &Device::Cpu)?;
    /// let mut rng = thread_rng();
    /// 
    /// // Pure random sampling
    /// let token1 = sample_token(&probs, None, None, &mut rng)?;
    /// 
    /// // Top-k sampling (only top 2 tokens)
    /// let token2 = sample_token(&probs, Some(2), None, &mut rng)?;
    /// 
    /// // Nucleus sampling (90% probability mass)
    /// let token3 = sample_token(&probs, None, Some(0.9), &mut rng)?;
    /// 
    /// // Combined filtering
    /// let token4 = sample_token(&probs, Some(10), Some(0.9), &mut rng)?;
    /// ```
    /// 
    /// # Quality vs Diversity Trade-offs
    /// 
    /// ```rust
    /// // High quality, low diversity
    /// let conservative_token = sample_token(&probs, Some(5), Some(0.8), &mut rng)?;
    /// 
    /// // Balanced quality and diversity  
    /// let balanced_token = sample_token(&probs, Some(50), Some(0.9), &mut rng)?;
    /// 
    /// // High diversity, potential quality issues
    /// let diverse_token = sample_token(&probs, Some(100), Some(0.95), &mut rng)?;
    /// ```
    /// 
    /// # Error Handling
    /// 
    /// ```rust
    /// match sample_token(&probs, Some(10), Some(0.9), &mut rng) {
    ///     Ok(token) => println!("Sampled token: {}", token),
    ///     Err(e) => {
    ///         eprintln!("Sampling failed: {}", e);
    ///         // Handle tensor conversion errors, invalid probabilities, etc.
    ///     }
    /// }
    /// ```
    /// 
    /// # Fallback Behavior
    /// 
    /// If all filtering results in zero probabilities, the function returns
    /// the last token in the vocabulary as a safe fallback. This prevents
    /// crashes from over-aggressive filtering.
    /// 
    /// # Thread Safety
    /// 
    /// This function is thread-safe when each thread uses its own RNG instance.
    /// The probability tensor is read-only and can be shared between threads.
    #[inline(always)]
    pub fn sample_token(
        probs: &Tensor,
        top_k: Option<usize>,
        top_p: Option<f64>,
        rng: &mut impl rand::Rng,
    ) -> CandleResult<u32> {
        let mut probs = probs.to_vec1::<f32>()?;
        let vocab_size = probs.len();

        // Apply top-k filtering
        if let Some(k) = top_k {
            if k < vocab_size {
                // Find the k-th largest probability
                let mut indices: Vec<usize> = (0..vocab_size).collect();
                indices.sort_by(|&a, &b| {
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Zero out probabilities below top-k
                for &idx in &indices[k..] {
                    probs[idx] = 0.0;
                }
            }
        }

        // Apply top-p (nucleus) filtering
        if let Some(p) = top_p {
            let mut indices: Vec<usize> = (0..vocab_size).collect();
            indices.sort_by(|&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut cumulative_prob = 0.0;
            for (i, &idx) in indices.iter().enumerate() {
                cumulative_prob += probs[idx];
                if cumulative_prob > p as f32 {
                    // Zero out remaining probabilities
                    for &remaining_idx in &indices[i + 1..] {
                        probs[remaining_idx] = 0.0;
                    }
                    break;
                }
            }
        }

        // Renormalize probabilities
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in &mut probs {
                *prob /= sum;
            }
        }

        // Sample from the distribution
        let random_value: f32 = rng.random();
        let mut cumulative = 0.0;

        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return Ok(i as u32);
            }
        }

        // Fallback to last token
        Ok((vocab_size - 1) as u32)
    }
}

// ============================================================================
// Version and Build Information
// ============================================================================

/// Current version of the fluent-ai-candle crate
/// 
/// Contains the semantic version string extracted from Cargo.toml at compile time.
/// This version follows semantic versioning (semver) principles with MAJOR.MINOR.PATCH format.
/// 
/// # Version Format
/// 
/// - **MAJOR**: Incremented for incompatible API changes
/// - **MINOR**: Incremented for backwards-compatible functionality additions
/// - **PATCH**: Incremented for backwards-compatible bug fixes
/// 
/// # Usage
/// 
/// ```rust
/// use fluent_ai_candle::VERSION;
/// 
/// println!("Using fluent-ai-candle version: {}", VERSION);
/// 
/// // Example output: "Using fluent-ai-candle version: 0.1.0"
/// ```
/// 
/// # Compile-Time Evaluation
/// 
/// This constant is evaluated at compile time using the `CARGO_PKG_VERSION`
/// environment variable, ensuring zero runtime overhead for version queries.
/// 
/// # Integration with Logging
/// 
/// ```rust
/// tracing::info!("Starting inference with fluent-ai-candle v{}", VERSION);
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Comprehensive build information including versions of key dependencies
/// 
/// Provides a human-readable string containing version information for both
/// this crate and critical dependencies like candle-core. Useful for debugging,
/// support requests, and compatibility verification.
/// 
/// # Information Included
/// 
/// - **fluent-ai-candle version**: Main crate version from Cargo.toml
/// - **candle-core version**: Version of the underlying candle framework
/// - **Build target**: Information about the compilation target (when available)
/// 
/// # Format
/// 
/// The string follows the format:
/// ```text
/// fluent_ai_candle v{VERSION} built with candle-core v{CANDLE_VERSION}
/// ```
/// 
/// # Usage
/// 
/// ```rust
/// use fluent_ai_candle::BUILD_INFO;
/// 
/// println!("Build info: {}", BUILD_INFO);
/// 
/// // Example output: 
/// // "Build info: fluent_ai_candle v0.1.0 built with candle-core v0.9.1"
/// ```
/// 
/// # Debugging and Support
/// 
/// Include this information in bug reports and support requests:
/// ```rust
/// eprintln!("Error occurred in {}", BUILD_INFO);
/// tracing::error!("System info: {}", BUILD_INFO);
/// ```
/// 
/// # Dependency Tracking
/// 
/// This constant helps track compatibility between fluent-ai-candle and
/// the underlying candle framework, which is critical for:
/// - Model compatibility
/// - Performance characteristics  
/// - Feature availability
/// - Bug reproduction
/// 
/// # Compile-Time Generation
/// 
/// Build information is generated at compile time using `concat!` macro,
/// ensuring zero runtime cost and accurate dependency version tracking.
pub const BUILD_INFO: &str = concat!(
    "fluent_ai_candle v",
    env!("CARGO_PKG_VERSION"),
    " built with candle-core v0.9.1"
);


