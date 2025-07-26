//! Client builder for creating CandleCompletionClient instances
//!
//! This module provides a fluent builder API for configuring and creating clients.

use super::super::config::{CandleClientConfig, DeviceType, QuantizationType};
use super::super::completion::CandleCompletionClient;
use crate::error::CandleResult;
use crate::generator::GenerationConfig;

/// Fluent builder for configuring and creating CandleCompletionClient instances
/// 
/// Provides a type-safe, ergonomic interface for setting up ML inference clients
/// with comprehensive configuration options for models, devices, and performance tuning.
/// 
/// # Builder Pattern Benefits
/// 
/// - **Type Safety**: Compile-time validation of configuration combinations
/// - **Ergonomic API**: Fluent chaining for clean, readable setup code
/// - **Sensible Defaults**: Production-ready defaults with selective customization
/// - **Comprehensive Options**: Full control over model, device, and performance settings
/// 
/// # Configuration Categories
/// 
/// ## Model Configuration
/// - **Model Path**: Local file path or model identifier
/// - **Tokenizer Path**: Custom tokenizer configuration
/// - **Quantization**: Memory and speed optimization options
/// 
/// ## Device Configuration  
/// - **Device Type**: CPU, CUDA, Metal selection
/// - **Memory Management**: Cache settings and memory limits
/// - **Performance Tuning**: Batch sizes and threading options
/// 
/// ## Generation Configuration
/// - **Sampling Parameters**: Temperature, top-k, top-p settings
/// - **Output Control**: Token limits, stop sequences
/// - **Quality Settings**: Repetition penalties, bias application
/// 
/// # Performance Optimization
/// 
/// The builder enables several performance optimizations:
/// - **Device Auto-Selection**: Automatically choose optimal compute device
/// - **Quantization Support**: Reduce memory usage while maintaining quality
/// - **Concurrent Request Handling**: Configure parallel request processing
/// - **Memory Pool Management**: Optimize allocation patterns
/// 
/// # Examples
/// 
/// ## Basic Setup
/// ```rust
/// use fluent_ai_candle::{CandleClientBuilder, DeviceType};
/// 
/// let client = CandleClientBuilder::new()
///     .model_path("path/to/model.safetensors")
///     .device_type(DeviceType::Auto)
///     .build()?;
/// ```
/// 
/// ## Advanced Configuration
/// ```rust
/// use fluent_ai_candle::{
///     CandleClientBuilder, DeviceType, QuantizationType, GenerationConfig
/// };
/// 
/// let generation_config = GenerationConfig::default()
///     .temperature(0.8)
///     .max_tokens(2048);
/// 
/// let client = CandleClientBuilder::new()
///     .model_path("models/llama-7b.safetensors")
///     .tokenizer_path("models/tokenizer.json")
///     .device_type(DeviceType::Cuda)
///     .quantization(QuantizationType::Q4)
///     .generation_config(generation_config)
///     .max_concurrent_requests(4)
///     .build()?;
/// ```
pub struct CandleClientBuilder {
    config: CandleClientConfig}

impl CandleClientBuilder {
    /// Creates a new CandleClientBuilder with default configuration
    /// 
    /// Initializes a builder instance with production-ready defaults suitable
    /// for most ML inference scenarios, providing a clean starting point for customization.
    /// 
    /// # Default Configuration
    /// 
    /// - **Model Path**: Empty (must be set before building)
    /// - **Tokenizer Path**: Auto-detected from model directory
    /// - **Device Type**: Auto-selection (CUDA → Metal → CPU)
    /// - **Generation Config**: Conservative settings (temperature=1.0, max_tokens=2048)
    /// - **Quantization**: Disabled (full precision)
    /// - **Max Concurrent Requests**: 1 (single-threaded)
    /// 
    /// # Performance Notes
    /// 
    /// - Zero allocation constructor using stack-allocated defaults
    /// - Configuration struct is lightweight (primarily scalar values)
    /// - All settings can be overridden before building
    /// 
    /// # Builder Pattern
    /// 
    /// Returns `Self` for fluent chaining with other configuration methods.
    /// Configuration is mutable until `.build()` is called to create the client.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::CandleClientBuilder;
    /// 
    /// // Create builder with defaults
    /// let builder = CandleClientBuilder::new();
    /// 
    /// // Chain configuration methods
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/gpt-4.safetensors")
    ///     .device_type(DeviceType::Auto)
    ///     .build()?;
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// This method is thread-safe and can be called concurrently.
    /// Each builder instance maintains independent configuration state.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            config: CandleClientConfig::default()}
    }

    /// Configures the path to the model file or model identifier for loading
    /// 
    /// Sets the location of the machine learning model that will be used for text generation.
    /// This can be a local file path to a SafeTensors file, a directory containing model files,
    /// or a model identifier for automatic downloading from model hubs.
    /// 
    /// # Arguments
    /// 
    /// * `path` - Model location as any type convertible to String:
    ///   - **Local Path**: `/path/to/model.safetensors` or `/path/to/model/directory/`
    ///   - **Hub Identifier**: `"microsoft/DialoGPT-medium"` or `"facebook/opt-1.3b"`
    ///   - **Relative Path**: `"./models/my-model.safetensors"` (relative to current directory)
    /// 
    /// # Model File Formats
    /// 
    /// ## SafeTensors (Recommended)
    /// - **Format**: `.safetensors` files containing model weights
    /// - **Benefits**: Memory-safe, fast loading, cross-platform compatibility
    /// - **Example**: `"models/llama-7b.safetensors"`
    /// 
    /// ## Model Directories
    /// - **Structure**: Directory containing `model.safetensors`, `config.json`, etc.
    /// - **Benefits**: Self-contained model packages with metadata
    /// - **Example**: `"models/gpt-3.5-turbo/"` (containing multiple files)
    /// 
    /// ## Hub Integration
    /// - **Automatic Download**: Models downloaded and cached locally
    /// - **Version Control**: Specific commits or tags can be specified
    /// - **Example**: `"openai-gpt@v1.0"` (specific version)
    /// 
    /// # Path Resolution
    /// 
    /// The client resolves paths in this order:
    /// 1. **Absolute Path**: Used directly if file/directory exists
    /// 2. **Relative Path**: Resolved relative to current working directory
    /// 3. **Hub Identifier**: Downloaded to cache directory if not found locally
    /// 4. **Error**: Path validation fails if none of the above succeed
    /// 
    /// # Examples
    /// 
    /// ## Local Model File
    /// ```rust
    /// use fluent_ai_candle::CandleClientBuilder;
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path("/home/user/models/my-model.safetensors")
    ///     .build()?;
    /// ```
    /// 
    /// ## Model Directory
    /// ```rust
    /// // Directory containing model.safetensors, config.json, tokenizer files
    /// let client = CandleClientBuilder::new()
    ///     .model_path("/opt/models/llama-7b/")
    ///     .build()?;
    /// ```
    /// 
    /// ## Relative Path
    /// ```rust
    /// // Relative to current working directory
    /// let client = CandleClientBuilder::new()
    ///     .model_path("./assets/models/gpt-small.safetensors")
    ///     .build()?;
    /// ```
    /// 
    /// ## Hub Model Identifier
    /// ```rust
    /// // Automatically downloaded from HuggingFace Hub
    /// let client = CandleClientBuilder::new()
    ///     .model_path("microsoft/DialoGPT-medium")
    ///     .build()?;
    /// ```
    /// 
    /// ## Environment Variable Path
    /// ```rust
    /// use std::env;
    /// 
    /// let model_path = env::var("MODEL_PATH")
    ///     .unwrap_or_else(|_| "./default-model.safetensors".to_string());
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path(model_path)
    ///     .build()?;
    /// ```
    /// 
    /// ## Dynamic Path Construction
    /// ```rust
    /// let model_name = "gpt-3.5";
    /// let model_version = "v1.2";
    /// let path = format!("models/{}-{}.safetensors", model_name, model_version);
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path(path)
    ///     .build()?;
    /// ```
    /// 
    /// # Validation
    /// 
    /// Path validation occurs during `.build()` call, not during this method:
    /// - **File Existence**: Verified when client is built
    /// - **Format Validation**: Model format checked during loading
    /// - **Permissions**: Read access verified during initialization
    /// 
    /// # Performance Considerations
    /// 
    /// - **Local Files**: Fastest loading, no network overhead
    /// - **Hub Downloads**: Initial download time, then cached locally
    /// - **Network Models**: Higher latency, requires internet connectivity
    /// - **SSD vs HDD**: SSDs provide significantly faster model loading
    /// 
    /// # Error Handling
    /// 
    /// Common path-related errors during `.build()`:
    /// ```rust
    /// match CandleClientBuilder::new().model_path("invalid/path").build() {
    ///     Err(CandleError::ModelNotFound(path)) => {
    ///         eprintln!("Model file not found: {}", path);
    ///         // Try alternative path or download model
    ///     }
    ///     Err(CandleError::InvalidModelFormat(msg)) => {
    ///         eprintln!("Invalid model format: {}", msg);
    ///         // Check model file integrity
    ///     }
    ///     Err(CandleError::PermissionDenied(path)) => {
    ///         eprintln!("Cannot read model file: {}", path);
    ///         // Check file permissions
    ///     }
    ///     Ok(client) => {
    ///         println!("Model loaded successfully");
    ///     }
    /// }
    /// ```
    /// 
    /// # Security Considerations
    /// 
    /// - **Path Traversal**: Relative paths are resolved safely
    /// - **File Validation**: Model files validated before loading
    /// - **Hub Downloads**: Downloaded models cached in secure location
    /// - **Permissions**: Only read access required, no write permissions
    /// 
    /// # Builder Pattern
    /// 
    /// Returns `Self` for fluent method chaining. The path is stored in the
    /// configuration and used during client initialization.
    /// 
    /// # Thread Safety
    /// 
    /// This method moves `self`, ensuring thread-safe configuration.
    /// Each builder instance maintains independent model path settings.
    /// 
    /// # Architecture Compliance
    /// 
    /// - ✅ **Zero Allocation**: Path stored directly without additional allocation
    /// - ✅ **Type Safety**: Generic `Into<String>` ensures type flexibility
    /// - ✅ **Error Handling**: Path validation deferred to build-time for better UX
    /// - ✅ **Performance**: String conversion optimized for common path types
    #[inline(always)]
    pub fn model_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.model_path = path.into();
        self
    }

    /// Configures the path to a custom tokenizer for text processing and token conversion
    /// 
    /// Sets the location of the tokenizer that will be used to convert between text and token IDs.
    /// If not specified, the client will attempt to auto-detect the tokenizer from the model
    /// directory or use a compatible default tokenizer for the model architecture.
    /// 
    /// # Arguments
    /// 
    /// * `path` - Tokenizer location as any type convertible to String:
    ///   - **Tokenizer File**: `"tokenizer.json"` (HuggingFace format)
    ///   - **Tokenizer Directory**: `"tokenizers/custom-tokenizer/"` (containing tokenizer files)
    ///   - **Hub Identifier**: `"bert-base-uncased"` (uses model's tokenizer)
    ///   - **Relative Path**: `"./tokenizers/my-tokenizer.json"`
    /// 
    /// # Tokenizer Formats
    /// 
    /// ## HuggingFace JSON Format (Recommended)
    /// - **File**: `tokenizer.json` containing complete tokenizer configuration
    /// - **Benefits**: Full feature support, consistent behavior across platforms
    /// - **Example**: `"tokenizers/llama-tokenizer.json"`
    /// 
    /// ## Tokenizer Directory
    /// - **Structure**: Directory with `tokenizer.json`, `tokenizer_config.json`, vocab files
    /// - **Benefits**: Complete tokenizer package with all associated files
    /// - **Example**: `"tokenizers/gpt-4-tokenizer/"` (complete tokenizer package)
    /// 
    /// ## Legacy Formats
    /// - **SentencePiece**: `.model` files for SentencePiece tokenizers
    /// - **Vocabulary Files**: `.vocab` files with token mappings
    /// - **Example**: `"tokenizers/sentencepiece.model"`
    /// 
    /// # Auto-Detection Behavior
    /// 
    /// When tokenizer path is not specified:
    /// 1. **Model Directory Search**: Look for tokenizer files in model directory
    /// 2. **Architecture Default**: Use architecture-specific default tokenizer
    /// 3. **Fallback Tokenizer**: Basic tokenizer if no specific tokenizer found
    /// 
    /// # Examples
    /// 
    /// ## Custom Tokenizer File
    /// ```rust
    /// use fluent_ai_candle::CandleClientBuilder;
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/custom-model.safetensors")
    ///     .tokenizer_path("tokenizers/custom-tokenizer.json")
    ///     .build()?;
    /// ```
    /// 
    /// ## Tokenizer Directory
    /// ```rust
    /// // Complete tokenizer package
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/llama-7b.safetensors")
    ///     .tokenizer_path("tokenizers/llama-tokenizer/")
    ///     .build()?;
    /// ```
    /// 
    /// ## Relative Path Configuration
    /// ```rust
    /// // Tokenizer relative to current directory
    /// let client = CandleClientBuilder::new()
    ///     .model_path("./models/gpt-model.safetensors")
    ///     .tokenizer_path("./tokenizers/gpt-tokenizer.json")
    ///     .build()?;
    /// ```
    /// 
    /// ## Environment-Based Configuration
    /// ```rust
    /// use std::env;
    /// 
    /// let tokenizer_path = env::var("TOKENIZER_PATH")
    ///     .unwrap_or_else(|_| "default-tokenizer.json".to_string());
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path(env::var("MODEL_PATH").unwrap())
    ///     .tokenizer_path(tokenizer_path)
    ///     .build()?;
    /// ```
    /// 
    /// ## Hub Model with Custom Tokenizer
    /// ```rust
    /// // Use hub model with local custom tokenizer
    /// let client = CandleClientBuilder::new()
    ///     .model_path("microsoft/DialoGPT-medium")
    ///     .tokenizer_path("local-tokenizers/custom-dialog-tokenizer.json")
    ///     .build()?;
    /// ```
    /// 
    /// ## Auto-Detection (Default Behavior)
    /// ```rust
    /// // Tokenizer auto-detected from model directory
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/complete-model-package/")  // Contains tokenizer files
    ///     // .tokenizer_path() not called - auto-detection used
    ///     .build()?;
    /// ```
    /// 
    /// # Tokenizer Compatibility
    /// 
    /// ## Model-Tokenizer Matching
    /// Ensure tokenizer compatibility with the model:
    /// - **Vocabulary Size**: Tokenizer vocab size must match model's input embedding size
    /// - **Special Tokens**: Tokenizer special tokens must match model's training configuration
    /// - **Encoding**: Token ID mappings must be consistent with model training
    /// 
    /// ## Validation Example
    /// ```rust
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/llama-7b.safetensors")
    ///     .tokenizer_path("tokenizers/llama-tokenizer.json")
    ///     .build()?;
    /// 
    /// // Validation occurs during initialization
    /// client.initialize().await?;
    /// 
    /// // Test tokenizer compatibility
    /// let test_tokens = client.tokenizer().encode("Hello world", false)?;
    /// println!("Tokenizer working - {} tokens generated", test_tokens.len());
    /// ```
    /// 
    /// # Performance Considerations
    /// 
    /// - **Loading Time**: Tokenizer loading is typically fast (1-10ms)
    /// - **Memory Usage**: Tokenizers use 1-50MB depending on vocabulary size
    /// - **Encoding Speed**: Fast tokenization crucial for inference performance
    /// - **Caching**: Tokenizer configuration cached after first load
    /// 
    /// # Error Handling
    /// 
    /// Common tokenizer-related errors:
    /// ```rust
    /// match client.build() {
    ///     Err(CandleError::TokenizerNotFound(path)) => {
    ///         eprintln!("Tokenizer file not found: {}", path);
    ///         // Try auto-detection or different path
    ///     }
    ///     Err(CandleError::TokenizerFormat(msg)) => {
    ///         eprintln!("Invalid tokenizer format: {}", msg);
    ///         // Check tokenizer file format and integrity
    ///     }
    ///     Err(CandleError::TokenizerMismatch(msg)) => {
    ///         eprintln!("Tokenizer-model mismatch: {}", msg);
    ///         // Ensure tokenizer matches model architecture
    ///     }
    ///     Ok(client) => {
    ///         println!("Client with custom tokenizer ready");
    ///     }
    /// }
    /// ```
    /// 
    /// # Special Token Configuration
    /// 
    /// Custom tokenizers should define standard special tokens:
    /// ```json
    /// {
    ///   "added_tokens": [
    ///     {"id": 0, "content": "<pad>", "special": true},
    ///     {"id": 1, "content": "<unk>", "special": true},
    ///     {"id": 2, "content": "<s>", "special": true},
    ///     {"id": 3, "content": "</s>", "special": true}
    ///   ]
    /// }
    /// ```
    /// 
    /// # Debugging Tokenizer Issues
    /// 
    /// ```rust
    /// // Enable tokenizer debugging
    /// let client = CandleClientBuilder::new()
    ///     .model_path(model_path)
    ///     .tokenizer_path(tokenizer_path)
    ///     .build()?;
    /// 
    /// client.initialize().await?;
    /// 
    /// // Test basic tokenization
    /// let test_text = "Hello, world!";
    /// let tokens = client.tokenizer().encode(test_text, false)?;
    /// let decoded = client.tokenizer().decode(&tokens, false)?;
    /// 
    /// println!("Original: {}", test_text);
    /// println!("Tokens: {:?}", tokens);
    /// println!("Decoded: {}", decoded);
    /// 
    /// // Should match original text
    /// assert_eq!(test_text, decoded);
    /// ```
    /// 
    /// # Builder Pattern
    /// 
    /// Returns `Self` for fluent method chaining. The tokenizer path is stored
    /// as an `Option<String>` and used during client initialization.
    /// 
    /// # Thread Safety
    /// 
    /// This method moves `self`, ensuring thread-safe configuration.
    /// Each builder instance maintains independent tokenizer settings.
    /// 
    /// # Architecture Compliance
    /// 
    /// - ✅ **Optional Configuration**: Tokenizer path is optional with auto-detection fallback
    /// - ✅ **Type Safety**: Generic `Into<String>` provides flexible input types
    /// - ✅ **Error Handling**: Validation deferred to build-time for better error context
    /// - ✅ **Performance**: Efficient path storage with zero unnecessary allocation
    #[inline(always)]
    pub fn tokenizer_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.tokenizer_path = Some(path.into());
        self
    }

    /// Configures the compute device type for model inference and tensor operations
    /// 
    /// Specifies which type of hardware acceleration to use for running the machine learning model.
    /// This choice significantly impacts performance, memory usage, and compatibility depending
    /// on the target hardware and model size.
    /// 
    /// # Arguments
    /// 
    /// * `device_type` - The compute device type to use:
    ///   - `DeviceType::Auto` - Automatic selection (CUDA → Metal → CPU)
    ///   - `DeviceType::Cpu` - CPU execution with optimized BLAS/LAPACK
    ///   - `DeviceType::Cuda` - NVIDIA GPU with CUDA acceleration
    ///   - `DeviceType::Metal` - Apple Silicon GPU with Metal Performance Shaders
    /// 
    /// # Device Types
    /// 
    /// ## Auto Selection (Recommended)
    /// - **Priority Order**: CUDA → Metal → CPU (based on availability)
    /// - **Benefits**: Optimal performance without manual configuration
    /// - **Fallback**: Gracefully falls back if preferred device unavailable
    /// - **Use Case**: Production deployments and general usage
    /// 
    /// ## CPU Execution
    /// - **Compatibility**: Works on all systems without special hardware
    /// - **Memory**: Can handle larger models with system RAM
    /// - **Threading**: Utilizes multiple CPU cores for parallel processing
    /// - **Use Case**: Development, testing, or when GPU unavailable
    /// 
    /// ## CUDA GPU Acceleration
    /// - **Performance**: Significant speedup for medium to large models
    /// - **Memory**: Limited by GPU VRAM (typically 4-80GB)
    /// - **Precision**: Supports FP16, BF16, and INT8 for memory efficiency
    /// - **Use Case**: Production inference on NVIDIA hardware
    /// 
    /// ## Metal GPU Acceleration
    /// - **Apple Silicon**: Optimized for M1, M2, M3 chips and Apple GPUs
    /// - **Unified Memory**: Leverages shared CPU-GPU memory architecture
    /// - **Efficiency**: Excellent performance per watt on Apple hardware
    /// - **Use Case**: macOS deployment and Apple Silicon optimization
    /// 
    /// # Performance Characteristics
    /// 
    /// ## Inference Speed (Relative)
    /// - **CUDA (Modern GPU)**: 10-50x faster than CPU
    /// - **Metal (Apple Silicon)**: 5-20x faster than CPU
    /// - **CPU (Optimized)**: 1x baseline performance
    /// 
    /// ## Memory Requirements
    /// - **GPU**: Model size + KV cache (limited by VRAM)
    /// - **CPU**: Model size + system overhead (uses system RAM)
    /// - **Metal**: Unified memory shared between CPU and GPU
    /// 
    /// # Examples
    /// 
    /// ## Auto Device Selection (Recommended)
    /// ```rust
    /// use fluent_ai_candle::{CandleClientBuilder, DeviceType};
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/llama-7b.safetensors")
    ///     .device_type(DeviceType::Auto)  // Automatically select best device
    ///     .build()?;
    /// 
    /// // Check which device was selected
    /// match client.device() {
    ///     Device::Cuda(id) => println!("Using CUDA GPU {}", id),
    ///     Device::Metal(id) => println!("Using Metal GPU {}", id),
    ///     Device::Cpu => println!("Using CPU execution"),
    /// }
    /// ```
    /// 
    /// ## Force CPU Execution
    /// ```rust
    /// // Useful for debugging or when GPU memory is insufficient
    /// let cpu_client = CandleClientBuilder::new()
    ///     .model_path("models/large-model.safetensors")
    ///     .device_type(DeviceType::Cpu)  // Force CPU even if GPU available
    ///     .build()?;
    /// 
    /// println!("Running on CPU with {} threads", num_cpus::get());
    /// ```
    /// 
    /// ## Force CUDA Execution
    /// ```rust
    /// // Ensure CUDA is used, fail if unavailable
    /// let gpu_client = CandleClientBuilder::new()
    ///     .model_path("models/gpu-optimized-model.safetensors")
    ///     .device_type(DeviceType::Cuda)  // Require CUDA, fail if unavailable
    ///     .build()?;
    /// 
    /// // Will error if CUDA not available or compiled without CUDA support
    /// ```
    /// 
    /// ## Apple Silicon Optimization
    /// ```rust
    /// // Optimize for Apple Silicon chips
    /// let metal_client = CandleClientBuilder::new()
    ///     .model_path("models/metal-optimized.safetensors")
    ///     .device_type(DeviceType::Metal)  // Use Metal Performance Shaders
    ///     .build()?;
    /// 
    /// // Leverages unified memory and Apple Neural Engine
    /// ```
    /// 
    /// ## Environment-Based Selection
    /// ```rust
    /// use std::env;
    /// 
    /// let device_type = match env::var("INFERENCE_DEVICE").as_deref() {
    ///     Ok("cpu") => DeviceType::Cpu,
    ///     Ok("cuda") => DeviceType::Cuda,
    ///     Ok("metal") => DeviceType::Metal,
    ///     _ => DeviceType::Auto,  // Default to auto-selection
    /// };
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path(model_path)
    ///     .device_type(device_type)
    ///     .build()?;
    /// ```
    /// 
    /// ## Performance Testing Across Devices
    /// ```rust
    /// use std::time::Instant;
    /// 
    /// // Test performance on different devices
    /// let devices = vec![
    ///     DeviceType::Cpu,
    ///     DeviceType::Cuda,
    ///     DeviceType::Metal,
    /// ];
    /// 
    /// for device_type in devices {
    ///     match CandleClientBuilder::new()
    ///         .model_path(model_path)
    ///         .device_type(device_type)
    ///         .build() 
    ///     {
    ///         Ok(client) => {
    ///             let start = Instant::now();
    ///             let response = client.complete(test_request.clone()).collect().await?;
    ///             let duration = start.elapsed();
    ///             
    ///             println!("{:?}: {:?} ({} tokens/sec)", 
    ///                      device_type, 
    ///                      duration,
    ///                      response.usage.total_tokens as f64 / duration.as_secs_f64());
    ///         }
    ///         Err(e) => {
    ///             println!("{:?}: Not available ({})", device_type, e);
    ///         }
    ///     }
    /// }
    /// ```
    /// 
    /// # Error Handling
    /// 
    /// Device-specific errors during `.build()`:
    /// ```rust
    /// match CandleClientBuilder::new()
    ///     .model_path(model_path)
    ///     .device_type(DeviceType::Cuda)
    ///     .build() 
    /// {
    ///     Err(CandleError::Msg(msg)) if msg.contains("CUDA") => {
    ///         eprintln!("CUDA not available: {}", msg);
    ///         
    ///         // Fallback to CPU
    ///         let cpu_client = CandleClientBuilder::new()
    ///             .model_path(model_path)
    ///             .device_type(DeviceType::Cpu)
    ///             .build()?;
    ///     }
    ///     Err(CandleError::DeviceAllocation(msg)) => {
    ///         eprintln!("Device memory insufficient: {}", msg);
    ///         // Try smaller model or quantization
    ///     }
    ///     Ok(client) => {
    ///         println!("Client ready with requested device");
    ///     }
    /// }
    /// ```
    /// 
    /// # Hardware Requirements
    /// 
    /// ## CUDA Requirements
    /// - **Driver**: NVIDIA driver 450+ with CUDA 11.0+
    /// - **Compute Capability**: 6.0+ (Pascal architecture or newer)
    /// - **Memory**: Sufficient VRAM for model size + overhead
    /// - **Compilation**: Built with `cuda` feature enabled
    /// 
    /// ## Metal Requirements
    /// - **Hardware**: Apple Silicon (M1/M2/M3) or Metal-compatible AMD GPU
    /// - **OS**: macOS 10.15+ with Metal Performance Shaders support
    /// - **Memory**: Unified memory architecture on Apple Silicon
    /// - **Compilation**: Built with `metal` feature enabled
    /// 
    /// ## CPU Requirements
    /// - **Architecture**: x86_64 or ARM64 with optimized BLAS library
    /// - **Threads**: Multiple cores recommended for parallel processing
    /// - **Memory**: Sufficient RAM for model size (typically 2-4x model size)
    /// - **BLAS**: OpenBLAS, Intel MKL, or Accelerate framework for optimization
    /// 
    /// # Optimization Guidelines
    /// 
    /// ## Small Models (< 1B parameters)
    /// - **Auto or CPU**: Minimal benefit from GPU acceleration
    /// - **Latency**: CPU often faster due to GPU setup overhead
    /// - **Memory**: Fits comfortably in system RAM
    /// 
    /// ## Medium Models (1B - 7B parameters)
    /// - **GPU Recommended**: Significant acceleration on modern GPUs
    /// - **Memory**: May require quantization on consumer GPUs
    /// - **Throughput**: GPU excels at batch processing
    /// 
    /// ## Large Models (7B+ parameters)
    /// - **High-end GPU**: Requires substantial VRAM (16GB+)
    /// - **CPU Alternative**: May be necessary for very large models
    /// - **Quantization**: Often required to fit in GPU memory
    /// 
    /// # Builder Pattern
    /// 
    /// Returns `Self` for fluent method chaining. The device type is stored
    /// in the configuration and used during client creation to select hardware.
    /// 
    /// # Thread Safety
    /// 
    /// This method moves `self`, ensuring thread-safe configuration.
    /// Device selection is deterministic based on the specified type.
    /// 
    /// # Architecture Compliance
    /// 
    /// - ✅ **Performance**: Enables optimal hardware utilization
    /// - ✅ **Flexibility**: Supports multiple hardware acceleration options
    /// - ✅ **Fallback**: Auto mode provides graceful degradation
    /// - ✅ **Error Handling**: Clear error messages for unsupported devices
    #[inline(always)]
    pub fn device_type(mut self, device_type: DeviceType) -> Self {
        self.config.device_type = device_type;
        self
    }

    /// Configures text generation parameters for controlling output quality and characteristics
    /// 
    /// Sets the generation configuration that controls how the model produces text responses,
    /// including sampling strategies, output constraints, and quality parameters. This configuration
    /// significantly impacts the creativity, coherence, and style of generated content.
    /// 
    /// # Arguments
    /// 
    /// * `config` - GenerationConfig containing all generation parameters:
    ///   - **Sampling**: Temperature, top-k, top-p, nucleus sampling
    ///   - **Constraints**: Max tokens, stop sequences, length penalties
    ///   - **Quality**: Repetition penalties, frequency penalties, presence penalties
    ///   - **Control**: Seed for reproducibility, bias application
    /// 
    /// # Generation Parameters
    /// 
    /// ## Sampling Controls
    /// - **Temperature**: Controls randomness (0.0 = deterministic, 2.0 = very creative)
    /// - **Top-K**: Limits consideration to top K most likely tokens
    /// - **Top-P**: Nucleus sampling - considers tokens up to cumulative probability P
    /// - **Min-P**: Minimum probability threshold for token consideration
    /// 
    /// ## Output Constraints
    /// - **Max Tokens**: Maximum number of tokens to generate
    /// - **Max Length**: Alternative length constraint in characters
    /// - **Stop Sequences**: Strings that terminate generation when encountered
    /// - **EOS Handling**: End-of-sequence token behavior
    /// 
    /// ## Quality Controls
    /// - **Repetition Penalty**: Reduces likelihood of repeated phrases
    /// - **Frequency Penalty**: Penalizes frequently used tokens
    /// - **Presence Penalty**: Encourages use of new topics and concepts
    /// - **Length Penalty**: Balances response length preferences
    /// 
    /// # Examples
    /// 
    /// ## Creative Writing Configuration
    /// ```rust
    /// use fluent_ai_candle::{CandleClientBuilder, GenerationConfig};
    /// 
    /// let creative_config = GenerationConfig::new()
    ///     .with_temperature(0.9)          // High creativity
    ///     .with_top_p(0.95)               // Nucleus sampling
    ///     .with_max_tokens(1000)          // Longer responses
    ///     .with_repetition_penalty(1.1)   // Reduce repetition
    ///     .with_presence_penalty(0.1);    // Encourage topic diversity
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/creative-writer.safetensors")
    ///     .generation_config(creative_config)
    ///     .build()?;
    /// ```
    /// 
    /// ## Technical Documentation Configuration
    /// ```rust
    /// let technical_config = GenerationConfig::new()
    ///     .with_temperature(0.3)          // Lower creativity, more focused
    ///     .with_top_k(40)                 // Limited token consideration
    ///     .with_max_tokens(2048)          // Detailed explanations
    ///     .with_stop_sequences(vec![      // Stop at section boundaries
    ///         "\n\n##".to_string(),
    ///         "\n---".to_string()
    ///     ])
    ///     .with_frequency_penalty(0.05);  // Slight penalty for repetition
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/technical-assistant.safetensors")
    ///     .generation_config(technical_config)
    ///     .build()?;
    /// ```
    /// 
    /// ## Chat Assistant Configuration
    /// ```rust
    /// let chat_config = GenerationConfig::new()
    ///     .with_temperature(0.7)          // Balanced creativity
    ///     .with_top_p(0.9)                // Focused but diverse
    ///     .with_max_tokens(500)           // Conversational length
    ///     .with_stop_sequences(vec![      // Stop at user indicators
    ///         "User:".to_string(),
    ///         "Human:".to_string(),
    ///         "\n\n".to_string()          // Double newline
    ///     ])
    ///     .with_presence_penalty(0.2);    // Encourage varied responses
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/chat-assistant.safetensors")
    ///     .generation_config(chat_config)
    ///     .build()?;
    /// ```
    /// 
    /// ## Code Generation Configuration
    /// ```rust
    /// let code_config = GenerationConfig::new()
    ///     .with_temperature(0.1)          // Low temperature for precise code
    ///     .with_top_k(20)                 // Limited choices for syntax
    ///     .with_max_tokens(1500)          // Complete functions/methods
    ///     .with_stop_sequences(vec![      // Stop at code boundaries
    ///         "\n\n\n".to_string(),      // Triple newline
    ///         "```".to_string(),         // Code block end
    ///         "def ".to_string(),        // New function definition
    ///         "class ".to_string()       // New class definition
    ///     ])
    ///     .with_repetition_penalty(1.05); // Minimal repetition penalty
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/code-generator.safetensors")
    ///     .generation_config(code_config)
    ///     .build()?;
    /// ```
    /// 
    /// ## Reproducible Generation Configuration
    /// ```rust
    /// let reproducible_config = GenerationConfig::new()
    ///     .with_temperature(0.0)          // Deterministic
    ///     .with_seed(42)                  // Fixed seed
    ///     .with_top_k(1)                  // Always pick most likely token
    ///     .with_max_tokens(300);
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path(model_path)
    ///     .generation_config(reproducible_config)
    ///     .build()?;
    /// 
    /// // This will always generate the same output for the same input
    /// let response1 = client.complete(request.clone()).collect().await?;
    /// let response2 = client.complete(request.clone()).collect().await?;
    /// assert_eq!(response1.text, response2.text);
    /// ```
    /// 
    /// ## Dynamic Configuration Based on Use Case
    /// ```rust
    /// fn create_config_for_task(task: &str) -> GenerationConfig {
    ///     match task {
    ///         "creative" => GenerationConfig::new()
    ///             .with_temperature(0.9)
    ///             .with_top_p(0.95)
    ///             .with_presence_penalty(0.2),
    ///         "factual" => GenerationConfig::new()
    ///             .with_temperature(0.2)
    ///             .with_top_k(20)
    ///             .with_frequency_penalty(0.1),
    ///         "code" => GenerationConfig::new()
    ///             .with_temperature(0.1)
    ///             .with_top_k(10)
    ///             .with_stop_sequences(vec!["```".to_string()]),
    ///         _ => GenerationConfig::default()
    ///     }
    /// }
    /// 
    /// let task_type = "creative";
    /// let config = create_config_for_task(task_type);
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path(model_path)
    ///     .generation_config(config)
    ///     .build()?;
    /// ```
    /// 
    /// # Parameter Guidelines
    /// 
    /// ## Temperature Recommendations
    /// - **0.0 - 0.2**: Deterministic, factual responses
    /// - **0.3 - 0.7**: Balanced creativity and coherence
    /// - **0.8 - 1.2**: Creative writing, brainstorming
    /// - **1.3+**: Experimental, highly creative (may be incoherent)
    /// 
    /// ## Top-K Guidelines
    /// - **1 - 10**: Very focused, deterministic
    /// - **20 - 40**: Balanced selection
    /// - **50 - 100**: Diverse options
    /// - **Disabled**: No limit (may reduce quality)
    /// 
    /// ## Top-P (Nucleus) Guidelines
    /// - **0.1 - 0.5**: Conservative, focused
    /// - **0.6 - 0.9**: Standard range for most applications
    /// - **0.9 - 0.99**: High diversity
    /// - **1.0**: No filtering (equivalent to disabled)
    /// 
    /// # Performance Impact
    /// 
    /// - **Temperature**: Minimal impact on generation speed
    /// - **Top-K/Top-P**: Small overhead for token filtering
    /// - **Stop Sequences**: String matching overhead during generation
    /// - **Penalties**: Additional computation for each token
    /// 
    /// # Quality vs. Performance Trade-offs
    /// 
    /// ```rust
    /// // High quality, slower generation
    /// let high_quality_config = GenerationConfig::new()
    ///     .with_temperature(0.7)
    ///     .with_top_k(50)
    ///     .with_top_p(0.9)
    ///     .with_repetition_penalty(1.1)
    ///     .with_frequency_penalty(0.1)
    ///     .with_presence_penalty(0.1);
    /// 
    /// // Fast generation, good quality
    /// let fast_config = GenerationConfig::new()
    ///     .with_temperature(0.7)
    ///     .with_top_k(20)        // Reduced computation
    ///     .with_max_tokens(200); // Shorter responses
    /// 
    /// // Maximum speed, basic quality
    /// let speed_config = GenerationConfig::new()
    ///     .with_temperature(0.0) // No sampling
    ///     .with_top_k(1)         // Greedy decoding
    ///     .with_max_tokens(100); // Short responses
    /// ```
    /// 
    /// # Builder Pattern
    /// 
    /// Returns `Self` for fluent method chaining. The generation configuration
    /// is stored and used during inference to control text generation behavior.
    /// 
    /// # Thread Safety
    /// 
    /// This method moves `self`, ensuring thread-safe configuration.
    /// Each builder instance maintains independent generation settings.
    /// 
    /// # Architecture Compliance
    /// 
    /// - ✅ **Quality Control**: Comprehensive parameters for output customization
    /// - ✅ **Performance**: Minimal overhead for parameter application
    /// - ✅ **Flexibility**: Supports wide range of generation strategies
    /// - ✅ **Defaults**: Sensible defaults with easy customization
    #[inline(always)]
    pub fn generation_config(mut self, config: GenerationConfig) -> Self {
        self.config.generation_config = config;
        self
    }

    /// Enables model quantization for memory optimization and inference acceleration
    /// 
    /// Configures the model to use quantization techniques that reduce memory usage
    /// and potentially increase inference speed by representing model weights and
    /// activations with lower precision numeric formats. This is especially beneficial
    /// for deploying large models on resource-constrained hardware.
    /// 
    /// # Arguments
    /// 
    /// * `quantization_type` - The quantization strategy to apply:
    ///   - `QuantizationType::Int8` - 8-bit integer quantization (significant memory reduction)
    ///   - `QuantizationType::Int4` - 4-bit integer quantization (maximum compression)
    ///   - `QuantizationType::Q4_0` - 4-bit with zero-point optimization
    ///   - `QuantizationType::Q4_1` - 4-bit with scale and zero-point
    ///   - `QuantizationType::Q8_0` - 8-bit with optimized layout
    /// 
    /// # Quantization Benefits
    /// 
    /// ## Memory Reduction
    /// - **FP16 → INT8**: ~50% memory reduction (16-bit → 8-bit)
    /// - **FP16 → INT4**: ~75% memory reduction (16-bit → 4-bit)
    /// - **Enables Larger Models**: Fit models that wouldn't fit in memory unquantized
    /// - **Batch Processing**: Support larger batch sizes with same memory
    /// 
    /// ## Performance Benefits
    /// - **Faster Loading**: Reduced model file size means faster initialization
    /// - **Memory Bandwidth**: Less data movement between memory and compute units
    /// - **Cache Efficiency**: Better cache utilization with smaller data footprint
    /// - **Hardware Acceleration**: Some devices have optimized INT8 execution units
    /// 
    /// # Quality Trade-offs
    /// 
    /// ## Quality Impact by Quantization Level
    /// - **INT8**: Minimal quality loss (< 1% degradation for most models)
    /// - **INT4**: Moderate quality loss (2-5% degradation, model dependent)
    /// - **Aggressive**: Some quality loss acceptable for resource constraints
    /// 
    /// ## Model Architecture Considerations
    /// - **Large Models**: Better tolerance to quantization (7B+ parameters)
    /// - **Small Models**: More sensitive to quantization effects
    /// - **Architecture Type**: Transformer models generally quantize well
    /// 
    /// # Examples
    /// 
    /// ## INT8 Quantization (Recommended)
    /// ```rust
    /// use fluent_ai_candle::{CandleClientBuilder, QuantizationType};
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/llama-7b.safetensors")
    ///     .quantization(QuantizationType::Int8)    // 50% memory reduction
    ///     .build()?;
    /// 
    /// // Model uses ~50% less memory with minimal quality loss
    /// println!("INT8 quantized model ready");
    /// ```
    /// 
    /// ## Aggressive INT4 Quantization
    /// ```rust
    /// // For maximum memory efficiency
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/large-model.safetensors")
    ///     .quantization(QuantizationType::Int4)    // 75% memory reduction
    ///     .device_type(DeviceType::Cpu)           // CPU often better for INT4
    ///     .build()?;
    /// 
    /// // Significant memory savings, some quality trade-off
    /// println!("INT4 quantized model ready - maximum compression");
    /// ```
    /// 
    /// ## GPU Memory Optimization
    /// ```rust
    /// // Fit larger model on GPU with limited VRAM
    /// let gpu_client = CandleClientBuilder::new()
    ///     .model_path("models/13b-model.safetensors")  // Large model
    ///     .device_type(DeviceType::Cuda)
    ///     .quantization(QuantizationType::Q8_0)        // Optimized 8-bit
    ///     .build()?;
    /// 
    /// // Check if model fits in GPU memory
    /// match gpu_client.device() {
    ///     Device::Cuda(_) => println!("✅ Large model fits on GPU with quantization"),
    ///     _ => println!("⚠️  Fallback to CPU - insufficient GPU memory"),
    /// }
    /// ```
    /// 
    /// ## Dynamic Quantization Selection
    /// ```rust
    /// use std::env;
    /// 
    /// // Select quantization based on available memory
    /// let available_memory_gb = get_available_gpu_memory_gb();
    /// let model_size_gb = estimate_model_size_gb(&model_path);
    /// 
    /// let quantization = if available_memory_gb > model_size_gb * 2 {
    ///     None  // No quantization needed
    /// } else if available_memory_gb > model_size_gb {
    ///     Some(QuantizationType::Int8)  // Moderate compression
    /// } else {
    ///     Some(QuantizationType::Int4)  // Maximum compression
    /// };
    /// 
    /// let mut builder = CandleClientBuilder::new()
    ///     .model_path(model_path);
    /// 
    /// if let Some(quant_type) = quantization {
    ///     builder = builder.quantization(quant_type);
    ///     println!("Using quantization: {:?}", quant_type);
    /// } else {
    ///     println!("No quantization needed");
    /// }
    /// 
    /// let client = builder.build()?;
    /// ```
    /// 
    /// ## Quality Comparison Testing
    /// ```rust
    /// // Test quality impact of different quantization levels
    /// let test_prompt = "Explain quantum computing in simple terms.";
    /// let quantization_types = vec![
    ///     None,                           // No quantization (baseline)
    ///     Some(QuantizationType::Int8),   // 8-bit
    ///     Some(QuantizationType::Int4),   // 4-bit
    /// ];
    /// 
    /// for (i, quant_type) in quantization_types.iter().enumerate() {
    ///     let mut builder = CandleClientBuilder::new()
    ///         .model_path(model_path);
    ///     
    ///     if let Some(qt) = quant_type {
    ///         builder = builder.quantization(*qt);
    ///     }
    ///     
    ///     let client = builder.build()?;
    ///     client.initialize().await?;
    ///     
    ///     let start = std::time::Instant::now();
    ///     let response = client.complete(test_request.clone()).collect().await?;
    ///     let duration = start.elapsed();
    ///     
    ///     println!("Configuration {}: {:?}", i, quant_type);
    ///     println!("  Generation time: {:?}", duration);
    ///     println!("  Response quality: {} chars", response.text.len());
    ///     println!("  Memory usage: {}MB", get_memory_usage());
    ///     println!();
    /// }
    /// ```
    /// 
    /// ## Production Deployment Configuration
    /// ```rust
    /// // Production setup with monitoring
    /// let production_client = CandleClientBuilder::new()
    ///     .model_path("models/production-model.safetensors")
    ///     .device_type(DeviceType::Auto)
    ///     .quantization(QuantizationType::Int8)     // Balance quality and memory
    ///     .max_concurrent_requests(10)             // High throughput
    ///     .build()?;
    /// 
    /// production_client.initialize().await?;
    /// 
    /// // Monitor quantization effectiveness
    /// let metrics = production_client.metrics();
    /// println!("Production client ready with INT8 quantization");
    /// println!("Memory efficiency: Optimized for high throughput");
    /// ```
    /// 
    /// # Hardware Considerations
    /// 
    /// ## GPU Quantization
    /// - **Modern GPUs**: Hardware support for INT8 operations
    /// - **Tensor Cores**: Some GPUs accelerate quantized operations
    /// - **Memory Bandwidth**: Quantization reduces memory transfer overhead
    /// 
    /// ## CPU Quantization
    /// - **SIMD Instructions**: Vectorized quantized operations
    /// - **Cache Efficiency**: Smaller data fits better in CPU caches
    /// - **Thread Scaling**: More data fits in memory for parallel processing
    /// 
    /// ## Apple Silicon
    /// - **Neural Engine**: Optimized for quantized inference
    /// - **Unified Memory**: Benefits from reduced memory pressure
    /// - **Metal Shaders**: Custom quantization kernel support
    /// 
    /// # Quantization Process
    /// 
    /// The quantization is applied during model loading:
    /// 1. **Weight Quantization**: Model weights converted to lower precision
    /// 2. **Calibration**: Activation ranges determined from calibration data
    /// 3. **Scale Computation**: Quantization scales computed for accuracy
    /// 4. **Runtime**: Quantized operations used during inference
    /// 
    /// # Error Handling
    /// 
    /// ```rust
    /// match CandleClientBuilder::new()
    ///     .model_path(model_path)
    ///     .quantization(QuantizationType::Int4)
    ///     .build() 
    /// {
    ///     Err(CandleError::QuantizationUnsupported(msg)) => {
    ///         eprintln!("Quantization not supported: {}", msg);
    ///         // Fallback to no quantization
    ///         let fallback = CandleClientBuilder::new()
    ///             .model_path(model_path)
    ///             .build()?;
    ///     }
    ///     Err(CandleError::InsufficientMemory(msg)) => {
    ///         eprintln!("Even with quantization, insufficient memory: {}", msg);
    ///         // Try more aggressive quantization or smaller model
    ///     }
    ///     Ok(client) => {
    ///         println!("Quantized client ready");
    ///     }
    /// }
    /// ```
    /// 
    /// # Builder Pattern
    /// 
    /// Returns `Self` for fluent method chaining. Quantization is enabled and
    /// the quantization type is stored for use during model loading.
    /// 
    /// # Thread Safety
    /// 
    /// This method moves `self`, ensuring thread-safe configuration.
    /// Quantization settings apply to the entire model and all inference operations.
    /// 
    /// # Architecture Compliance
    /// 
    /// - ✅ **Memory Efficiency**: Significant reduction in memory usage
    /// - ✅ **Performance**: Potential speed improvements on compatible hardware
    /// - ✅ **Quality Preservation**: Advanced quantization maintains model quality
    /// - ✅ **Hardware Optimization**: Leverages hardware-accelerated quantized operations
    #[inline(always)]
    pub fn quantization(mut self, quantization_type: QuantizationType) -> Self {
        self.config.enable_quantization = true;
        self.config.quantization_type = quantization_type;
        self
    }

    /// Configures the maximum number of concurrent completion requests the client can handle
    /// 
    /// Sets the concurrency limit for simultaneous completion requests to prevent resource
    /// exhaustion and maintain stable performance under load. This limit applies to both
    /// streaming and non-streaming requests, providing backpressure when the system is
    /// at capacity.
    /// 
    /// # Arguments
    /// 
    /// * `max` - Maximum concurrent requests (1 to 1000+):
    ///   - **Low (1-4)**: Single-user applications, development, testing
    ///   - **Medium (5-20)**: Small team applications, moderate load services
    ///   - **High (20-100)**: Production services, high-throughput applications
    ///   - **Very High (100+)**: Large-scale deployments with extensive resources
    /// 
    /// # Concurrency Benefits
    /// 
    /// ## Throughput Optimization
    /// - **Parallel Processing**: Multiple requests processed simultaneously
    /// - **Resource Utilization**: Better utilization of available CPU/GPU resources
    /// - **Pipeline Efficiency**: Overlapping I/O and compute operations
    /// - **Batch Processing**: Opportunities for batched inference operations
    /// 
    /// ## User Experience
    /// - **Reduced Latency**: Requests don't wait in single-threaded queue
    /// - **Scalability**: Service can handle multiple users simultaneously
    /// - **Responsiveness**: Interactive applications remain responsive under load
    /// - **Load Distribution**: Smooth handling of traffic spikes
    /// 
    /// # Resource Considerations
    /// 
    /// ## Memory Usage
    /// Each concurrent request requires:
    /// - **Context Memory**: Storage for prompt and generated tokens
    /// - **KV Cache**: Key-value cache for attention mechanism
    /// - **Working Memory**: Temporary tensors and computation buffers
    /// - **Total**: ~50-500MB per request depending on model size and context length
    /// 
    /// ## Compute Resources
    /// - **CPU Cores**: Each request can utilize available CPU threads
    /// - **GPU Memory**: Shared GPU memory across all requests
    /// - **GPU Compute**: Parallel execution on GPU cores
    /// - **Memory Bandwidth**: Shared memory bandwidth across requests
    /// 
    /// # Examples
    /// 
    /// ## Single-User Development Setup
    /// ```rust
    /// use fluent_ai_candle::CandleClientBuilder;
    /// 
    /// let dev_client = CandleClientBuilder::new()
    ///     .model_path("models/dev-model.safetensors")
    ///     .max_concurrent_requests(1)       // Single request at a time
    ///     .build()?;
    /// 
    /// // Ideal for development, testing, and debugging
    /// println!("Development client - single request processing");
    /// ```
    /// 
    /// ## Small Team Application
    /// ```rust
    /// let team_client = CandleClientBuilder::new()
    ///     .model_path("models/team-model.safetensors")
    ///     .max_concurrent_requests(5)       // Small team concurrency
    ///     .device_type(DeviceType::Auto)
    ///     .build()?;
    /// 
    /// // Suitable for 5-10 team members with moderate usage
    /// println!("Team client ready - supports 5 concurrent users");
    /// ```
    /// 
    /// ## Production Service Configuration
    /// ```rust
    /// let production_client = CandleClientBuilder::new()
    ///     .model_path("models/production-model.safetensors")
    ///     .device_type(DeviceType::Cuda)    // GPU for high throughput
    ///     .quantization(QuantizationType::Int8)  // Memory efficiency
    ///     .max_concurrent_requests(25)      // High concurrency
    ///     .build()?;
    /// 
    /// // Production setup for high-throughput service
    /// println!("Production client - 25 concurrent requests supported");
    /// ```
    /// 
    /// ## Memory-Based Limit Calculation
    /// ```rust
    /// // Calculate optimal concurrency based on available memory
    /// let available_memory_gb = get_available_gpu_memory_gb();
    /// let model_size_gb = estimate_model_size_gb(&model_path);
    /// let memory_per_request_gb = 0.5;  // Estimated memory per request
    /// 
    /// let optimal_concurrency = std::cmp::max(1, 
    ///     ((available_memory_gb - model_size_gb) / memory_per_request_gb) as u32
    /// );
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path(model_path)
    ///     .max_concurrent_requests(optimal_concurrency)
    ///     .build()?;
    /// 
    /// println!("Optimal concurrency: {} requests", optimal_concurrency);
    /// ```
    /// 
    /// ## Dynamic Scaling Configuration
    /// ```rust
    /// use std::env;
    /// 
    /// // Environment-based concurrency configuration
    /// let concurrency = env::var("MAX_CONCURRENT_REQUESTS")
    ///     .unwrap_or_else(|_| "10".to_string())
    ///     .parse::<u32>()
    ///     .unwrap_or(10);
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path(model_path)
    ///     .max_concurrent_requests(concurrency)
    ///     .build()?;
    /// 
    /// println!("Configured for {} concurrent requests", concurrency);
    /// ```
    /// 
    /// ## Load Testing and Validation
    /// ```rust
    /// use tokio::time::{timeout, Duration};
    /// use futures_util::stream::FuturesUnordered;
    /// 
    /// // Test concurrent request handling
    /// async fn test_concurrency(client: Arc<CandleCompletionClient>, max_requests: u32) {
    ///     let mut futures = FuturesUnordered::new();
    ///     
    ///     // Launch maximum concurrent requests
    ///     for i in 0..max_requests {
    ///         let client = Arc::clone(&client);
    ///         let request = create_test_request(i);
    ///         
    ///         futures.push(async move {
    ///             let start = std::time::Instant::now();
    ///             let result = client.complete(request).collect().await;
    ///             (i, start.elapsed(), result)
    ///         });
    ///     }
    ///     
    ///     // Collect results
    ///     let mut completed = 0;
    ///     let mut total_time = Duration::ZERO;
    ///     
    ///     while let Some((request_id, duration, result)) = futures.next().await {
    ///         match result {
    ///             Ok(response) => {
    ///                 completed += 1;
    ///                 total_time += duration;
    ///                 println!("Request {} completed in {:?}", request_id, duration);
    ///             }
    ///             Err(e) => {
    ///                 eprintln!("Request {} failed: {}", request_id, e);
    ///             }
    ///         }
    ///     }
    ///     
    ///     println!("Completed {}/{} requests", completed, max_requests);
    ///     println!("Average time: {:?}", total_time / completed);
    /// }
    /// 
    /// // Test the configured concurrency
    /// test_concurrency(Arc::new(client), 25).await;
    /// ```
    /// 
    /// ## Rate Limiting Integration
    /// ```rust
    /// use tokio::time::{interval, Duration};
    /// 
    /// // Implement rate limiting with concurrency control
    /// async fn process_requests_with_rate_limit(
    ///     client: Arc<CandleCompletionClient>,
    ///     requests: Vec<CompletionRequest>,
    ///     requests_per_second: u32
    /// ) {
    ///     let mut interval = interval(Duration::from_millis(1000 / requests_per_second as u64));
    ///     let mut active_futures = FuturesUnordered::new();
    ///     let mut request_iter = requests.into_iter();
    ///     
    ///     loop {
    ///         tokio::select! {
    ///             // Rate-limited request submission
    ///             _ = interval.tick() => {
    ///                 if let Some(request) = request_iter.next() {
    ///                     let client = Arc::clone(&client);
    ///                     active_futures.push(async move {
    ///                         client.complete(request).collect().await
    ///                     });
    ///                 }
    ///             }
    ///             
    ///             // Handle completed requests
    ///             result = active_futures.next(), if !active_futures.is_empty() => {
    ///                 if let Some(result) = result {
    ///                     match result {
    ///                         Ok(response) => println!("Request completed: {} tokens", 
    ///                                                 response.usage.total_tokens),
    ///                         Err(e) => eprintln!("Request failed: {}", e),
    ///                     }
    ///                 }
    ///             }
    ///             
    ///             else => break, // No more requests and no active futures
    ///         }
    ///     }
    /// }
    /// ```
    /// 
    /// # Performance Guidelines
    /// 
    /// ## Concurrency vs. Model Size
    /// - **Small Models (< 1B)**: High concurrency (50-100+) often beneficial
    /// - **Medium Models (1-7B)**: Moderate concurrency (10-30) optimal
    /// - **Large Models (7B+)**: Lower concurrency (2-10) due to memory constraints
    /// 
    /// ## Hardware-Specific Recommendations
    /// - **CPU Only**: Concurrency = CPU cores × 2-4
    /// - **Single GPU**: Concurrency limited by GPU memory
    /// - **Multi-GPU**: Higher concurrency with proper load balancing
    /// - **High Memory**: Can support higher concurrency levels
    /// 
    /// # Error Handling
    /// 
    /// When concurrent request limits are exceeded:
    /// ```rust
    /// // Client automatically returns rate limiting errors
    /// match client.complete(request).collect().await {
    ///     Err(CandleError::CompletionError(CompletionError::RateLimited { 
    ///         retry_after, message 
    ///     })) => {
    ///         println!("Rate limited: {}. Retry in {}ms", message, retry_after);
    ///         tokio::time::sleep(Duration::from_millis(retry_after)).await;
    ///         // Retry request
    ///     }
    ///     Ok(response) => {
    ///         println!("Request completed successfully");
    ///     }
    ///     Err(e) => {
    ///         eprintln!("Other error: {}", e);
    ///     }
    /// }
    /// ```
    /// 
    /// # Monitoring and Metrics
    /// 
    /// ```rust
    /// // Monitor concurrency usage
    /// let metrics = client.metrics();
    /// let current_concurrent = metrics.concurrent_requests.load(Ordering::Relaxed);
    /// let total_requests = metrics.total_requests.load(Ordering::Relaxed);
    /// let successful_requests = metrics.successful_requests.load(Ordering::Relaxed);
    /// 
    /// println!("Current concurrent requests: {}", current_concurrent);
    /// println!("Success rate: {:.1}%", 
    ///          successful_requests as f64 / total_requests as f64 * 100.0);
    /// ```
    /// 
    /// # Builder Pattern
    /// 
    /// Returns `Self` for fluent method chaining. The concurrency limit is stored
    /// in the configuration and enforced during request processing.
    /// 
    /// # Thread Safety
    /// 
    /// This method moves `self`, ensuring thread-safe configuration.
    /// The concurrency limit applies to all client operations.
    /// 
    /// # Architecture Compliance
    /// 
    /// - ✅ **Scalability**: Enables high-throughput concurrent processing
    /// - ✅ **Resource Protection**: Prevents resource exhaustion under load
    /// - ✅ **Backpressure**: Provides clear feedback when capacity exceeded
    /// - ✅ **Performance**: Optimizes resource utilization across requests
    #[inline(always)]
    pub fn max_concurrent_requests(mut self, max: u32) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }

    /// Constructs the final CandleCompletionClient with all configured settings
    /// 
    /// Creates and returns a fully configured completion client using all the settings
    /// specified through the builder methods. This method performs validation of the
    /// configuration, initializes core components, and prepares the client for use.
    /// 
    /// # Returns
    /// 
    /// `CandleResult<CandleCompletionClient>` - Result containing:
    /// - `Ok(client)` - Successfully created client ready for initialization
    /// - `Err(error)` - Configuration validation error with detailed information
    /// 
    /// # Validation Process
    /// 
    /// ## Configuration Validation
    /// 1. **Model Path**: Verifies model path is specified and accessible
    /// 2. **Device Compatibility**: Checks device type availability and features
    /// 3. **Memory Requirements**: Estimates memory needs vs. available resources
    /// 4. **Quantization Support**: Validates quantization compatibility
    /// 5. **Parameter Ranges**: Ensures all parameters are within valid ranges
    /// 
    /// ## Component Initialization
    /// 1. **Device Selection**: Initializes the compute device (CPU/CUDA/Metal)
    /// 2. **Model Preparation**: Sets up model loading infrastructure
    /// 3. **Tokenizer Setup**: Configures tokenizer with specified or default settings
    /// 4. **Generator Creation**: Initializes the text generation engine
    /// 5. **Metrics Setup**: Prepares performance monitoring infrastructure
    /// 
    /// # Client State After Build
    /// 
    /// The returned client is in an **uninitialized** state:
    /// - **Configuration**: Complete and validated
    /// - **Components**: Created but not loaded (model, tokenizer)
    /// - **Ready for**: `initialize()` call to load model and tokenizer
    /// - **Not Ready for**: Completion requests (will return not-initialized errors)
    /// 
    /// # Examples
    /// 
    /// ## Basic Client Creation
    /// ```rust
    /// use fluent_ai_candle::{CandleClientBuilder, DeviceType};
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/gpt-3.5.safetensors")
    ///     .device_type(DeviceType::Auto)
    ///     .build()?;
    /// 
    /// // Client created but not yet initialized
    /// assert!(!client.is_initialized());
    /// println!("Client built successfully");
    /// 
    /// // Initialize before use
    /// client.initialize().await?;
    /// assert!(client.is_initialized());
    /// ```
    /// 
    /// ## Advanced Configuration Build
    /// ```rust
    /// use fluent_ai_candle::{
    ///     CandleClientBuilder, DeviceType, QuantizationType, GenerationConfig
    /// };
    /// 
    /// let generation_config = GenerationConfig::new()
    ///     .with_temperature(0.8)
    ///     .with_max_tokens(1000)
    ///     .with_top_p(0.9);
    /// 
    /// let client = CandleClientBuilder::new()
    ///     .model_path("models/llama-7b.safetensors")
    ///     .tokenizer_path("tokenizers/llama-tokenizer.json")
    ///     .device_type(DeviceType::Cuda)
    ///     .quantization(QuantizationType::Int8)
    ///     .generation_config(generation_config)
    ///     .max_concurrent_requests(10)
    ///     .build()?;
    /// 
    /// println!("Advanced client configuration built successfully");
    /// println!("Device: {:?}", client.device());
    /// ```
    /// 
    /// ## Error Handling During Build
    /// ```rust
    /// match CandleClientBuilder::new()
    ///     .model_path("nonexistent/model.safetensors")
    ///     .device_type(DeviceType::Cuda)
    ///     .build() 
    /// {
    ///     Ok(client) => {
    ///         println!("✅ Client built successfully");
    ///         
    ///         // Proceed with initialization
    ///         client.initialize().await?;
    ///     }
    ///     Err(CandleError::ModelNotFound(path)) => {
    ///         eprintln!("❌ Model file not found: {}", path);
    ///         
    ///         // Try alternative model path
    ///         let fallback_client = CandleClientBuilder::new()
    ///             .model_path("models/fallback-model.safetensors")
    ///             .device_type(DeviceType::Auto)
    ///             .build()?;
    ///     }
    ///     Err(CandleError::Msg(msg)) if msg.contains("CUDA") => {
    ///         eprintln!("❌ CUDA not available: {}", msg);
    ///         
    ///         // Fallback to CPU
    ///         let cpu_client = CandleClientBuilder::new()
    ///             .model_path(model_path)
    ///             .device_type(DeviceType::Cpu)
    ///             .build()?;
    ///     }
    ///     Err(CandleError::InsufficientMemory(msg)) => {
    ///         eprintln!("❌ Insufficient memory: {}", msg);
    ///         
    ///         // Try with quantization
    ///         let quantized_client = CandleClientBuilder::new()
    ///             .model_path(model_path)
    ///             .quantization(QuantizationType::Int8)
    ///             .build()?;
    ///     }
    ///     Err(e) => {
    ///         eprintln!("❌ Build failed: {}", e);
    ///         return Err(e);
    ///     }
    /// }
    /// ```
    /// 
    /// ## Build with Validation
    /// ```rust
    /// fn build_validated_client(config: ClientConfig) -> CandleResult<CandleCompletionClient> {
    ///     // Pre-build validation
    ///     if !std::path::Path::new(&config.model_path).exists() {
    ///         return Err(CandleError::ModelNotFound(config.model_path));
    ///     }
    ///     
    ///     if config.max_concurrent_requests > 100 {
    ///         return Err(CandleError::ValidationError(
    ///             "Excessive concurrent requests - maximum is 100".to_string()
    ///         ));
    ///     }
    ///     
    ///     // Build with validated configuration
    ///     let client = CandleClientBuilder::new()
    ///         .model_path(config.model_path)
    ///         .device_type(config.device_type)
    ///         .max_concurrent_requests(config.max_concurrent_requests)
    ///         .build()?;
    ///     
    ///     println!("Client built with validation successful");
    ///     Ok(client)
    /// }
    /// ```
    /// 
    /// ## Resource Estimation
    /// ```rust
    /// // Estimate resource requirements before building
    /// fn estimate_and_build(model_path: &str) -> CandleResult<CandleCompletionClient> {
    ///     let model_size_gb = estimate_model_size_gb(model_path)?;
    ///     let available_gpu_memory = get_available_gpu_memory_gb();
    ///     let available_system_memory = get_available_system_memory_gb();
    ///     
    ///     println!("Model size: {:.1}GB", model_size_gb);
    ///     println!("Available GPU memory: {:.1}GB", available_gpu_memory);
    ///     println!("Available system memory: {:.1}GB", available_system_memory);
    ///     
    ///     let mut builder = CandleClientBuilder::new()
    ///         .model_path(model_path);
    ///     
    ///     // Configure based on available resources
    ///     if model_size_gb < available_gpu_memory * 0.8 {
    ///         println!("Using GPU - sufficient VRAM available");
    ///         builder = builder.device_type(DeviceType::Cuda);
    ///     } else if model_size_gb * 0.5 < available_gpu_memory * 0.8 {
    ///         println!("Using GPU with quantization - tight VRAM");
    ///         builder = builder
    ///             .device_type(DeviceType::Cuda)
    ///             .quantization(QuantizationType::Int8);
    ///     } else if model_size_gb < available_system_memory * 0.6 {
    ///         println!("Using CPU - insufficient GPU memory");
    ///         builder = builder.device_type(DeviceType::Cpu);
    ///     } else {
    ///         return Err(CandleError::InsufficientMemory(
    ///             "Model too large for available memory".to_string()
    ///         ));
    ///     }
    ///     
    ///     builder.build()
    /// }
    /// ```
    /// 
    /// ## Production Setup with Monitoring
    /// ```rust
    /// // Production client with comprehensive error handling
    /// async fn create_production_client() -> CandleResult<CandleCompletionClient> {
    ///     let client = CandleClientBuilder::new()
    ///         .model_path(std::env::var("MODEL_PATH")?)
    ///         .device_type(DeviceType::Auto)
    ///         .quantization(QuantizationType::Int8)
    ///         .max_concurrent_requests(20)
    ///         .generation_config(
    ///             GenerationConfig::new()
    ///                 .with_temperature(0.7)
    ///                 .with_max_tokens(1000)
    ///         )
    ///         .build()?;
    ///     
    ///     println!("✅ Production client built");
    ///     
    ///     // Initialize and validate
    ///     client.initialize().await?;
    ///     
    ///     // Perform health check
    ///     let health_request = create_health_check_request();
    ///     let health_response = client.complete(health_request).collect().await?;
    ///     
    ///     if health_response.text.len() > 10 {
    ///         println!("✅ Health check passed - client ready for production");
    ///         Ok(client)
    ///     } else {
    ///         Err(CandleError::ValidationError(
    ///             "Health check failed - invalid response".to_string()
    ///         ))
    ///     }
    /// }
    /// ```
    /// 
    /// # Performance Considerations
    /// 
    /// - **Build Time**: Typically fast (< 100ms) as heavy operations deferred to `initialize()`
    /// - **Memory Usage**: Minimal during build, full allocation during initialization
    /// - **Validation Overhead**: Configuration validation adds minimal cost
    /// - **Error Recovery**: Build failures are fast and don't consume significant resources
    /// 
    /// # Common Build Errors
    /// 
    /// ## Configuration Errors
    /// - **Missing Model Path**: Model path not specified or empty
    /// - **Invalid Device**: Requested device type not available or supported
    /// - **Parameter Range**: Generation parameters outside valid ranges
    /// - **Resource Estimates**: Insufficient memory for configuration
    /// 
    /// ## Path Resolution Errors
    /// - **Model Not Found**: Model file doesn't exist at specified path
    /// - **Permission Denied**: No read access to model file
    /// - **Invalid Format**: Model file format not supported
    /// - **Tokenizer Issues**: Tokenizer path invalid or incompatible
    /// 
    /// # Builder Pattern
    /// 
    /// This method consumes the builder (takes `self` by value) and returns
    /// the final client instance. The builder cannot be reused after calling `build()`.
    /// 
    /// # Thread Safety
    /// 
    /// The build process is thread-safe, and the returned client can be safely
    /// shared across threads using `Arc<CandleCompletionClient>`.
    /// 
    /// # Architecture Compliance
    /// 
    /// - ✅ **Error Handling**: Comprehensive validation with detailed error messages
    /// - ✅ **Resource Management**: Efficient resource allocation patterns
    /// - ✅ **Performance**: Fast build with deferred heavy operations
    /// - ✅ **Thread Safety**: Safe for concurrent access and sharing
    #[inline(always)]
    pub fn build(self) -> CandleResult<CandleCompletionClient> {
        CandleCompletionClient::new(self.config)
    }
}

impl Default for CandleClientBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}