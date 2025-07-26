//! Configuration structures and builders for VarBuilder

use candle_core::{DType, Device};

/// Ultra-compact VarBuilder configuration
///
/// All configuration stored on stack with atomic flags for runtime behavior.
/// Optimized for cache efficiency and zero-allocation operations.
#[repr(C, align(64))] // Cache line aligned
#[derive(Clone, Debug)]
pub struct VarBuilderConfig {
    /// Device for tensor loading
    device: Device,

    /// Default data type
    dtype: DType,

    /// Maximum file size for memory mapping (bytes)
    max_mmap_size: u64,

    /// Tensor name prefix for scoping
    tensor_prefix: Option<String>,

    /// Configuration flags (bit-packed)
    /// Bit 0: use_memory_mapping
    /// Bit 1: validate_tensors
    /// Bit 2: cache_shapes
    /// Bit 3: enable_lazy_loading
    /// Bit 4: optimize_device_placement
    /// Bit 5: enable_tensor_fusion
    /// Bit 6: enable_tensor_cache
    /// Bits 7-63: Reserved
    flags: u64}

impl VarBuilderConfig {
    /// Create new VarBuilder configuration with optimal default settings
    ///
    /// Creates a configuration optimized for typical ML workloads with:
    /// - CPU device by default (can be changed)
    /// - F32 data type for maximum compatibility
    /// - 2GB memory mapping limit for large models
    /// - All performance optimizations enabled
    /// - Cache-line aligned for optimal memory access
    ///
    /// # Returns
    /// A new VarBuilderConfig with production-ready defaults
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            max_mmap_size: 2 * 1024 * 1024 * 1024, // 2GB
            tensor_prefix: None,
            flags: 0b1111111, // Enable all optimizations by default
        }
    }

    /// Set the target device for tensor loading and computation
    ///
    /// Configures where tensors will be loaded and stored. Common devices
    /// include CPU, CUDA GPUs, and Metal (Apple Silicon).
    ///
    /// # Arguments
    /// * `device` - The target device (CPU, CUDA, Metal, etc.)
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Device placement affects memory bandwidth and computation speed.
    /// GPU devices typically provide better performance for large models.
    #[inline(always)]
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the default data type for loaded tensors
    ///
    /// Controls the precision and memory usage of tensors. Common types:
    /// - F32: Full precision, maximum compatibility
    /// - F16: Half precision, 50% memory savings
    /// - BF16: Brain float, good for training
    /// - I8/U8: 8-bit quantization for inference
    ///
    /// # Arguments
    /// * `dtype` - The target data type
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Lower precision types reduce memory usage but may affect accuracy.
    #[inline(always)]
    pub const fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set maximum file size for memory mapping in bytes
    ///
    /// Files larger than this limit will be loaded into memory instead
    /// of being memory-mapped. Memory mapping is more efficient for
    /// large files but has OS-specific limits.
    ///
    /// # Arguments
    /// * `size` - Maximum size in bytes for memory mapping
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Larger limits enable memory mapping for bigger models,
    /// reducing memory usage and improving load times.
    #[inline(always)]
    pub const fn with_max_mmap_size(mut self, size: u64) -> Self {
        self.max_mmap_size = size;
        self
    }

    /// Enable memory mapping for efficient large file access
    ///
    /// Memory mapping allows direct access to file contents without
    /// loading into RAM, significantly reducing memory usage for
    /// large model files. Recommended for most use cases.
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Reduces memory usage and improves load times for large models.
    /// May have slight overhead for small files.
    #[inline(always)]
    pub const fn enable_memory_mapping(mut self) -> Self {
        self.flags |= 1;
        self
    }

    /// Disable memory mapping and force loading into RAM
    ///
    /// Forces all tensor data to be loaded into system memory.
    /// Use this if memory mapping causes issues on your platform
    /// or if you need guaranteed memory locality.
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Increases memory usage but may improve access patterns
    /// for some workloads.
    #[inline(always)]
    pub const fn disable_memory_mapping(mut self) -> Self {
        self.flags &= !1;
        self
    }

    /// Enable comprehensive tensor validation during loading
    ///
    /// Validates tensor shapes, data types, and ranges during loading.
    /// Helps catch data corruption and format issues early.
    /// Recommended for development and critical applications.
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Adds validation overhead but improves reliability.
    /// Disable for maximum performance in production.
    #[inline(always)]
    pub const fn enable_validation(mut self) -> Self {
        self.flags |= 2;
        self
    }

    /// Enable caching of tensor shapes for faster repeated access
    ///
    /// Caches tensor shape information to avoid repeated computation.
    /// Particularly beneficial when accessing the same tensors multiple
    /// times or when shape queries are performance-critical.
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Improves performance for shape-heavy operations at the cost
    /// of a small amount of additional memory usage.
    #[inline(always)]
    pub const fn enable_shape_caching(mut self) -> Self {
        self.flags |= 4;
        self
    }

    /// Enable lazy loading of tensors for reduced memory usage
    ///
    /// Defers tensor loading until actually needed, reducing initial
    /// memory footprint and startup time. Particularly beneficial for
    /// large models where only a subset of tensors are used.
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Reduces initial memory usage and startup time, but may add
    /// latency to first access of each tensor.
    #[inline(always)]
    pub const fn enable_lazy_loading(mut self) -> Self {
        self.flags |= 8;
        self
    }

    /// Enable automatic device placement optimization
    ///
    /// Automatically optimizes tensor placement across available devices
    /// for optimal memory bandwidth and computation efficiency.
    /// Particularly beneficial in multi-GPU setups.
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Can significantly improve performance in multi-device environments
    /// by optimizing data locality and parallelism.
    #[inline(always)]
    pub const fn enable_device_optimization(mut self) -> Self {
        self.flags |= 16;
        self
    }

    /// Enable tensor fusion for improved memory efficiency
    ///
    /// Automatically fuses compatible tensors to reduce memory
    /// fragmentation and improve cache locality. Beneficial for
    /// models with many small tensors.
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Reduces memory fragmentation and improves cache efficiency,
    /// but may add some overhead during loading.
    #[inline(always)]
    pub const fn enable_tensor_fusion(mut self) -> Self {
        self.flags |= 32;
        self
    }

    /// Enable global tensor caching for repeated access patterns
    ///
    /// Caches frequently accessed tensors in memory to avoid repeated
    /// loading from storage. Particularly beneficial for models with
    /// shared or repeatedly accessed parameters.
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Improves performance for repeated tensor access at the cost
    /// of additional memory usage.
    #[inline(always)]
    pub const fn enable_tensor_cache(mut self) -> Self {
        self.flags |= 64;
        self
    }

    /// Disable tensor caching to minimize memory usage
    ///
    /// Disables tensor caching to reduce memory footprint.
    /// Use this when memory is constrained or when tensors
    /// are accessed infrequently.
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Performance
    /// Reduces memory usage but may increase loading times
    /// for repeatedly accessed tensors.
    #[inline(always)]
    pub const fn disable_tensor_cache(mut self) -> Self {
        self.flags &= !64;
        self
    }

    /// Set a prefix for all tensor names for scoping
    ///
    /// Adds a prefix to all tensor names for organizational purposes
    /// or to avoid naming conflicts when combining multiple models.
    /// Useful for model composition and debugging.
    ///
    /// # Arguments
    /// * `prefix` - String prefix to add to all tensor names
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Examples
    /// ```
    /// config.with_tensor_prefix("encoder.")
    /// // Tensor "weight" becomes "encoder.weight"
    /// ```
    #[inline(always)]
    pub fn with_tensor_prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.tensor_prefix = Some(prefix.into());
        self
    }

    /// Check if global tensor caching is currently enabled
    ///
    /// Returns true if tensor caching is enabled, which means
    /// frequently accessed tensors will be cached in memory
    /// for improved performance.
    ///
    /// # Returns
    /// true if tensor caching is enabled, false otherwise
    #[inline(always)]
    pub const fn tensor_cache_enabled(&self) -> bool {
        (self.flags & 64) != 0
    }

    /// Check if memory mapping is enabled for file access
    ///
    /// Returns true if memory mapping is enabled, which allows
    /// direct access to file contents without loading into RAM.
    ///
    /// # Returns
    /// true if memory mapping is enabled, false otherwise
    #[inline(always)]
    pub const fn use_memory_mapping(&self) -> bool {
        (self.flags & 1) != 0
    }

    /// Check if tensor validation is enabled during loading
    ///
    /// Returns true if tensor validation is enabled, which means
    /// tensors will be validated for correctness during loading.
    ///
    /// # Returns
    /// true if tensor validation is enabled, false otherwise
    #[inline(always)]
    pub const fn validate_tensors(&self) -> bool {
        (self.flags & 2) != 0
    }

    /// Check if tensor shape caching is enabled
    ///
    /// Returns true if shape caching is enabled, which means
    /// tensor shape information will be cached for faster
    /// repeated access.
    ///
    /// # Returns
    /// true if shape caching is enabled, false otherwise
    #[inline(always)]
    pub const fn cache_shapes(&self) -> bool {
        (self.flags & 4) != 0
    }

    /// Check if lazy loading is enabled for tensors
    ///
    /// Returns true if lazy loading is enabled, which means
    /// tensors will be loaded on-demand rather than upfront.
    ///
    /// # Returns
    /// true if lazy loading is enabled, false otherwise
    #[inline(always)]
    pub const fn lazy_loading(&self) -> bool {
        (self.flags & 8) != 0
    }

    /// Check if automatic device placement optimization is enabled
    ///
    /// Returns true if device optimization is enabled, which means
    /// tensors will be automatically placed on optimal devices.
    ///
    /// # Returns
    /// true if device optimization is enabled, false otherwise
    #[inline(always)]
    pub const fn device_optimization(&self) -> bool {
        (self.flags & 16) != 0
    }

    /// Check if tensor fusion optimization is enabled
    ///
    /// Returns true if tensor fusion is enabled, which means
    /// compatible tensors will be fused for improved efficiency.
    ///
    /// # Returns
    /// true if tensor fusion is enabled, false otherwise
    #[inline(always)]
    pub const fn tensor_fusion(&self) -> bool {
        (self.flags & 32) != 0
    }

    /// Get a reference to the configured target device
    ///
    /// Returns a reference to the device where tensors will be
    /// loaded and computations will be performed.
    ///
    /// # Returns
    /// Reference to the configured Device
    #[inline(always)]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// Get the configured default data type for tensors
    ///
    /// Returns the data type that will be used for loaded tensors
    /// unless explicitly overridden.
    ///
    /// # Returns
    /// The configured DType (F32, F16, etc.)
    #[inline(always)]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the maximum file size for memory mapping in bytes
    ///
    /// Returns the size limit above which files will be loaded
    /// into memory instead of being memory-mapped.
    ///
    /// # Returns
    /// Maximum memory mapping size in bytes
    #[inline(always)]
    pub const fn max_mmap_size(&self) -> u64 {
        self.max_mmap_size
    }

    /// Get the configured tensor name prefix
    ///
    /// Returns the prefix that will be added to all tensor names,
    /// or None if no prefix is configured.
    ///
    /// # Returns
    /// Optional reference to the tensor name prefix string
    #[inline(always)]
    pub fn tensor_prefix(&self) -> Option<&str> {
        self.tensor_prefix.as_deref()
    }
}

impl Default for VarBuilderConfig {
    /// Create a default VarBuilderConfig using new()
    ///
    /// Provides the same defaults as VarBuilderConfig::new():
    /// CPU device, F32 data type, 2GB memory mapping limit,
    /// and all optimizations enabled.
    ///
    /// # Returns
    /// A VarBuilderConfig with default settings
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration builder with fluent API for VarBuilderConfig
///
/// Provides a fluent interface for building VarBuilderConfig instances
/// with method chaining. This builder wraps VarBuilderConfig and delegates
/// to its methods for a consistent API.
#[derive(Debug, Clone)]
pub struct VarBuilderConfigBuilder {
    /// The configuration being built
    config: VarBuilderConfig}

impl VarBuilderConfigBuilder {
    /// Create a new configuration builder with default settings
    ///
    /// Creates a builder initialized with VarBuilderConfig::new() defaults.
    /// All methods return Self for fluent chaining.
    ///
    /// # Returns
    /// A new VarBuilderConfigBuilder with default configuration
    #[inline(always)]
    pub fn new() -> Self {
        Self { config: VarBuilderConfig::new() }
    }

    /// Set the target device for tensor operations
    ///
    /// # Arguments
    /// * `device` - Target device (CPU, CUDA, Metal, etc.)
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn device(mut self, device: Device) -> Self {
        self.config = self.config.with_device(device); self
    }

    /// Set the default data type for tensors
    ///
    /// # Arguments
    /// * `dtype` - Target data type (F32, F16, etc.)
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.config = self.config.with_dtype(dtype); self
    }

    /// Set maximum memory mapping file size
    ///
    /// # Arguments
    /// * `size` - Maximum size in bytes for memory mapping
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn max_mmap_size(mut self, size: u64) -> Self {
        self.config = self.config.with_max_mmap_size(size); self
    }

    /// Enable memory mapping for efficient file access
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn enable_memory_mapping(mut self) -> Self {
        self.config = self.config.enable_memory_mapping(); self
    }

    /// Disable memory mapping and force loading into RAM
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn disable_memory_mapping(mut self) -> Self {
        self.config = self.config.disable_memory_mapping(); self
    }

    /// Enable tensor validation during loading
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn enable_validation(mut self) -> Self {
        self.config = self.config.enable_validation(); self
    }

    /// Enable tensor shape caching for performance
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn enable_shape_caching(mut self) -> Self {
        self.config = self.config.enable_shape_caching(); self
    }

    /// Enable lazy loading of tensors
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn enable_lazy_loading(mut self) -> Self {
        self.config = self.config.enable_lazy_loading(); self
    }

    /// Enable automatic device placement optimization
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn enable_device_optimization(mut self) -> Self {
        self.config = self.config.enable_device_optimization(); self
    }

    /// Enable tensor fusion optimization
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn enable_tensor_fusion(mut self) -> Self {
        self.config = self.config.enable_tensor_fusion(); self
    }

    /// Enable global tensor caching
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn enable_tensor_cache(mut self) -> Self {
        self.config = self.config.enable_tensor_cache(); self
    }

    /// Disable tensor caching to save memory
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn disable_tensor_cache(mut self) -> Self {
        self.config = self.config.disable_tensor_cache(); self
    }

    /// Set a prefix for all tensor names
    ///
    /// # Arguments
    /// * `prefix` - String prefix to add to tensor names
    ///
    /// # Returns
    /// Self for method chaining
    #[inline(always)]
    pub fn tensor_prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.config = self.config.with_tensor_prefix(prefix); self
    }

    /// Build and return the final configuration
    ///
    /// Consumes the builder and returns the constructed VarBuilderConfig.
    ///
    /// # Returns
    /// The built VarBuilderConfig instance
    #[inline(always)]
    pub fn build(self) -> VarBuilderConfig { self.config }
}

impl Default for VarBuilderConfigBuilder {
    /// Create a default configuration builder
    ///
    /// Equivalent to VarBuilderConfigBuilder::new().
    ///
    /// # Returns
    /// A new VarBuilderConfigBuilder with default settings
    #[inline(always)]
    fn default() -> Self { Self::new() }
}