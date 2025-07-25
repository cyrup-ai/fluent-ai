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
    /// Create new configuration with defaults
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

    /// Set device for tensor loading
    #[inline(always)]
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set default data type
    #[inline(always)]
    pub const fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set maximum memory mapping size
    #[inline(always)]
    pub const fn with_max_mmap_size(mut self, size: u64) -> Self {
        self.max_mmap_size = size;
        self
    }

    /// Enable memory mapping
    #[inline(always)]
    pub const fn enable_memory_mapping(mut self) -> Self {
        self.flags |= 1;
        self
    }

    /// Disable memory mapping
    #[inline(always)]
    pub const fn disable_memory_mapping(mut self) -> Self {
        self.flags &= !1;
        self
    }

    /// Enable tensor validation
    #[inline(always)]
    pub const fn enable_validation(mut self) -> Self {
        self.flags |= 2;
        self
    }

    /// Enable shape caching
    #[inline(always)]
    pub const fn enable_shape_caching(mut self) -> Self {
        self.flags |= 4;
        self
    }

    /// Enable lazy loading
    #[inline(always)]
    pub const fn enable_lazy_loading(mut self) -> Self {
        self.flags |= 8;
        self
    }

    /// Enable device placement optimization
    #[inline(always)]
    pub const fn enable_device_optimization(mut self) -> Self {
        self.flags |= 16;
        self
    }

    /// Enable tensor fusion optimization
    #[inline(always)]
    pub const fn enable_tensor_fusion(mut self) -> Self {
        self.flags |= 32;
        self
    }

    /// Enable tensor caching
    #[inline(always)]
    pub const fn enable_tensor_cache(mut self) -> Self {
        self.flags |= 64;
        self
    }

    /// Disable tensor caching
    #[inline(always)]
    pub const fn disable_tensor_cache(mut self) -> Self {
        self.flags &= !64;
        self
    }

    /// Set tensor name prefix
    #[inline(always)]
    pub fn with_tensor_prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.tensor_prefix = Some(prefix.into());
        self
    }

    /// Check if tensor caching is enabled
    #[inline(always)]
    pub const fn tensor_cache_enabled(&self) -> bool {
        (self.flags & 64) != 0
    }

    /// Check if memory mapping is enabled
    #[inline(always)]
    pub const fn use_memory_mapping(&self) -> bool {
        (self.flags & 1) != 0
    }

    /// Check if tensor validation is enabled
    #[inline(always)]
    pub const fn validate_tensors(&self) -> bool {
        (self.flags & 2) != 0
    }

    /// Check if shape caching is enabled
    #[inline(always)]
    pub const fn cache_shapes(&self) -> bool {
        (self.flags & 4) != 0
    }

    /// Check if lazy loading is enabled
    #[inline(always)]
    pub const fn lazy_loading(&self) -> bool {
        (self.flags & 8) != 0
    }

    /// Check if device optimization is enabled
    #[inline(always)]
    pub const fn device_optimization(&self) -> bool {
        (self.flags & 16) != 0
    }

    /// Check if tensor fusion is enabled
    #[inline(always)]
    pub const fn tensor_fusion(&self) -> bool {
        (self.flags & 32) != 0
    }

    /// Get device reference
    #[inline(always)]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// Get data type
    #[inline(always)]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get maximum memory mapping size
    #[inline(always)]
    pub const fn max_mmap_size(&self) -> u64 {
        self.max_mmap_size
    }

    /// Get tensor prefix
    #[inline(always)]
    pub fn tensor_prefix(&self) -> Option<&str> {
        self.tensor_prefix.as_deref()
    }
}

impl Default for VarBuilderConfig {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration builder with fluent API
#[derive(Debug, Clone)]
pub struct VarBuilderConfigBuilder {
    config: VarBuilderConfig}

impl VarBuilderConfigBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self { config: VarBuilderConfig::new() }
    }

    #[inline(always)]
    pub fn device(mut self, device: Device) -> Self {
        self.config = self.config.with_device(device); self
    }

    #[inline(always)]
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.config = self.config.with_dtype(dtype); self
    }

    #[inline(always)]
    pub fn max_mmap_size(mut self, size: u64) -> Self {
        self.config = self.config.with_max_mmap_size(size); self
    }

    #[inline(always)]
    pub fn enable_memory_mapping(mut self) -> Self {
        self.config = self.config.enable_memory_mapping(); self
    }

    #[inline(always)]
    pub fn disable_memory_mapping(mut self) -> Self {
        self.config = self.config.disable_memory_mapping(); self
    }

    #[inline(always)]
    pub fn enable_validation(mut self) -> Self {
        self.config = self.config.enable_validation(); self
    }

    #[inline(always)]
    pub fn enable_shape_caching(mut self) -> Self {
        self.config = self.config.enable_shape_caching(); self
    }

    #[inline(always)]
    pub fn enable_lazy_loading(mut self) -> Self {
        self.config = self.config.enable_lazy_loading(); self
    }

    #[inline(always)]
    pub fn enable_device_optimization(mut self) -> Self {
        self.config = self.config.enable_device_optimization(); self
    }

    #[inline(always)]
    pub fn enable_tensor_fusion(mut self) -> Self {
        self.config = self.config.enable_tensor_fusion(); self
    }

    #[inline(always)]
    pub fn enable_tensor_cache(mut self) -> Self {
        self.config = self.config.enable_tensor_cache(); self
    }

    #[inline(always)]
    pub fn disable_tensor_cache(mut self) -> Self {
        self.config = self.config.disable_tensor_cache(); self
    }

    #[inline(always)]
    pub fn tensor_prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.config = self.config.with_tensor_prefix(prefix); self
    }

    #[inline(always)]
    pub fn build(self) -> VarBuilderConfig { self.config }
}

impl Default for VarBuilderConfigBuilder {
    #[inline(always)]
    fn default() -> Self { Self::new() }
}