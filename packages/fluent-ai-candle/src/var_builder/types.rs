//! Core types and constants for VarBuilder

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use arrayvec::ArrayString;
use candle_core::{DType, Device};

/// Convert safetensors Dtype to candle DType with optimal mapping
///
/// Converts data types from the safetensors format to candle's native data types,
/// using the most appropriate candle type for each safetensors type. Some mappings
/// require promotion to larger types since candle has a more limited set of data types.
///
/// # Arguments
///
/// * `dtype` - The safetensors data type to convert
///
/// # Returns
///
/// The corresponding candle `DType`, with fallback to `F32` for unknown types
///
/// # Type Mappings
///
/// - `BOOL` → `U8` (candle doesn't have native bool)
/// - `I8`, `I16`, `I32` → `I64` (promoted to I64)
/// - `U16` → `U32` (promoted to U32)
/// - `U64` → `U32` (demoted to U32, candle doesn't support U64)
/// - `F16`, `BF16`, `F32`, `F64` → direct mapping
/// - Unknown types → `F32` (default fallback)
///
/// # Example
///
/// ```rust
/// let candle_dtype = convert_dtype(safetensors::Dtype::F32);
/// assert_eq!(candle_dtype, DType::F32);
/// ```
pub fn convert_dtype(dtype: safetensors::Dtype) -> DType {
    match dtype {
        safetensors::Dtype::BOOL => DType::U8, // Candle doesn't have native bool, use U8
        safetensors::Dtype::U8 => DType::U8,
        safetensors::Dtype::I8 => DType::I64, // Candle doesn't have I8, use I64
        safetensors::Dtype::U16 => DType::U32, // Candle doesn't have U16, use U32
        safetensors::Dtype::I16 => DType::I64, // Candle doesn't have I16, use I64
        safetensors::Dtype::U32 => DType::U32,
        safetensors::Dtype::I32 => DType::I64, // Candle doesn't have I32, use I64
        safetensors::Dtype::U64 => DType::U32, // Candle doesn't have U64, use U32
        safetensors::Dtype::I64 => DType::I64,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F64 => DType::F64,
        _ => DType::F32, // Default fallback
    }
}

/// Maximum tensor name length for stack allocation
///
/// This constant defines the maximum length for tensor names when using
/// stack-allocated `ArrayString` for zero-allocation tensor name storage.
/// Tensor names longer than this will require heap allocation.
pub const MAX_TENSOR_NAME_LEN: usize = 256;

/// Maximum number of tensors for stack allocation
///
/// This constant defines the maximum number of tensors that can be handled
/// using stack-based data structures for optimal performance. Models with
/// more tensors will use dynamic allocation.
pub const MAX_TENSORS: usize = 2048;

/// Maximum configuration entries for stack allocation
///
/// This constant defines the maximum number of configuration key-value pairs
/// that can be stored using stack-based data structures for zero-allocation
/// configuration management.
pub const MAX_CONFIG_ENTRIES: usize = 128;

/// Maximum file paths for sharded models
///
/// This constant defines the maximum number of file paths that can be handled
/// for sharded model loading, where a single model is split across multiple
/// safetensors files.
pub const MAX_FILE_PATHS: usize = 32;

/// Cache line size for alignment optimization
///
/// This constant defines the cache line size used for aligning data structures
/// to optimize memory access patterns and reduce cache misses. This value is
/// typically 64 bytes on modern x86_64 architectures.
pub const CACHE_LINE_SIZE: usize = 64;

/// Error message for tensor not found in model
///
/// This static string is used for zero-allocation error reporting when
/// a requested tensor name is not found in the loaded model.
pub const ERR_TENSOR_NOT_FOUND: &str = "Tensor not found";

/// Error message for invalid tensor shape
///
/// This static string is used for zero-allocation error reporting when
/// a tensor has an invalid or unexpected shape that doesn't match requirements.
pub const ERR_INVALID_SHAPE: &str = "Invalid tensor shape";

/// Error message for device mismatch
///
/// This static string is used for zero-allocation error reporting when
/// tensors are placed on incompatible devices or device transfer fails.
pub const ERR_DEVICE_MISMATCH: &str = "Device mismatch";

/// Error message for model loading failure
///
/// This static string is used for zero-allocation error reporting when
/// model loading operations fail due to file system, format, or other issues.
pub const ERR_MODEL_LOADING: &str = "Model loading failed";

/// Error message for metadata parsing failure
///
/// This static string is used for zero-allocation error reporting when
/// model metadata cannot be parsed or is in an unexpected format.
pub const ERR_METADATA_PARSING: &str = "Metadata parsing failed";

/// Ultra-compact tensor name with stack allocation
///
/// A stack-allocated string type for tensor names that avoids heap allocation
/// for improved performance. Can store tensor names up to `MAX_TENSOR_NAME_LEN`
/// characters without any dynamic memory allocation.
///
/// # Examples
///
/// ```rust
/// let tensor_name: TensorName = "model.layers.0.weight".try_into().unwrap();
/// ```
pub type TensorName = ArrayString<MAX_TENSOR_NAME_LEN>;

/// Ultra-compact configuration key with stack allocation
///
/// A stack-allocated string type for configuration keys that avoids heap
/// allocation. Can store configuration keys up to 64 characters without
/// any dynamic memory allocation.
///
/// # Examples
///
/// ```rust
/// let config_key: ConfigKey = "device_type".try_into().unwrap();
/// ```
pub type ConfigKey = ArrayString<64>;

/// Ultra-compact configuration value with stack allocation
///
/// A stack-allocated string type for configuration values that avoids heap
/// allocation. Can store configuration values up to 256 characters without
/// any dynamic memory allocation.
///
/// # Examples
///
/// ```rust
/// let config_value: ConfigValue = "cuda:0".try_into().unwrap();
/// ```
pub type ConfigValue = ArrayString<256>;

/// Tensor loading strategy for memory optimization
#[derive(Debug, Clone)]
pub enum TensorLoadStrategy {
    /// Load tensor immediately into memory
    Immediate,
    /// Memory-map the tensor data
    MemoryMapped,
    /// Lazy loading with on-demand creation
    Lazy}

/// Metadata for a tensor in the safetensors file
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
    /// Byte offset in file
    pub offset: usize,
    /// Byte length in file
    pub length: usize,
    /// Loading strategy
    pub strategy: TensorLoadStrategy}

/// Device placement hint for optimal performance
#[derive(Debug, Clone)]
pub enum DeviceHint {
    /// Prefer CPU placement
    PreferCpu,
    /// Prefer GPU placement
    PreferGpu,
    /// Automatic device selection
    Auto,
    /// Enforce specific device
    ForceDevice(Device)}

/// Loading statistics for performance tracking
#[repr(C, align(64))] // Cache line aligned
#[derive(Debug)]
pub struct LoadingStats {
    /// Total tensors loaded
    pub tensors_loaded: AtomicUsize,
    /// Total bytes loaded
    pub bytes_loaded: AtomicU64,
    /// Total loading time in nanoseconds
    pub loading_time_ns: AtomicU64,
    /// Cache hits
    pub cache_hits: AtomicUsize,
    /// Cache misses
    pub cache_misses: AtomicUsize,
    /// Memory mapping operations
    pub mmap_operations: AtomicUsize,
    /// Device transfers
    pub device_transfers: AtomicUsize,
    /// Validation operations
    pub validation_operations: AtomicUsize,
    /// Tensor fusions performed
    pub tensor_fusions: AtomicUsize,
    /// Memory usage in bytes (current)
    pub memory_usage_bytes: AtomicU64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: AtomicU64,
    /// Failed operations count
    pub failed_operations: AtomicUsize}

impl LoadingStats {
    /// Create new loading statistics
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            tensors_loaded: AtomicUsize::new(0),
            bytes_loaded: AtomicU64::new(0),
            loading_time_ns: AtomicU64::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            mmap_operations: AtomicUsize::new(0),
            device_transfers: AtomicUsize::new(0),
            validation_operations: AtomicUsize::new(0),
            tensor_fusions: AtomicUsize::new(0),
            memory_usage_bytes: AtomicU64::new(0),
            peak_memory_bytes: AtomicU64::new(0),
            failed_operations: AtomicUsize::new(0)}
    }

    /// Record tensor load
    #[inline(always)]
    pub fn record_tensor_load(&self, bytes: usize, duration_ns: u64) {
        self.tensors_loaded.fetch_add(1, Ordering::Relaxed);
        self.bytes_loaded.fetch_add(bytes as u64, Ordering::Relaxed);
        self.loading_time_ns.fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Record cache hit
    #[inline(always)]
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache miss
    #[inline(always)]
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record memory mapping operation
    #[inline(always)]
    pub fn record_mmap_operation(&self) {
        self.mmap_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record device transfer
    #[inline(always)]
    pub fn record_device_transfer(&self) {
        self.device_transfers.fetch_add(1, Ordering::Relaxed);
    }

    /// Record validation operation
    #[inline(always)]
    pub fn record_validation(&self) {
        self.validation_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record tensor fusion
    #[inline(always)]
    pub fn record_tensor_fusion(&self) {
        self.tensor_fusions.fetch_add(1, Ordering::Relaxed);
    }

    /// Update memory usage
    #[inline(always)]
    pub fn update_memory_usage(&self, bytes: u64) {
        self.memory_usage_bytes.store(bytes, Ordering::Relaxed);
        let current_peak = self.peak_memory_bytes.load(Ordering::Relaxed);
        if bytes > current_peak {
            self.peak_memory_bytes.store(bytes, Ordering::Relaxed);
        }
    }

    /// Record failed operation
    #[inline(always)]
    pub fn record_failure(&self) {
        self.failed_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get cache hit rate as percentage
    #[inline(always)]
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed) as f64;
        let misses = self.cache_misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total > 0.0 {
            hits / total * 100.0
        } else {
            0.0
        }
    }

    /// Get average loading speed in MB/s
    #[inline(always)]
    pub fn loading_speed_mbps(&self) -> f64 {
        let bytes = self.bytes_loaded.load(Ordering::Relaxed) as f64;
        let time_ns = self.loading_time_ns.load(Ordering::Relaxed) as f64;
        if time_ns > 0.0 {
            (bytes / (1024.0 * 1024.0)) / (time_ns / 1_000_000_000.0)
        } else {
            0.0
        }
    }

    /// Get efficiency score (0-100)
    #[inline(always)]
    pub fn efficiency_score(&self) -> f64 {
        let hit_rate = self.cache_hit_rate();
        let failures = self.failed_operations.load(Ordering::Relaxed) as f64;
        let total_ops = self.tensors_loaded.load(Ordering::Relaxed) as f64;
        
        if total_ops > 0.0 {
            let failure_rate = failures / total_ops * 100.0;
            let efficiency = hit_rate * 0.7 + (100.0 - failure_rate) * 0.3;
            efficiency.min(100.0).max(0.0)
        } else {
            0.0
        }
    }

    /// Get current time in nanoseconds since UNIX epoch
    #[inline(always)]
    pub fn current_time_nanos() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
}

impl Default for LoadingStats {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}