//! Core types and constants for VarBuilder

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use arrayvec::ArrayString;
use candle_core::{DType, Device};

/// Convert safetensors Dtype to candle DType
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
pub const MAX_TENSOR_NAME_LEN: usize = 256;

/// Maximum number of tensors for stack allocation
pub const MAX_TENSORS: usize = 2048;

/// Maximum configuration entries
pub const MAX_CONFIG_ENTRIES: usize = 128;

/// Maximum file paths for sharded models
pub const MAX_FILE_PATHS: usize = 32;

/// Cache line size for alignment
pub const CACHE_LINE_SIZE: usize = 64;

/// Static error messages (zero allocation)
pub const ERR_TENSOR_NOT_FOUND: &str = "Tensor not found";
pub const ERR_INVALID_SHAPE: &str = "Invalid tensor shape";
pub const ERR_DEVICE_MISMATCH: &str = "Device mismatch";
pub const ERR_MODEL_LOADING: &str = "Model loading failed";
pub const ERR_METADATA_PARSING: &str = "Metadata parsing failed";

/// Ultra-compact tensor name (stack allocated)
pub type TensorName = ArrayString<MAX_TENSOR_NAME_LEN>;

/// Ultra-compact configuration key
pub type ConfigKey = ArrayString<64>;

/// Ultra-compact configuration value  
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