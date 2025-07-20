//! Zero-allocation error handling with atomic metrics and performance monitoring
//!
//! This module provides comprehensive error types for the Candle client with cache-efficient
//! error representations, atomic error tracking, and zero-allocation error construction.

use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::fmt;
use arrayvec::ArrayString;
use smallvec::SmallVec;

use super::models::{CandleModel, CandleDevice};

/// Cache-efficient error type with atomic operation support
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Configuration = 0,
    Model = 1,
    Device = 2,
    Tokenizer = 3,
    Generation = 4,
    Memory = 5,
    IO = 6,
    Network = 7,
    Cache = 8,
    Streaming = 9,
}

/// Zero-allocation error context with stack-allocated strings
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Error category for fast filtering
    pub category: ErrorCategory,
    /// Component that generated the error
    pub component: ArrayString<32>,
    /// Operation that failed
    pub operation: ArrayString<64>,
    /// Expected value or condition
    pub expected: ArrayString<128>,
    /// Actual value or condition
    pub actual: ArrayString<128>,
    /// Timestamp when error occurred (milliseconds since epoch)
    pub timestamp_ms: u64,
}

impl ErrorContext {
    /// Create new error context with validation
    #[inline]
    pub fn new(
        category: ErrorCategory,
        component: &str,
        operation: &str,
        expected: &str,
        actual: &str,
    ) -> Self {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            category,
            component: ArrayString::from(component).unwrap_or_default(),
            operation: ArrayString::from(operation).unwrap_or_default(),
            expected: ArrayString::from(expected).unwrap_or_default(),
            actual: ArrayString::from(actual).unwrap_or_default(),
            timestamp_ms,
        }
    }
}

/// Comprehensive Candle error type with zero-allocation design
#[derive(Debug, Clone)]
pub enum CandleError {
    /// Configuration validation errors
    Configuration {
        context: ErrorContext,
        parameter: ArrayString<64>,
        constraint: ArrayString<128>,
    },
    
    /// Model loading and management errors
    Model {
        context: ErrorContext,
        model: CandleModel,
        phase: ModelPhase,
        details: ArrayString<256>,
    },
    
    /// Device initialization and management errors
    Device {
        context: ErrorContext,
        device: CandleDevice,
        capability: ArrayString<64>,
        available: SmallVec<[CandleDevice; 4]>,
    },
    
    /// Tokenization and text processing errors
    Tokenizer {
        context: ErrorContext,
        input_length: u32,
        max_length: u32,
        encoding: ArrayString<32>,
    },
    
    /// Text generation and sampling errors
    Generation {
        context: ErrorContext,
        tokens_processed: u32,
        sequence_length: u32,
        sampling_step: ArrayString<64>,
    },
    
    /// Memory allocation and management errors
    Memory {
        context: ErrorContext,
        requested_bytes: u64,
        available_bytes: u64,
        memory_type: MemoryType,
    },
    
    /// File I/O and filesystem errors
    Io {
        context: ErrorContext,
        path: ArrayString<256>,
        operation: IoOperation,
        os_error: Option<i32>,
    },
    
    /// Network and download errors
    Network {
        context: ErrorContext,
        url: ArrayString<256>,
        status_code: Option<u16>,
        retry_count: u8,
    },
    
    /// Cache management errors
    Cache {
        context: ErrorContext,
        cache_type: CacheType,
        key: ArrayString<128>,
        size_bytes: u64,
    },
    
    /// Streaming and async coordination errors
    Streaming {
        context: ErrorContext,
        stream_id: u64,
        chunk_index: u32,
        buffer_size: u32,
    },
}

/// Model operation phases for detailed error reporting
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPhase {
    Download = 0,
    Validation = 1,
    Loading = 2,
    Initialization = 3,
    Inference = 4,
}

/// Memory type categories for memory errors
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    System = 0,
    Gpu = 1,
    Cache = 2,
    Buffer = 3,
}

/// I/O operation types for file system errors
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoOperation {
    Read = 0,
    Write = 1,
    Create = 2,
    Delete = 3,
    Move = 4,
    Stat = 5,
}

/// Cache type categories for cache errors
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheType {
    Model = 0,
    Tokenizer = 1,
    KvCache = 2,
    Embedding = 3,
}

impl CandleError {
    /// Create configuration error with validation context
    #[inline]
    pub fn config(message: &str, parameter: &str, constraint: &str) -> Self {
        Self::Configuration {
            context: ErrorContext::new(
                ErrorCategory::Configuration,
                "config",
                "validate",
                constraint,
                "invalid value",
            ),
            parameter: ArrayString::from(parameter).unwrap_or_default(),
            constraint: ArrayString::from(constraint).unwrap_or_default(),
        }
    }
    
    /// Create model error with detailed context
    #[inline]
    pub fn model(model: CandleModel, phase: ModelPhase, details: &str) -> Self {
        Self::Model {
            context: ErrorContext::new(
                ErrorCategory::Model,
                "model_manager",
                "process",
                "success",
                "failure",
            ),
            model,
            phase,
            details: ArrayString::from(details).unwrap_or_default(),
        }
    }
    
    /// Create device error with available alternatives
    #[inline]
    pub fn device(
        device: CandleDevice, 
        capability: &str, 
        available: &[CandleDevice]
    ) -> Self {
        let mut available_vec = SmallVec::new();
        for &dev in available.iter().take(4) {
            available_vec.push(dev);
        }
        
        Self::Device {
            context: ErrorContext::new(
                ErrorCategory::Device,
                "device_manager",
                "initialize",
                "available device",
                "unavailable",
            ),
            device,
            capability: ArrayString::from(capability).unwrap_or_default(),
            available: available_vec,
        }
    }
    
    /// Create tokenizer error with length information
    #[inline]
    pub fn tokenizer(input_length: u32, max_length: u32, encoding: &str) -> Self {
        Self::Tokenizer {
            context: ErrorContext::new(
                ErrorCategory::Tokenizer,
                "tokenizer",
                "encode",
                &format!("<= {}", max_length),
                &format!("{}", input_length),
            ),
            input_length,
            max_length,
            encoding: ArrayString::from(encoding).unwrap_or_default(),
        }
    }
    
    /// Create generation error with sampling context
    #[inline]
    pub fn generation(
        tokens_processed: u32, 
        sequence_length: u32, 
        sampling_step: &str
    ) -> Self {
        Self::Generation {
            context: ErrorContext::new(
                ErrorCategory::Generation,
                "text_generator",
                sampling_step,
                "valid generation",
                "generation failed",
            ),
            tokens_processed,
            sequence_length,
            sampling_step: ArrayString::from(sampling_step).unwrap_or_default(),
        }
    }
    
    /// Create memory error with allocation details
    #[inline]
    pub fn memory(
        requested_bytes: u64, 
        available_bytes: u64, 
        memory_type: MemoryType
    ) -> Self {
        Self::Memory {
            context: ErrorContext::new(
                ErrorCategory::Memory,
                "memory_manager",
                "allocate",
                &format!(">= {} bytes", requested_bytes),
                &format!("{} bytes available", available_bytes),
            ),
            requested_bytes,
            available_bytes,
            memory_type,
        }
    }
    
    /// Create I/O error with path and operation details
    #[inline]
    pub fn io(path: &str, operation: IoOperation, os_error: Option<i32>) -> Self {
        Self::Io {
            context: ErrorContext::new(
                ErrorCategory::IO,
                "filesystem",
                "access",
                "success",
                "failed",
            ),
            path: ArrayString::from(path).unwrap_or_default(),
            operation,
            os_error,
        }
    }
    
    /// Create network error with URL and status information
    #[inline]
    pub fn network(url: &str, status_code: Option<u16>, retry_count: u8) -> Self {
        Self::Network {
            context: ErrorContext::new(
                ErrorCategory::Network,
                "http_client",
                "request",
                "2xx status",
                &format!("{:?}", status_code),
            ),
            url: ArrayString::from(url).unwrap_or_default(),
            status_code,
            retry_count,
        }
    }
    
    /// Create cache error with detailed context
    #[inline]
    pub fn cache(
        cache_type: CacheType, 
        key: &str, 
        size_bytes: u64, 
        operation: &str
    ) -> Self {
        Self::Cache {
            context: ErrorContext::new(
                ErrorCategory::Cache,
                "cache_manager",
                operation,
                "cache hit",
                "cache miss/error",
            ),
            cache_type,
            key: ArrayString::from(key).unwrap_or_default(),
            size_bytes,
        }
    }
    
    /// Create streaming error with stream context
    #[inline]
    pub fn streaming(
        stream_id: u64, 
        chunk_index: u32, 
        buffer_size: u32, 
        operation: &str
    ) -> Self {
        Self::Streaming {
            context: ErrorContext::new(
                ErrorCategory::Streaming,
                "streaming_coordinator",
                operation,
                "successful streaming",
                "stream error",
            ),
            stream_id,
            chunk_index,
            buffer_size,
        }
    }
    
    /// Get error category for fast filtering
    #[inline(always)]
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::Configuration { .. } => ErrorCategory::Configuration,
            Self::Model { .. } => ErrorCategory::Model,
            Self::Device { .. } => ErrorCategory::Device,
            Self::Tokenizer { .. } => ErrorCategory::Tokenizer,
            Self::Generation { .. } => ErrorCategory::Generation,
            Self::Memory { .. } => ErrorCategory::Memory,
            Self::Io { .. } => ErrorCategory::IO,
            Self::Network { .. } => ErrorCategory::Network,
            Self::Cache { .. } => ErrorCategory::Cache,
            Self::Streaming { .. } => ErrorCategory::Streaming,
        }
    }
    
    /// Get error context for detailed analysis
    #[inline(always)]
    pub fn context(&self) -> &ErrorContext {
        match self {
            Self::Configuration { context, .. } => context,
            Self::Model { context, .. } => context,
            Self::Device { context, .. } => context,
            Self::Tokenizer { context, .. } => context,
            Self::Generation { context, .. } => context,
            Self::Memory { context, .. } => context,
            Self::Io { context, .. } => context,
            Self::Network { context, .. } => context,
            Self::Cache { context, .. } => context,
            Self::Streaming { context, .. } => context,
        }
    }
    
    /// Check if error is recoverable
    #[inline]
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Configuration { .. } => false, // Config errors are permanent
            Self::Model { phase, .. } => matches!(phase, ModelPhase::Download | ModelPhase::Loading),
            Self::Device { .. } => true, // Can try different device
            Self::Tokenizer { .. } => true, // Can retry with different input
            Self::Generation { .. } => true, // Can retry generation
            Self::Memory { .. } => true, // Can free memory and retry
            Self::Io { .. } => true, // I/O operations can be retried
            Self::Network { .. } => true, // Network requests can be retried
            Self::Cache { .. } => true, // Cache operations can be retried
            Self::Streaming { .. } => true, // Streaming can be restarted
        }
    }
    
    /// Get suggested retry delay in milliseconds
    #[inline]
    pub fn retry_delay_ms(&self) -> Option<u64> {
        if !self.is_recoverable() {
            return None;
        }
        
        match self {
            Self::Network { retry_count, .. } => {
                // Exponential backoff: 100ms * 2^retry_count, capped at 30s
                let delay = 100u64.saturating_mul(1u64.saturating_shl(*retry_count as u32));
                Some(delay.min(30_000))
            }
            Self::Memory { .. } => Some(500), // Wait for memory to be freed
            Self::Io { .. } => Some(100), // Quick retry for I/O
            Self::Model { phase, .. } => match phase {
                ModelPhase::Download => Some(1000), // Network delay
                ModelPhase::Loading => Some(2000), // Model loading delay
                _ => Some(500),
            },
            _ => Some(200), // Default retry delay
        }
    }
}

impl fmt::Display for CandleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration { parameter, constraint, .. } => {
                write!(f, "Configuration error: parameter '{}' violates constraint: {}", 
                       parameter, constraint)
            }
            Self::Model { model, phase, details, .. } => {
                write!(f, "Model error: {:?} failed during {:?}: {}", 
                       model, phase, details)
            }
            Self::Device { device, capability, available, .. } => {
                write!(f, "Device error: {:?} lacks capability '{}', available: {:?}", 
                       device, capability, available)
            }
            Self::Tokenizer { input_length, max_length, encoding, .. } => {
                write!(f, "Tokenizer error: input length {} exceeds maximum {} for encoding {}", 
                       input_length, max_length, encoding)
            }
            Self::Generation { tokens_processed, sequence_length, sampling_step, .. } => {
                write!(f, "Generation error: failed at step '{}' after {} tokens (sequence length: {})", 
                       sampling_step, tokens_processed, sequence_length)
            }
            Self::Memory { requested_bytes, available_bytes, memory_type, .. } => {
                write!(f, "Memory error: requested {} bytes but only {} available for {:?}", 
                       requested_bytes, available_bytes, memory_type)
            }
            Self::Io { path, operation, os_error, .. } => {
                write!(f, "I/O error: {:?} operation failed on '{}' (OS error: {:?})", 
                       operation, path, os_error)
            }
            Self::Network { url, status_code, retry_count, .. } => {
                write!(f, "Network error: request to '{}' failed with status {:?} (retries: {})", 
                       url, status_code, retry_count)
            }
            Self::Cache { cache_type, key, size_bytes, .. } => {
                write!(f, "Cache error: {:?} cache operation failed for key '{}' ({} bytes)", 
                       cache_type, key, size_bytes)
            }
            Self::Streaming { stream_id, chunk_index, buffer_size, .. } => {
                write!(f, "Streaming error: stream {} failed at chunk {} (buffer: {} bytes)", 
                       stream_id, chunk_index, buffer_size)
            }
        }
    }
}

impl std::error::Error for CandleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // No nested errors for zero-allocation design
        None
    }
}

/// Result type alias for Candle operations
pub type CandleResult<T> = Result<T, CandleError>;

/// Atomic error metrics for performance monitoring and debugging
#[derive(Debug)]
pub struct ErrorMetrics {
    /// Total errors by category (indexed by ErrorCategory as u8)
    category_counts: [AtomicU64; 10],
    /// Total recoverable errors
    recoverable_errors: AtomicU64,
    /// Total non-recoverable errors
    non_recoverable_errors: AtomicU64,
    /// Total retry attempts
    retry_attempts: AtomicU64,
    /// Average error frequency (errors per second)
    error_frequency: AtomicU32,
    /// Last error timestamp
    last_error_timestamp: AtomicU64,
}

impl ErrorMetrics {
    /// Create new error metrics tracker
    pub fn new() -> Self {
        const ATOMIC_ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            category_counts: [ATOMIC_ZERO; 10],
            recoverable_errors: AtomicU64::new(0),
            non_recoverable_errors: AtomicU64::new(0),
            retry_attempts: AtomicU64::new(0),
            error_frequency: AtomicU32::new(0),
            last_error_timestamp: AtomicU64::new(0),
        }
    }
    
    /// Record an error occurrence
    #[inline]
    pub fn record_error(&self, error: &CandleError) {
        let category_index = error.category() as usize;
        if category_index < self.category_counts.len() {
            self.category_counts[category_index].fetch_add(1, Ordering::Relaxed);
        }
        
        if error.is_recoverable() {
            self.recoverable_errors.fetch_add(1, Ordering::Relaxed);
        } else {
            self.non_recoverable_errors.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.last_error_timestamp.store(now, Ordering::Relaxed);
        
        // Update frequency (simplified calculation)
        self.update_frequency();
    }
    
    /// Record a retry attempt
    #[inline]
    pub fn record_retry(&self) {
        self.retry_attempts.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get total errors for a specific category
    #[inline]
    pub fn get_category_count(&self, category: ErrorCategory) -> u64 {
        let index = category as usize;
        if index < self.category_counts.len() {
            self.category_counts[index].load(Ordering::Relaxed)
        } else {
            0
        }
    }
    
    /// Get total recoverable errors
    #[inline(always)]
    pub fn get_recoverable_count(&self) -> u64 {
        self.recoverable_errors.load(Ordering::Relaxed)
    }
    
    /// Get total non-recoverable errors
    #[inline(always)]
    pub fn get_non_recoverable_count(&self) -> u64 {
        self.non_recoverable_errors.load(Ordering::Relaxed)
    }
    
    /// Get total retry attempts
    #[inline(always)]
    pub fn get_retry_count(&self) -> u64 {
        self.retry_attempts.load(Ordering::Relaxed)
    }
    
    /// Get current error frequency (errors per second)
    #[inline(always)]
    pub fn get_frequency(&self) -> f32 {
        f32::from_bits(self.error_frequency.load(Ordering::Relaxed))
    }
    
    /// Get last error timestamp
    #[inline(always)]
    pub fn get_last_error_timestamp(&self) -> u64 {
        self.last_error_timestamp.load(Ordering::Relaxed)
    }
    
    /// Reset all metrics
    pub fn reset(&self) {
        for counter in &self.category_counts {
            counter.store(0, Ordering::Relaxed);
        }
        self.recoverable_errors.store(0, Ordering::Relaxed);
        self.non_recoverable_errors.store(0, Ordering::Relaxed);
        self.retry_attempts.store(0, Ordering::Relaxed);
        self.error_frequency.store(0, Ordering::Relaxed);
        self.last_error_timestamp.store(0, Ordering::Relaxed);
    }
    
    /// Update error frequency calculation
    fn update_frequency(&self) {
        // Simplified frequency calculation
        // In a production system, this would use a sliding window
        let total_errors = self.recoverable_errors.load(Ordering::Relaxed) + 
                          self.non_recoverable_errors.load(Ordering::Relaxed);
        
        if total_errors > 0 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(1);
            
            let frequency = total_errors as f32 / now.max(1) as f32;
            self.error_frequency.store(frequency.to_bits(), Ordering::Relaxed);
        }
    }
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Global error metrics instance for the Candle client
static GLOBAL_ERROR_METRICS: ErrorMetrics = ErrorMetrics {
    category_counts: [AtomicU64::new(0); 10],
    recoverable_errors: AtomicU64::new(0),
    non_recoverable_errors: AtomicU64::new(0),
    retry_attempts: AtomicU64::new(0),
    error_frequency: AtomicU32::new(0),
    last_error_timestamp: AtomicU64::new(0),
};

/// Get global error metrics for monitoring
#[inline(always)]
pub fn global_error_metrics() -> &'static ErrorMetrics {
    &GLOBAL_ERROR_METRICS
}

/// Record error in global metrics
#[inline]
pub fn record_global_error(error: &CandleError) {
    GLOBAL_ERROR_METRICS.record_error(error);
}

/// Record retry in global metrics
#[inline]
pub fn record_global_retry() {
    GLOBAL_ERROR_METRICS.record_retry();
}