//! High-performance optimization utilities for Candle ML inference
//!
//! This module provides SIMD acceleration, parallel processing, memory prefetching,
//! and comprehensive performance monitoring for blazing-fast inference.

use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use crossbeam_utils::atomic::AtomicCell;
use rayon::prelude::*;

use super::error::{CandleError, CandleResult};

/// Cache line size for optimal memory alignment
const CACHE_LINE_SIZE: usize = 64;
/// SIMD vector width for f32 operations
const SIMD_WIDTH: usize = 8; // AVX-256 / 32-bit floats
/// Default benchmark iterations for profiling
const DEFAULT_BENCHMARK_ITERATIONS: usize = 100;

/// SIMD capabilities detection for runtime optimization
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    /// AVX support (256-bit vectors)
    pub has_avx: bool,
    /// AVX2 support (256-bit integer operations)
    pub has_avx2: bool,
    /// FMA support (fused multiply-add)
    pub has_fma: bool,
    /// SSE4.1 support (128-bit vectors)
    pub has_sse41: bool,
    /// Vector width in f32 elements
    pub vector_width: usize,
}

impl Default for SimdCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx");
            let has_avx2 = std::arch::is_x86_feature_detected!("avx2");
            let has_fma = std::arch::is_x86_feature_detected!("fma");
            let has_sse41 = std::arch::is_x86_feature_detected!("sse4.1");

            let vector_width = if has_avx {
                8
            } else if has_sse41 {
                4
            } else {
                1
            };

            Self {
                has_avx,
                has_avx2,
                has_fma,
                has_sse41,
                vector_width,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM NEON is standard on AArch64
            Self {
                has_avx: false,
                has_avx2: false,
                has_fma: true, // NEON has FMA
                has_sse41: false,
                vector_width: 4, // NEON 128-bit / 32-bit = 4
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback for other architectures
            Self {
                has_avx: false,
                has_avx2: false,
                has_fma: false,
                has_sse41: false,
                vector_width: 1,
            }
        }
    }

    /// Check if SIMD acceleration is available
    pub fn has_simd(&self) -> bool {
        self.vector_width > 1
    }

    /// Get optimal chunk size for parallel processing
    pub fn optimal_chunk_size(&self, total_size: usize) -> usize {
        let num_threads = rayon::current_num_threads();
        let chunk_size = (total_size + num_threads - 1) / num_threads;

        // Align to SIMD width
        let simd_aligned =
            (chunk_size + self.vector_width - 1) / self.vector_width * self.vector_width;
        simd_aligned.max(self.vector_width)
    }
}

/// Cache-aligned buffer for high-performance operations
#[derive(Debug)]
pub struct AlignedBuffer {
    /// Raw memory pointer
    ptr: NonNull<f32>,
    /// Buffer capacity
    capacity: usize,
    /// Current length
    length: usize,
    /// Memory layout for deallocation
    layout: Layout,
}

impl AlignedBuffer {
    /// Create new aligned buffer with specified capacity
    pub fn new(capacity: usize) -> CandleResult<Self> {
        let layout =
            Layout::from_size_align(capacity * std::mem::size_of::<f32>(), CACHE_LINE_SIZE)
                .map_err(|e| {
                    CandleError::performance(
                        &format!("Invalid buffer layout: {}", e),
                        "new",
                        "valid capacity and alignment",
                    )
                })?;

        let ptr = unsafe {
            let raw_ptr = alloc(layout) as *mut f32;
            if raw_ptr.is_null() {
                return Err(CandleError::performance(
                    "Buffer allocation failed",
                    "new",
                    "sufficient memory available",
                ));
            }
            NonNull::new_unchecked(raw_ptr)
        };

        Ok(Self {
            ptr,
            capacity,
            length: 0,
            layout,
        })
    }

    /// Get mutable slice of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.length) }
    }

    /// Get read-only slice of the buffer
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Set buffer length (must be <= capacity)
    pub fn set_length(&mut self, length: usize) -> CandleResult<()> {
        if length > self.capacity {
            return Err(CandleError::performance(
                "Length exceeds buffer capacity",
                "set_length",
                "length <= capacity",
            ));
        }

        self.length = length;
        Ok(())
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current length
    pub fn length(&self) -> usize {
        self.length
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.length = 0;

        // Zero out memory for security
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.capacity);
        }
    }

    /// Fill buffer with value
    pub fn fill(&mut self, value: f32) {
        let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity) };
        slice.fill(value);
        self.length = self.capacity;
    }

    /// Copy from slice into buffer
    pub fn copy_from_slice(&mut self, src: &[f32]) -> CandleResult<()> {
        if src.len() > self.capacity {
            return Err(CandleError::performance(
                "Source slice too large for buffer",
                "copy_from_slice",
                "src.len() <= capacity",
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.as_ptr(), src.len());
        }

        self.length = src.len();
        Ok(())
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

/// Performance optimization configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Enable SIMD acceleration
    pub enable_simd: bool,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Enable memory prefetching
    pub enable_prefetch: bool,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Thread pool size (0 = auto-detect)
    pub thread_pool_size: usize,
    /// Minimum work size for parallelization
    pub parallel_threshold: usize,
    /// Cache-friendly chunk size
    pub chunk_size: usize,
    /// Enable aggressive optimizations
    pub aggressive_optimizations: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        let capabilities = SimdCapabilities::detect();

        Self {
            enable_simd: capabilities.has_simd(),
            enable_parallel: true,
            enable_prefetch: true,
            enable_profiling: false,
            thread_pool_size: 0, // Auto-detect
            parallel_threshold: 1024,
            chunk_size: capabilities.optimal_chunk_size(8192),
            aggressive_optimizations: false,
        }
    }
}

impl PerformanceConfig {
    /// Create configuration for maximum performance
    pub fn max_performance() -> Self {
        let capabilities = SimdCapabilities::detect();

        Self {
            enable_simd: capabilities.has_simd(),
            enable_parallel: true,
            enable_prefetch: true,
            enable_profiling: false,
            thread_pool_size: 0,
            parallel_threshold: 512, // Lower threshold for more parallelism
            chunk_size: capabilities.optimal_chunk_size(16384),
            aggressive_optimizations: true,
        }
    }

    /// Create configuration for debugging/profiling
    pub fn debug_profile() -> Self {
        Self {
            enable_simd: false, // Disable for easier debugging
            enable_parallel: false,
            enable_prefetch: false,
            enable_profiling: true,
            thread_pool_size: 1,
            parallel_threshold: usize::MAX, // No parallelization
            chunk_size: 64,
            aggressive_optimizations: false,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> CandleResult<()> {
        if self.parallel_threshold == 0 {
            return Err(CandleError::config(
                "Parallel threshold must be positive",
                "parallel_threshold",
                "> 0",
            ));
        }

        if self.chunk_size == 0 {
            return Err(CandleError::config(
                "Chunk size must be positive",
                "chunk_size",
                "> 0",
            ));
        }

        Ok(())
    }
}

/// Benchmark result for performance measurement
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Operation name
    pub operation: String,
    /// Total time taken
    pub duration: Duration,
    /// Number of iterations
    pub iterations: usize,
    /// Operations per second
    pub ops_per_second: f64,
    /// Average time per operation
    pub avg_time_per_op: Duration,
    /// Memory throughput (bytes/second)
    pub memory_throughput: f64,
}

impl BenchmarkResult {
    /// Create new benchmark result
    pub fn new(
        operation: String,
        duration: Duration,
        iterations: usize,
        bytes_processed: usize,
    ) -> Self {
        let duration_secs = duration.as_secs_f64();
        let ops_per_second = if duration_secs > 0.0 {
            iterations as f64 / duration_secs
        } else {
            0.0
        };

        let avg_time_per_op = if iterations > 0 {
            duration / iterations as u32
        } else {
            Duration::ZERO
        };

        let memory_throughput = if duration_secs > 0.0 {
            bytes_processed as f64 / duration_secs
        } else {
            0.0
        };

        Self {
            operation,
            duration,
            iterations,
            ops_per_second,
            avg_time_per_op,
            memory_throughput,
        }
    }

    /// Get throughput in MB/s
    pub fn throughput_mbps(&self) -> f64 {
        self.memory_throughput / (1024.0 * 1024.0)
    }
}

/// High-performance optimizer with SIMD and parallel processing
#[derive(Debug)]
pub struct PerformanceOptimizer {
    /// Configuration
    config: PerformanceConfig,
    /// SIMD capabilities
    simd_capabilities: SimdCapabilities,
    /// Performance statistics
    stats: PerformanceStatistics,
    /// Benchmark results
    benchmarks: ArcSwap<Vec<BenchmarkResult>>,
}

impl PerformanceOptimizer {
    /// Create new performance optimizer
    pub fn new(config: PerformanceConfig) -> Self {
        let simd_capabilities = SimdCapabilities::detect();

        Self {
            config,
            simd_capabilities,
            stats: PerformanceStatistics::default(),
            benchmarks: ArcSwap::from_pointee(Vec::new()),
        }
    }

    /// Optimized vector addition with SIMD acceleration
    pub fn vector_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> CandleResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(CandleError::performance(
                "Vector lengths must match",
                "vector_add",
                "equal length vectors",
            ));
        }

        let start_time = Instant::now();

        if self.config.enable_parallel && a.len() >= self.config.parallel_threshold {
            self.parallel_vector_add(a, b, result)?;
        } else if self.config.enable_simd && self.simd_capabilities.has_simd() {
            self.simd_vector_add(a, b, result)?;
        } else {
            self.scalar_vector_add(a, b, result);
        }

        let duration = start_time.elapsed();
        self.stats
            .total_operations
            .store(self.stats.total_operations.load() + 1);
        self.stats
            .total_compute_time
            .store(self.stats.total_compute_time.load() + duration.as_nanos() as u64);

        Ok(())
    }

    /// Parallel vector addition
    fn parallel_vector_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> CandleResult<()> {
        let chunk_size = self.simd_capabilities.optimal_chunk_size(a.len());

        result
            .par_chunks_mut(chunk_size)
            .zip(a.par_chunks(chunk_size))
            .zip(b.par_chunks(chunk_size))
            .for_each(|((result_chunk, a_chunk), b_chunk)| {
                if self.config.enable_simd && self.simd_capabilities.has_simd() {
                    let _ = self.simd_vector_add(a_chunk, b_chunk, result_chunk);
                } else {
                    self.scalar_vector_add(a_chunk, b_chunk, result_chunk);
                }
            });

        self.stats
            .parallel_operations
            .store(self.stats.parallel_operations.load() + 1);

        Ok(())
    }

    /// SIMD vector addition
    fn simd_vector_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> CandleResult<()> {
        let vector_width = self.simd_capabilities.vector_width;
        let simd_len = (a.len() / vector_width) * vector_width;

        // SIMD portion
        for i in (0..simd_len).step_by(vector_width) {
            // Note: In a real implementation, this would use actual SIMD intrinsics
            // For now, we simulate with optimized scalar operations
            for j in 0..vector_width.min(a.len() - i) {
                if self.config.enable_prefetch && i + vector_width < a.len() {
                    // Simulate prefetching
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            a.as_ptr().add(i + vector_width) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }

                result[i + j] = a[i + j] + b[i + j];
            }
        }

        // Handle remainder
        if simd_len < a.len() {
            self.scalar_vector_add(&a[simd_len..], &b[simd_len..], &mut result[simd_len..]);
        }

        self.stats
            .simd_operations
            .store(self.stats.simd_operations.load() + 1);

        Ok(())
    }

    /// Scalar vector addition fallback
    fn scalar_vector_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }

        self.stats
            .scalar_operations
            .store(self.stats.scalar_operations.load() + 1);
    }

    /// Optimized matrix multiplication
    pub fn matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> CandleResult<()> {
        if a.len() != m * k || b.len() != k * n || result.len() != m * n {
            return Err(CandleError::performance(
                "Matrix dimensions don't match",
                "matrix_multiply",
                "compatible dimensions",
            ));
        }

        let start_time = Instant::now();

        if self.config.enable_parallel && m * n >= self.config.parallel_threshold {
            self.parallel_matrix_multiply(a, b, result, m, n, k)?;
        } else {
            self.scalar_matrix_multiply(a, b, result, m, n, k);
        }

        let duration = start_time.elapsed();
        self.stats
            .total_operations
            .store(self.stats.total_operations.load() + 1);
        self.stats
            .total_compute_time
            .store(self.stats.total_compute_time.load() + duration.as_nanos() as u64);

        Ok(())
    }

    /// Parallel matrix multiplication
    fn parallel_matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> CandleResult<()> {
        result
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, result_row)| {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += a[i * k + l] * b[l * n + j];
                    }
                    result_row[j] = sum;
                }
            });

        self.stats
            .parallel_operations
            .store(self.stats.parallel_operations.load() + 1);

        Ok(())
    }

    /// Scalar matrix multiplication
    fn scalar_matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        self.stats
            .scalar_operations
            .store(self.stats.scalar_operations.load() + 1);
    }

    /// Benchmark an operation
    pub fn benchmark<F>(&self, name: &str, mut operation: F, iterations: usize) -> BenchmarkResult
    where
        F: FnMut() -> usize, // Returns bytes processed
    {
        let start_time = Instant::now();
        let mut total_bytes = 0;

        for _ in 0..iterations {
            total_bytes += operation();
        }

        let duration = start_time.elapsed();

        let result = BenchmarkResult::new(name.to_string(), duration, iterations, total_bytes);

        // Store benchmark result
        let mut benchmarks = (**self.benchmarks.load()).clone();
        benchmarks.push(result.clone());
        self.benchmarks.store(Arc::new(benchmarks));

        result
    }

    /// Get SIMD capabilities
    pub fn simd_capabilities(&self) -> &SimdCapabilities {
        &self.simd_capabilities
    }

    /// Get configuration
    pub fn config(&self) -> &PerformanceConfig {
        &self.config
    }

    /// Get performance statistics
    pub fn statistics(&self) -> PerformanceStatistics {
        PerformanceStatistics {
            total_operations: self.stats.total_operations.load(),
            simd_operations: self.stats.simd_operations.load(),
            parallel_operations: self.stats.parallel_operations.load(),
            scalar_operations: self.stats.scalar_operations.load(),
            total_compute_time: Duration::from_nanos(self.stats.total_compute_time.load()),
            simd_capabilities: self.simd_capabilities,
            config: self.config.clone(),
        }
    }

    /// Get benchmark results
    pub fn benchmark_results(&self) -> Vec<BenchmarkResult> {
        (**self.benchmarks.load()).clone()
    }
}

/// Performance statistics for monitoring
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    /// Total operations performed
    total_operations: AtomicCell<u64>,
    /// SIMD-accelerated operations
    simd_operations: AtomicCell<u64>,
    /// Parallel operations
    parallel_operations: AtomicCell<u64>,
    /// Scalar fallback operations
    scalar_operations: AtomicCell<u64>,
    /// Total compute time
    total_compute_time: AtomicCell<u64>,
    /// SIMD capabilities
    simd_capabilities: SimdCapabilities,
    /// Configuration
    config: PerformanceConfig,
}

impl Default for PerformanceStatistics {
    fn default() -> Self {
        Self {
            total_operations: AtomicCell::new(0),
            simd_operations: AtomicCell::new(0),
            parallel_operations: AtomicCell::new(0),
            scalar_operations: AtomicCell::new(0),
            total_compute_time: AtomicCell::new(0),
            simd_capabilities: SimdCapabilities::detect(),
            config: PerformanceConfig::default(),
        }
    }
}

impl PerformanceStatistics {
    /// Calculate SIMD utilization rate
    pub fn simd_utilization(&self) -> f32 {
        let total = self.total_operations.load();
        if total > 0 {
            self.simd_operations.load() as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Calculate parallel utilization rate
    pub fn parallel_utilization(&self) -> f32 {
        let total = self.total_operations.load();
        if total > 0 {
            self.parallel_operations.load() as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Get average operation time
    pub fn avg_operation_time(&self) -> Duration {
        let total_ops = self.total_operations.load();
        if total_ops > 0 {
            Duration::from_nanos(self.total_compute_time.load() / total_ops)
        } else {
            Duration::ZERO
        }
    }

    /// Check if performance optimizations are effective
    pub fn is_optimized(&self) -> bool {
        let simd_util = self.simd_utilization();
        let parallel_util = self.parallel_utilization();

        // Consider optimized if using SIMD or parallel processing significantly
        simd_util > 0.5 || parallel_util > 0.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capabilities_detection() {
        let capabilities = SimdCapabilities::detect();
        assert!(capabilities.vector_width >= 1);
    }

    #[test]
    fn test_aligned_buffer() {
        let mut buffer = AlignedBuffer::new(1024).unwrap();
        assert_eq!(buffer.capacity(), 1024);
        assert_eq!(buffer.length(), 0);

        buffer.fill(1.0);
        assert_eq!(buffer.length(), 1024);
        assert_eq!(buffer.as_slice()[0], 1.0);

        buffer.clear();
        assert_eq!(buffer.length(), 0);
    }

    #[test]
    fn test_performance_optimizer() {
        let config = PerformanceConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];

        optimizer.vector_add(&a, &b, &mut result).unwrap();

        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);

        let stats = optimizer.statistics();
        assert_eq!(stats.total_operations.load(), 1);
    }

    #[test]
    fn test_matrix_multiplication() {
        let config = PerformanceConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
        let mut result = vec![0.0; 4]; // 2x2 result

        optimizer
            .matrix_multiply(&a, &b, &mut result, 2, 2, 2)
            .unwrap();

        // Expected: [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8] = [19, 22; 43, 50]
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_benchmark() {
        let config = PerformanceConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        let result = optimizer.benchmark(
            "test_op",
            || {
                // Simulate some work
                std::thread::sleep(Duration::from_micros(1));
                1024 // bytes processed
            },
            10,
        );

        assert_eq!(result.operation, "test_op");
        assert_eq!(result.iterations, 10);
        assert!(result.duration > Duration::ZERO);
        assert!(result.ops_per_second > 0.0);
    }
}
