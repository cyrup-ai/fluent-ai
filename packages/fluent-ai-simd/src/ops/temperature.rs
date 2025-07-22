//! Temperature scaling operations with SIMD optimization
//!
//! Provides blazing-fast temperature scaling for logits processing with runtime
//! CPU feature detection and optimal SIMD utilization.

use once_cell::sync::Lazy;

use crate::error::{SimdError, SimdResult};
use crate::runtime::{get_cpu_features, CpuFeatures, TemperatureDispatch};

/// Global temperature scaling dispatch table with runtime optimization
static TEMPERATURE_DISPATCH: Lazy<TemperatureDispatch> = Lazy::new(|| TemperatureDispatch {
    avx2: Some(apply_temperature_avx2),
    sse41: Some(apply_temperature_sse41),
    neon: Some(apply_temperature_neon),
    scalar: apply_temperature_scalar,
});

/// Apply temperature scaling to logits with optimal SIMD acceleration
///
/// Scales logits by dividing by temperature value. Uses runtime CPU feature
/// detection to select the fastest available implementation.
///
/// # Arguments
/// * `logits` - Mutable slice of logits to scale
/// * `temperature` - Temperature value (must be > 0.0)
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(SimdError)` if temperature is invalid
///
/// # Performance
/// - AVX2: ~8x speedup (8 f32 elements per cycle)
/// - NEON: ~4x speedup (4 f32 elements per cycle)  
/// - SSE4.1: ~4x speedup (4 f32 elements per cycle)
/// - Scalar: baseline performance
#[inline]
pub fn apply_temperature_scaling(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    if temperature <= 0.0 {
        return Err(SimdError::InvalidInput(
            "Temperature must be positive".to_string(),
        ));
    }

    TEMPERATURE_DISPATCH.get_fn()(logits, temperature)
}

/// In-place temperature scaling (alias for apply_temperature_scaling)
#[inline(always)]
pub fn apply_temperature_scaling_inplace(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    apply_temperature_scaling(logits, temperature)
}

/// AVX2-optimized temperature scaling
///
/// Processes 8 f32 elements per iteration using 256-bit AVX2 instructions
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn apply_temperature_avx2(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    #[cfg(target_feature = "avx2")]
    {
        unsafe { apply_temperature_avx2_impl(logits, temperature) }
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { apply_temperature_avx2_impl(logits, temperature) }
        } else {
            apply_temperature_scalar(logits, temperature)
        }
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn apply_temperature_avx2(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    apply_temperature_scalar(logits, temperature)
}

/// AVX2 implementation with manual inlining for maximum performance
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn apply_temperature_avx2_impl(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    use std::arch::x86_64::*;

    let inv_temp = 1.0 / temperature;
    let inv_temp_vec = _mm256_set1_ps(inv_temp);

    let mut i = 0;
    let len = logits.len();

    // Process 8 elements at a time (256-bit AVX2)
    while i + 8 <= len {
        let logits_vec = _mm256_loadu_ps(logits.as_ptr().add(i));
        let scaled = _mm256_mul_ps(logits_vec, inv_temp_vec);
        _mm256_storeu_ps(logits.as_mut_ptr().add(i), scaled);
        i += 8;
    }

    // Handle remaining elements with scalar operations
    while i < len {
        *logits.get_unchecked_mut(i) *= inv_temp;
        i += 1;
    }

    Ok(())
}

/// SSE4.1-optimized temperature scaling  
///
/// Processes 4 f32 elements per iteration using 128-bit SSE instructions
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn apply_temperature_sse41(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    #[cfg(target_feature = "sse4.1")]
    {
        unsafe { apply_temperature_sse41_impl(logits, temperature) }
    }

    #[cfg(not(target_feature = "sse4.1"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            unsafe { apply_temperature_sse41_impl(logits, temperature) }
        } else {
            apply_temperature_scalar(logits, temperature)
        }
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn apply_temperature_sse41(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    apply_temperature_scalar(logits, temperature)
}

/// SSE4.1 implementation
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn apply_temperature_sse41_impl(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    use std::arch::x86_64::*;

    let inv_temp = 1.0 / temperature;
    let inv_temp_vec = _mm_set1_ps(inv_temp);

    let mut i = 0;
    let len = logits.len();

    // Process 4 elements at a time (128-bit SSE)
    while i + 4 <= len {
        let logits_vec = _mm_loadu_ps(logits.as_ptr().add(i));
        let scaled = _mm_mul_ps(logits_vec, inv_temp_vec);
        _mm_storeu_ps(logits.as_mut_ptr().add(i), scaled);
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        *logits.get_unchecked_mut(i) *= inv_temp;
        i += 1;
    }

    Ok(())
}

/// NEON-optimized temperature scaling for ARM64
///
/// Processes 4 f32 elements per iteration using 128-bit NEON instructions
#[cfg(target_arch = "aarch64")]
fn apply_temperature_neon(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    #[cfg(target_feature = "neon")]
    {
        unsafe { apply_temperature_neon_impl(logits, temperature) }
    }

    #[cfg(not(target_feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { apply_temperature_neon_impl(logits, temperature) }
        } else {
            apply_temperature_scalar(logits, temperature)
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn apply_temperature_neon(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    apply_temperature_scalar(logits, temperature)
}

/// NEON implementation  
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_temperature_neon_impl(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    use std::arch::aarch64::*;

    let inv_temp = 1.0 / temperature;
    let inv_temp_vec = vdupq_n_f32(inv_temp);

    let mut i = 0;
    let len = logits.len();

    // Process 4 elements at a time (128-bit NEON)
    while i + 4 <= len {
        let logits_vec = vld1q_f32(logits.as_ptr().add(i));
        let scaled = vmulq_f32(logits_vec, inv_temp_vec);
        vst1q_f32(logits.as_mut_ptr().add(i), scaled);
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        *logits.get_unchecked_mut(i) *= inv_temp;
        i += 1;
    }

    Ok(())
}

/// Scalar temperature scaling fallback
///
/// Provides baseline performance for all platforms
fn apply_temperature_scalar(logits: &mut [f32], temperature: f32) -> SimdResult<()> {
    let inv_temp = 1.0 / temperature;

    // Use iterator for potential auto-vectorization
    for logit in logits.iter_mut() {
        *logit *= inv_temp;
    }

    Ok(())
}

/// High-level temperature processor with statistics and memory management
pub struct TemperatureProcessor {
    stats_temperature_applications: std::sync::atomic::AtomicU64,
    stats_total_elements_processed: std::sync::atomic::AtomicU64,
    stats_simd_operations: std::sync::atomic::AtomicU64,
}

impl TemperatureProcessor {
    /// Create new temperature processor
    pub fn new() -> Self {
        Self {
            stats_temperature_applications: std::sync::atomic::AtomicU64::new(0),
            stats_total_elements_processed: std::sync::atomic::AtomicU64::new(0),
            stats_simd_operations: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Apply temperature scaling with statistics tracking
    pub fn apply_scaling(&self, logits: &mut [f32], temperature: f32) -> SimdResult<()> {
        let features = get_cpu_features();
        let is_simd = features.has_simd() && logits.len() >= features.vector_width();

        let result = apply_temperature_scaling(logits, temperature);

        if result.is_ok() {
            self.stats_temperature_applications
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.stats_total_elements_processed
                .fetch_add(logits.len() as u64, std::sync::atomic::Ordering::Relaxed);

            if is_simd {
                self.stats_simd_operations
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        result
    }

    /// Get temperature scaling statistics
    pub fn get_stats(&self) -> TemperatureStats {
        TemperatureStats {
            applications: self
                .stats_temperature_applications
                .load(std::sync::atomic::Ordering::Relaxed),
            total_elements: self
                .stats_total_elements_processed
                .load(std::sync::atomic::Ordering::Relaxed),
            simd_operations: self
                .stats_simd_operations
                .load(std::sync::atomic::Ordering::Relaxed),
            cpu_features: get_cpu_features(),
        }
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.stats_temperature_applications
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.stats_total_elements_processed
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.stats_simd_operations
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for TemperatureProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for temperature scaling operations
#[derive(Debug, Clone)]
pub struct TemperatureStats {
    /// Number of temperature scaling applications
    pub applications: u64,
    /// Total elements processed
    pub total_elements: u64,
    /// Number of SIMD operations performed
    pub simd_operations: u64,
    /// Current CPU features in use
    pub cpu_features: CpuFeatures,
}

impl TemperatureStats {
    /// Calculate SIMD utilization percentage
    pub fn simd_utilization(&self) -> f64 {
        if self.applications == 0 {
            0.0
        } else {
            (self.simd_operations as f64 / self.applications as f64) * 100.0
        }
    }

    /// Calculate average elements per operation
    pub fn avg_elements_per_operation(&self) -> f64 {
        if self.applications == 0 {
            0.0
        } else {
            self.total_elements as f64 / self.applications as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_temperature_scaling_basic() {
        let mut logits = [1.0, 2.0, 3.0, 4.0];
        let temperature = 2.0;

        apply_temperature_scaling(&mut logits, temperature).unwrap();

        assert_float_eq!(logits[0], 0.5, abs <= 1e-6);
        assert_float_eq!(logits[1], 1.0, abs <= 1e-6);
        assert_float_eq!(logits[2], 1.5, abs <= 1e-6);
        assert_float_eq!(logits[3], 2.0, abs <= 1e-6);
    }

    #[test]
    fn test_temperature_scaling_invalid() {
        let mut logits = [1.0, 2.0, 3.0];

        let result = apply_temperature_scaling(&mut logits, 0.0);
        assert!(matches!(result, Err(SimdError::InvalidInput(_))));

        let result = apply_temperature_scaling(&mut logits, -1.0);
        assert!(matches!(result, Err(SimdError::InvalidInput(_))));
    }

    #[test]
    fn test_temperature_scaling_large() {
        let mut logits = vec![1.0; 1024];
        let temperature = 0.5;

        apply_temperature_scaling(&mut logits, temperature).unwrap();

        for &logit in &logits {
            assert_float_eq!(logit, 2.0, abs <= 1e-6);
        }
    }

    #[test]
    fn test_temperature_processor() {
        let processor = TemperatureProcessor::new();
        let mut logits = [1.0, 2.0, 3.0, 4.0];

        processor.apply_scaling(&mut logits, 2.0).unwrap();

        let stats = processor.get_stats();
        assert_eq!(stats.applications, 1);
        assert_eq!(stats.total_elements, 4);
    }

    #[test]
    fn test_temperature_stats() {
        let stats = TemperatureStats {
            applications: 10,
            total_elements: 1000,
            simd_operations: 8,
            cpu_features: CpuFeatures::Scalar,
        };

        assert_float_eq!(stats.simd_utilization(), 80.0, abs <= 1e-6);
        assert_float_eq!(stats.avg_elements_per_operation(), 100.0, abs <= 1e-6);
    }

    #[test]
    fn test_scalar_implementation() {
        let mut logits = [1.0, 2.0, 3.0, 4.0];
        apply_temperature_scalar(&mut logits, 2.0).unwrap();

        assert_float_eq!(logits[0], 0.5, abs <= 1e-6);
        assert_float_eq!(logits[1], 1.0, abs <= 1e-6);
        assert_float_eq!(logits[2], 1.5, abs <= 1e-6);
        assert_float_eq!(logits[3], 2.0, abs <= 1e-6);
    }
}
