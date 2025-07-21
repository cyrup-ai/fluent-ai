//! Utility functions for SIMD operations

use thiserror::Error;

use crate::SimdError;

/// Check if SIMD operations are available on the current platform
pub fn simd_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("sse4.1")
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// Find maximum value using SIMD acceleration
pub fn simd_find_max(values: &[f32]) -> Result<f32, SimdError> {
    if values.is_empty() {
        return Err(SimdError::InvalidInputLength {
            expected: 1,
            actual: 0,
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_find_max_avx2(values) };
        } else if is_x86_feature_detected!("sse4.1") {
            return unsafe { simd_find_max_sse(values) };
        }
    }

    // Fallback to scalar implementation
    values
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or_else(|| SimdError::NumericalError("Failed to find maximum".to_string()))
}

/// SIMD-accelerated exponential function
#[inline(always)]
pub fn simd_exp(x: f32) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_exp_avx2(x) };
        } else if is_x86_feature_detected!("sse4.1") {
            return unsafe { simd_exp_sse(x) };
        }
    }

    // Fallback to scalar implementation
    x.exp()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn simd_find_max_avx2(values: &[f32]) -> Result<f32, SimdError> {
    use std::arch::x86_64::*;
    
    let mut max = std::f32::NEG_INFINITY;
    let mut i = 0;
    
    // Process 8 elements at a time with AVX2
    let chunk_size = 8;
    while i + chunk_size <= values.len() {
        let chunk = _mm256_loadu_ps(values.as_ptr().add(i));
        let max_chunk = _mm256_max_ps(chunk, _mm256_permute2f128_ps(chunk, chunk, 0x01));
        let max_chunk = _mm256_max_ps(max_chunk, _mm256_shuffle_ps(max_chunk, max_chunk, 0x4E));
        let max_chunk = _mm256_max_ps(max_chunk, _mm256_shuffle_ps(max_chunk, max_chunk, 0x11));
        let max_val = _mm_cvtss_f32(_mm256_castps256_ps128(max_chunk));
        
        max = max.max(max_val);
        i += chunk_size;
    }
    
    // Process remaining elements
    while i < values.len() {
        max = max.max(values[i]);
        i += 1;
    }
    
    Ok(max)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_find_max_sse(values: &[f32]) -> Result<f32, SimdError> {
    use std::arch::x86_64::*;
    
    let mut max = std::f32::NEG_INFINITY;
    let mut i = 0;
    
    // Process 4 elements at a time with SSE
    let chunk_size = 4;
    while i + chunk_size <= values.len() {
        let chunk = _mm_loadu_ps(values.as_ptr().add(i));
        let max_chunk = _mm_max_ps(chunk, _mm_shuffle_ps(chunk, chunk, 0x4E));
        let max_chunk = _mm_max_ps(max_chunk, _mm_shuffle_ps(max_chunk, max_chunk, 0x11));
        let max_val = _mm_cvtss_f32(max_chunk);
        
        max = max.max(max_val);
        i += chunk_size;
    }
    
    // Process remaining elements
    while i < values.len() {
        max = max.max(values[i]);
        i += 1;
    }
    
    Ok(max)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn simd_exp_avx2(x: f32) -> f32 {
    use std::arch::x86_64::*;
    
    // Approximate exp using AVX2 intrinsics
    let x = _mm256_set1_ps(x);
    let a = _mm256_set1_ps(12102203.0);
    let b = _mm256_set1_ps(1065353216.0);
    let c = _mm256_set1_ps(1.0);
    
    let y = _mm256_mul_ps(x, a);
    let y = _mm256_add_ps(y, b);
    let y = _mm256_castsi256_ps(_mm256_cvttps_epi32(y));
    let y = _mm256_mul_ps(y, c);
    
    _mm256_cvtss_f32(y)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_exp_sse(x: f32) -> f32 {
    use std::arch::x86_64::*;
    
    // Approximate exp using SSE4.1 intrinsics
    let x = _mm_set1_ps(x);
    let a = _mm_set1_ps(12102203.0);
    let b = _mm_set1_ps(1065353216.0);
    let c = _mm_set1_ps(1.0);
    
    let y = _mm_mul_ps(x, a);
    let y = _mm_add_ps(y, b);
    let y = _mm_castsi128_ps(_mm_cvttps_epi32(y));
    let y = _mm_mul_ps(y, c);
    
    _mm_cvtss_f32(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;
    
    #[test]
    fn test_simd_available() {
        // Just verify it doesn't panic
        let _ = simd_available();
    }
    
    #[test]
    fn test_simd_find_max() {
        let values = [1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0];
        assert_float_eq!(simd_find_max(&values).unwrap(), 8.0, abs <= 1e-6);
    }
    
    #[test]
    fn test_simd_exp() {
        let x = 1.0;
        let expected = x.exp();
        let actual = simd_exp(x);
        assert_float_eq!(actual, expected, rel <= 1e-3);
    }
}
