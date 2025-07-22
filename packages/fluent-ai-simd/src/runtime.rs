//! Runtime CPU feature detection and SIMD dispatch system
//!
//! Provides zero-cost runtime dispatch to optimal SIMD implementations based on
//! detected CPU capabilities. Supports AVX2, NEON, and scalar fallbacks.

use std::sync::atomic::{AtomicU8, Ordering};

/// CPU capability flags for runtime dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CpuFeatures {
    /// Scalar operations only
    Scalar = 0,
    /// ARM NEON SIMD support
    Neon = 1,
    /// x86 SSE4.1 support  
    Sse41 = 2,
    /// x86 AVX2 support (highest performance)
    Avx2 = 3,
}

impl CpuFeatures {
    /// Check if this feature level supports SIMD operations
    #[inline(always)]
    pub const fn has_simd(self) -> bool {
        matches!(self, Self::Neon | Self::Sse41 | Self::Avx2)
    }

    /// Get SIMD vector width in f32 elements
    #[inline(always)]
    pub const fn vector_width(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Neon | Self::Sse41 => 4,
            Self::Avx2 => 8,
        }
    }

    /// Get optimal chunk size for processing
    #[inline(always)]
    pub const fn chunk_size(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Neon | Self::Sse41 => 16, // 4 vectors of 4 elements
            Self::Avx2 => 32,               // 4 vectors of 8 elements
        }
    }
}

/// Cached CPU feature detection result
static CPU_FEATURES: AtomicU8 = AtomicU8::new(0xFF); // 0xFF = uninitialized

/// Runtime CPU feature detection with caching
#[inline]
pub fn get_cpu_features() -> CpuFeatures {
    let cached = CPU_FEATURES.load(Ordering::Relaxed);
    if cached != 0xFF {
        // SAFETY: We only store valid CpuFeatures values
        return unsafe { std::mem::transmute(cached) };
    }

    let features = detect_cpu_features();
    CPU_FEATURES.store(features as u8, Ordering::Relaxed);
    features
}

/// Detect available CPU features at runtime
#[cold]
fn detect_cpu_features() -> CpuFeatures {
    // Priority order: AVX2 > SSE4.1 > NEON > Scalar

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return CpuFeatures::Avx2;
        }
        if is_x86_feature_detected!("sse4.1") {
            return CpuFeatures::Sse41;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return CpuFeatures::Neon;
        }
    }

    CpuFeatures::Scalar
}

/// Function pointer type for temperature scaling operations
pub type TemperatureScaleFn = fn(&mut [f32], f32) -> crate::error::SimdResult<()>;

/// Function pointer type for softmax operations  
pub type SoftmaxFn = fn(&[f32]) -> crate::error::SimdResult<Vec<f32>>;

/// Runtime dispatch table for temperature scaling
pub struct TemperatureDispatch {
    /// AVX2 optimized temperature scaling function
    pub avx2: Option<TemperatureScaleFn>,
    /// SSE4.1 optimized temperature scaling function
    pub sse41: Option<TemperatureScaleFn>,
    /// ARM NEON optimized temperature scaling function
    pub neon: Option<TemperatureScaleFn>,
    /// Scalar fallback temperature scaling function
    pub scalar: TemperatureScaleFn,
}

/// Runtime dispatch table for softmax operations
pub struct SoftmaxDispatch {
    /// AVX2 optimized softmax function
    pub avx2: Option<SoftmaxFn>,
    /// SSE4.1 optimized softmax function
    pub sse41: Option<SoftmaxFn>,
    /// ARM NEON optimized softmax function
    pub neon: Option<SoftmaxFn>,
    /// Scalar fallback softmax function
    pub scalar: SoftmaxFn,
}

impl TemperatureDispatch {
    /// Get optimal function for current CPU
    #[inline]
    pub fn get_fn(&self) -> TemperatureScaleFn {
        match get_cpu_features() {
            CpuFeatures::Avx2 => self.avx2.unwrap_or(self.scalar),
            CpuFeatures::Sse41 => self.sse41.unwrap_or(self.scalar),
            CpuFeatures::Neon => self.neon.unwrap_or(self.scalar),
            CpuFeatures::Scalar => self.scalar,
        }
    }
}

impl SoftmaxDispatch {
    /// Get optimal function for current CPU
    #[inline]
    pub fn get_fn(&self) -> SoftmaxFn {
        match get_cpu_features() {
            CpuFeatures::Avx2 => self.avx2.unwrap_or(self.scalar),
            CpuFeatures::Sse41 => self.sse41.unwrap_or(self.scalar),
            CpuFeatures::Neon => self.neon.unwrap_or(self.scalar),
            CpuFeatures::Scalar => self.scalar,
        }
    }
}

/// Check if SIMD operations are available and beneficial for given size
#[inline(always)]
pub fn should_use_simd(size: usize) -> bool {
    let features = get_cpu_features();
    features.has_simd() && size >= features.vector_width() * 2
}

/// Get optimal chunk size for bulk operations
#[inline(always)]
pub fn get_optimal_chunk_size() -> usize {
    get_cpu_features().chunk_size()
}

/// CPU feature information for debugging and optimization
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Detected CPU feature capabilities
    pub features: CpuFeatures,
    /// SIMD vector width in f32 elements
    pub vector_width: usize,
    /// Optimal processing chunk size
    pub chunk_size: usize,
    /// Whether SIMD operations are available
    pub has_simd: bool,
    /// Target architecture string
    pub arch: &'static str,
}

/// Get comprehensive CPU information
pub fn get_cpu_info() -> CpuInfo {
    let features = get_cpu_features();

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "x86") {
        "x86"
    } else {
        "other"
    };

    CpuInfo {
        features,
        vector_width: features.vector_width(),
        chunk_size: features.chunk_size(),
        has_simd: features.has_simd(),
        arch,
    }
}

/// Force CPU feature detection (for testing)
#[cfg(test)]
pub fn force_feature_detection() -> CpuFeatures {
    CPU_FEATURES.store(0xFF, Ordering::Relaxed);
    get_cpu_features()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        let features = get_cpu_features();
        // Should detect some valid feature set
        assert!(matches!(
            features,
            CpuFeatures::Scalar | CpuFeatures::Neon | CpuFeatures::Sse41 | CpuFeatures::Avx2
        ));
    }

    #[test]
    fn test_vector_width() {
        assert_eq!(CpuFeatures::Scalar.vector_width(), 1);
        assert_eq!(CpuFeatures::Neon.vector_width(), 4);
        assert_eq!(CpuFeatures::Sse41.vector_width(), 4);
        assert_eq!(CpuFeatures::Avx2.vector_width(), 8);
    }

    #[test]
    fn test_should_use_simd() {
        // Small arrays should use scalar
        assert!(!should_use_simd(1));
        assert!(!should_use_simd(2));

        // Larger arrays might use SIMD depending on capabilities
        let use_simd = should_use_simd(16);
        let features = get_cpu_features();
        if features.has_simd() {
            assert!(use_simd);
        }
    }

    #[test]
    fn test_cpu_info() {
        let info = get_cpu_info();
        assert!(info.vector_width >= 1);
        assert!(info.chunk_size >= 1);
        assert!(!info.arch.is_empty());
    }

    #[test]
    fn test_caching() {
        // Multiple calls should return same result
        let features1 = get_cpu_features();
        let features2 = get_cpu_features();
        assert_eq!(features1, features2);
    }
}
