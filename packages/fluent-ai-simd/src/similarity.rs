//! Ultra-High-Performance Vector Similarity Operations
//!
//! This module provides production-ready SIMD-accelerated vector similarity functions
//! with runtime feature detection and optimal implementation selection.

use std::sync::atomic::{AtomicU64, Ordering};
use wide::{f32x4, f32x8, CmpGe};
use cfg_if::cfg_if;

/// Atomic metrics for similarity operations
#[derive(Debug, Default)]
struct SimilarityMetrics {
    /// Total similarity calculations performed
    total_calculations: AtomicU64,
    /// Total vector elements processed
    total_elements_processed: AtomicU64,
    /// Total time spent in SIMD operations (nanoseconds)
    simd_time_ns: AtomicU64,
}

impl SimilarityMetrics {
    #[inline]
    fn record_calculation(&self, elements: usize, duration_ns: u64) {
        self.total_calculations.fetch_add(1, Ordering::Relaxed);
        self.total_elements_processed
            .fetch_add(elements as u64, Ordering::Relaxed);
        self.simd_time_ns.fetch_add(duration_ns, Ordering::Relaxed);
    }

    fn get_metrics(&self) -> SimilarityMetricsSnapshot {
        SimilarityMetricsSnapshot {
            total_calculations: self.total_calculations.load(Ordering::Relaxed),
            total_elements_processed: self.total_elements_processed.load(Ordering::Relaxed),
            simd_time_ns: self.simd_time_ns.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of similarity metrics
#[derive(Debug, Clone, Copy)]
pub struct SimilarityMetricsSnapshot {
    /// Total similarity calculations performed
    pub total_calculations: u64,
    /// Total vector elements processed
    pub total_elements_processed: u64,
    /// Total time spent in SIMD operations (nanoseconds)
    pub simd_time_ns: u64,
}

lazy_static::lazy_static! {
    static ref METRICS: SimilarityMetrics = SimilarityMetrics::default();
}

/// Get current similarity metrics snapshot
#[inline]
pub fn get_similarity_metrics() -> SimilarityMetricsSnapshot {
    METRICS.get_metrics()
}

/// Reset all similarity metrics to zero
#[inline]
pub fn reset_similarity_metrics() {
    let metrics = &*METRICS;
    metrics.total_calculations.store(0, Ordering::Relaxed);
    metrics.total_elements_processed.store(0, Ordering::Relaxed);
    metrics.simd_time_ns.store(0, Ordering::Relaxed);
}

/// Smart cosine similarity with automatic implementation selection
///
/// This function automatically selects the optimal implementation based on:
/// - Vector length
/// - CPU features available at runtime
/// - Cache line alignment
///
/// # Panics
/// - If input vectors have different lengths
#[inline]
pub fn smart_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");
    
    // For very small vectors, scalar is faster due to SIMD overhead
    if a.len() < 4 {
        return scalar_cosine_similarity(a, b);
    }
    
    // Start timing for metrics
    let start = std::time::Instant::now();
    
    let result = if a.len() >= 128 {
        // For large vectors, use the most aggressive SIMD available
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe { avx512_cosine_similarity(a, b) }
            } else if is_x86_feature_detected!("avx2") {
                unsafe { avx2_cosine_similarity(a, b) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { sse41_cosine_similarity(a, b) }
            } else {
                fallback_simd_cosine_similarity(a, b)
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                unsafe { neon_cosine_similarity(a, b) }
            } else {
                fallback_simd_cosine_similarity(a, b)
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            fallback_simd_cosine_similarity(a, b)
        }
    } else {
        // For medium-sized vectors, use a balanced approach
        fallback_simd_cosine_similarity(a, b)
    };
    
    // Record metrics
    let elapsed = start.elapsed();
    METRICS.record_calculation(a.len(), elapsed.as_nanos() as u64);
    
    result
}

/// Fallback SIMD implementation using portable wide vectors
///
/// This implementation uses the `wide` crate for portable SIMD operations
/// that work on any platform with 128-bit vectors.
#[inline(always)]
fn fallback_simd_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let simd_width = 8; // Using f32x8 for wider vectors
    let simd_chunks = len / simd_width;
    let remainder = len % simd_width;

    // Initialize accumulators with SIMD vectors for better pipelining
    let mut dot_product_simd = f32x8::ZERO;
    let mut norm_a_simd = f32x8::ZERO;
    let mut norm_b_simd = f32x8::ZERO;

    // Process 8 elements at a time with f32x8
    for i in 0..simd_chunks {
        let base_idx = i * simd_width;
        
        // Load 8 elements at once (will be optimized to SIMD load)
        let a_chunk = f32x8::new([
            a[base_idx], a[base_idx + 1], a[base_idx + 2], a[base_idx + 3],
            a[base_idx + 4], a[base_idx + 5], a[base_idx + 6], a[base_idx + 7],
        ]);
        
        let b_chunk = f32x8::new([
            b[base_idx], b[base_idx + 1], b[base_idx + 2], b[base_idx + 3],
            b[base_idx + 4], b[base_idx + 5], b[base_idx + 6], b[base_idx + 7],
        ]);
        
        // Fused multiply-add operations for better precision and performance
        dot_product_simd = a_chunk.mul_add(b_chunk, dot_product_simd);
        norm_a_simd = a_chunk.mul_add(a_chunk, norm_a_simd);
        norm_b_simd = b_chunk.mul_add(b_chunk, norm_b_simd);
    }

    // Horizontal sum of SIMD vectors
    let dot_product = dot_product_simd.reduce_sum();
    let norm_a = norm_a_simd.reduce_sum();
    let norm_b = norm_b_simd.reduce_sum();

    // Process remaining elements with scalar operations
    let remaining_start = simd_chunks * simd_width;
    let (dot_product_remainder, norm_a_remainder, norm_b_remainder) = 
        scalar_cosine_components(&a[remaining_start..], &b[remaining_start..]);

    // Combine results
    let final_dot = dot_product + dot_product_remainder;
    let final_norm_a = norm_a + norm_a_remainder;
    let final_norm_b = norm_b + norm_b_remainder;

    // Final similarity calculation with guard against division by zero
    let norm_product = (final_norm_a * final_norm_b).sqrt();
    if norm_product <= f32::EPSILON {
        0.0
    } else {
        (final_dot / norm_product).clamp(-1.0, 1.0)
    }
}

/// Scalar cosine similarity for small vectors or fallback
#[inline(always)]
fn scalar_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (dot, norm_a, norm_b) = scalar_cosine_components(a, b);
    
    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product <= f32::EPSILON {
        0.0
    } else {
        (dot / norm_product).clamp(-1.0, 1.0)
    }
}

/// Compute cosine similarity components using scalar operations
#[inline(always)]
fn scalar_cosine_components(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    
    for (&a_val, &b_val) in a.iter().zip(b) {
        dot += a_val * b_val;
        norm_a += a_val * a_val;
        norm_b += b_val * b_val;
    }
    
    (dot, norm_a, norm_b)
}
        let b_chunk = f32x4::new([
            b[base_idx],
            b[base_idx + 1],
            b[base_idx + 2],
            b[base_idx + 3],
        ]);

        // SIMD operations: dot product and norms
        dot_product_simd += a_chunk * b_chunk;
        norm_a_simd += a_chunk * a_chunk;
        norm_b_simd += b_chunk * b_chunk;
    }

    // Sum the SIMD results (manually extract and sum components)
    let dot_array: [f32; 4] = dot_product_simd.into();
    let norm_a_array: [f32; 4] = norm_a_simd.into();
    let norm_b_array: [f32; 4] = norm_b_simd.into();

    let mut dot_product = dot_array[0] + dot_array[1] + dot_array[2] + dot_array[3];
    let mut norm_a_squared = norm_a_array[0] + norm_a_array[1] + norm_a_array[2] + norm_a_array[3];
    let mut norm_b_squared = norm_b_array[0] + norm_b_array[1] + norm_b_array[2] + norm_b_array[3];

    // Process remaining elements (scalar operations for remainder)
    for i in (simd_chunks * 4)..len {
        let a_val = a[i];
        let b_val = b[i];
        dot_product += a_val * b_val;
        norm_a_squared += a_val * a_val;
        norm_b_squared += b_val * b_val;
    }

    // Calculate final cosine similarity
    let norm_a = norm_a_squared.sqrt();
    let norm_b = norm_b_squared.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Scalar cosine similarity for small vectors (< 16 elements) and fallback cases  
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}