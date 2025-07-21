//! SIMD-accelerated similarity operations
//!
//! This module contains platform-specific SIMD implementations of similarity operations.
//! The appropriate implementation is selected at runtime based on CPU feature detection.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub mod portable;

use super::traits::RuntimeSelectable;
use lazy_static::lazy_static;
use std::sync::Arc;

/// Get the best available SIMD implementation for the current CPU
pub fn best_available() -> Arc<dyn RuntimeSelectable> {
    lazy_static! {
        static ref BEST_IMPL: Arc<dyn RuntimeSelectable> = {
            // Check for platform-specific SIMD features first
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    return Arc::new(x86::avx512::Avx512Similarity::new());
                }
                if is_x86_feature_detected!("avx2") {
                    return Arc::new(x86::avx2::Avx2Similarity::new());
                }
                if is_x86_feature_detected!("sse4.1") {
                    return Arc::new(x86::sse41::Sse41Similarity::new());
                }
            }
            
            #[cfg(target_arch = "aarch64")]
            {
                if is_aarch64_feature_detected!("neon") {
                    return Arc::new(aarch64::neon::NeonSimilarity::new());
                }
            }
            
            // Fall back to portable SIMD
            Arc::new(portable::PortableSimdSimilarity::new())
        };
    }
    
    BEST_IMPL.clone()
}

/// Get the best implementation for a given vector length
pub fn best_for_length(len: usize) -> Arc<dyn RuntimeSelectable> {
    let best = best_available();
    
    // If the best implementation is not optimal for this length,
    // fall back to a more appropriate one
    if len < best.optimal_vector_length() {
        return Arc::new(crate::similarity::scalar::ScalarSimilarity::new());
    }
    
    best
}
