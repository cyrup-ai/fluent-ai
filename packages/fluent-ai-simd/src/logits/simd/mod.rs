//! SIMD-accelerated logits processing implementations

// Re-export all SIMD functions for logits processing
pub use super::processing::*;
pub use super::topk::*;
pub use super::penalties::*;

// Legacy compatibility - scalar functions are now SIMD functions
pub use super::processing::normalize_probabilities_simd as normalize_probabilities_scalar;
pub use super::topk::topk_filtering_simd as topk_filtering_scalar;
pub use super::penalties::apply_penalties_simd as apply_penalties_scalar;

// Nucleus sampling from processing module
pub use super::nucleus::prepare_nucleus_sampling_simd as prepare_nucleus_sampling_scalar;