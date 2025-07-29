//! ARM AArch64 SIMD optimizations

pub mod neon;

pub use neon::{is_neon_available, NeonSimilarity};
