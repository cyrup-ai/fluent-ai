//! SIMD-optimized operations for machine learning workloads
//!
//! High-performance implementations of common ML operations with runtime CPU
//! feature detection and optimal SIMD utilization.

pub mod softmax;
pub mod temperature;

// Re-export main operation functions for convenient access
pub use softmax::{
    compute_log_softmax, compute_softmax, compute_softmax_inplace, SoftmaxProcessor, SoftmaxStats,
};
pub use temperature::{
    apply_temperature_scaling, apply_temperature_scaling_inplace, TemperatureProcessor,
    TemperatureStats,
};
