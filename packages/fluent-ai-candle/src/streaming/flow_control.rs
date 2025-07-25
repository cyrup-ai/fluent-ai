//! Flow control and backpressure management for streaming (compatibility layer)
//!
//! This module re-exports all functionality from the decomposed flow control modules
//! to maintain API compatibility while providing a cleaner internal structure.
//!
//! The functionality is now split across:
//! - `flow_types`: Core types and enums
//! - `rate_limiter`: Token rate limiting
//! - `flow_controller`: Main flow controller implementation  
//! - `flow_utils`: Utility functions and tests

// Re-export all types from decomposed modules
pub use super::flow_controller::FlowController;
pub use super::flow_types::{AdaptiveParams, BackpressureStrategy, FlowStats};
pub use super::flow_utils::{
    aggressive_flow_controller, burst_tolerant_rate_limiter, conservative_flow_controller,
    disabled_flow_controller, high_throughput_flow_controller, low_latency_flow_controller,
    rate_limited_flow_controller, streaming_rate_limiter, validate_flow_config,
    validate_rate_limit,
};
pub use super::rate_limiter::TokenRateLimiter;

// Maintain backward compatibility with original module interface
pub mod flow_utils {
    pub use crate::streaming::flow_utils::*;
}