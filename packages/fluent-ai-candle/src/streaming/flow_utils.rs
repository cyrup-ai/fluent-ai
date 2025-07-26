//! Flow control utility functions and factory methods
//!
//! Provides convenient factory methods for common flow control configurations:
//! - Pre-configured controllers for common scenarios
//! - Configuration validation
//! - Comprehensive test suite

use super::flow_controller::FlowController;
use super::flow_types::BackpressureStrategy;
use super::rate_limiter::TokenRateLimiter;
use crate::streaming::StreamingError;

/// Create flow controller with conservative settings
#[inline]
pub fn conservative_flow_controller() -> FlowController {
    FlowController::new(true, 0.7, BackpressureStrategy::LinearDelay)
}

/// Create flow controller with aggressive settings
#[inline]
pub fn aggressive_flow_controller() -> FlowController {
    FlowController::new(true, 0.9, BackpressureStrategy::Exponential)
}

/// Create flow controller optimized for low-latency scenarios
#[inline]
pub fn low_latency_flow_controller() -> FlowController {
    FlowController::new(true, 0.95, BackpressureStrategy::DropOldest)
}

/// Create flow controller for high-throughput scenarios
#[inline]
pub fn high_throughput_flow_controller() -> FlowController {
    let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::Adaptive);
    controller.configure_adaptive(0.2, 1.5);
    controller
}

/// Create flow controller with custom rate limiting
#[inline]
pub fn rate_limited_flow_controller(
    threshold: f32,
    strategy: BackpressureStrategy,
    max_tokens_per_second: f64,
) -> FlowController {
    FlowController::with_rate_limit(true, threshold, strategy, max_tokens_per_second)
}

/// Create disabled flow controller (no flow control)
#[inline]
pub fn disabled_flow_controller() -> FlowController {
    FlowController::new(false, 1.0, BackpressureStrategy::None)
}

/// Validate flow control configuration
#[inline]
pub fn validate_flow_config(
    threshold: f32,
    strategy: BackpressureStrategy,
) -> Result<(), StreamingError> {
    if threshold < 0.0 || threshold > 1.0 {
        return Err(StreamingError::FlowControlError(
            "Threshold must be between 0.0 and 1.0".to_string(),
        ));
    }

    if let BackpressureStrategy::Custom {
        base_delay_ms,
        multiplier,
        max_delay_ms} = strategy
    {
        if base_delay_ms > max_delay_ms {
            return Err(StreamingError::FlowControlError(
                "Base delay cannot exceed max delay".to_string(),
            ));
        }
        if multiplier < 0.0 {
            return Err(StreamingError::FlowControlError(
                "Multiplier must be non-negative".to_string(),
            ));
        }
    }

    Ok(())
}

/// Validate rate limiting configuration
#[inline]
pub fn validate_rate_limit(max_tokens_per_second: f64) -> Result<(), StreamingError> {
    if max_tokens_per_second < 0.0 {
        return Err(StreamingError::FlowControlError(
            "Rate limit must be non-negative".to_string(),
        ));
    }
    Ok(())
}

/// Create optimized rate limiter for streaming scenarios
#[inline]
pub fn streaming_rate_limiter(max_tokens_per_second: f64) -> TokenRateLimiter {
    TokenRateLimiter::with_window_duration(max_tokens_per_second, 1000) // 1 second window
}

/// Create burst-tolerant rate limiter with larger window
#[inline]
pub fn burst_tolerant_rate_limiter(max_tokens_per_second: f64) -> TokenRateLimiter {
    TokenRateLimiter::with_window_duration(max_tokens_per_second, 5000) // 5 second window
}
