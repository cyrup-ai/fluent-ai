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
        max_delay_ms,
    } = strategy
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

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_flow_controller_creation() {
        let controller = FlowController::new(true, 0.8, BackpressureStrategy::Exponential);
        assert!(controller.is_enabled());
        assert_eq!(controller.threshold(), 0.8);
        assert_eq!(controller.strategy(), BackpressureStrategy::Exponential);
        assert!(!controller.is_backpressure_active());
    }

    #[test]
    fn test_threshold_clamping() {
        let controller = FlowController::new(true, 1.5, BackpressureStrategy::None);
        assert_eq!(controller.threshold(), 1.0);

        let controller = FlowController::new(true, -0.5, BackpressureStrategy::None);
        assert_eq!(controller.threshold(), 0.0);
    }

    #[test]
    fn test_backpressure_detection() {
        let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::None);

        // Below threshold - no backpressure
        controller.check_backpressure(0.7);
        assert!(!controller.is_backpressure_active());

        // Above threshold - backpressure detected
        controller.check_backpressure(0.9);
        assert!(controller.is_backpressure_active());

        // Back below threshold
        controller.check_backpressure(0.7);
        assert!(!controller.is_backpressure_active());
    }

    #[test]
    fn test_delay_strategies() {
        let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::LinearDelay);

        controller.check_backpressure(0.9);
        assert!(controller.current_delay().as_micros() > 0);

        controller.update_strategy(BackpressureStrategy::Exponential);
        controller.check_backpressure(0.95);
        assert!(controller.current_delay().as_micros() > 0);
    }

    #[test]
    fn test_custom_delay_strategy() {
        let custom_strategy = BackpressureStrategy::Custom {
            base_delay_ms: 10,
            multiplier: 2.0,
            max_delay_ms: 100,
        };
        let mut controller = FlowController::new(true, 0.8, custom_strategy);

        controller.check_backpressure(0.9);
        assert!(controller.current_delay().as_millis() > 0);
        assert!(controller.current_delay().as_millis() <= 100);
    }

    #[test]
    fn test_stats_tracking() {
        let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::None);

        // Trigger backpressure multiple times
        controller.check_backpressure(0.9);
        controller.check_backpressure(0.95);

        let stats = controller.get_stats();
        assert!(stats.backpressure_events >= 1);
        assert_eq!(stats.buffer_utilization, 0.95);

        controller.reset_stats();
        let stats = controller.get_stats();
        assert_eq!(stats.backpressure_events, 0);
    }

    #[test]
    fn test_rate_limiter() {
        let rate_limiter = TokenRateLimiter::new(10.0); // 10 tokens per second

        // Should allow initial tokens
        assert!(rate_limiter.should_allow_token());

        // Test basic functionality
        assert!(rate_limiter.is_enabled());
        assert_eq!(rate_limiter.rate_limit(), 10.0);
    }

    #[test]
    fn test_flow_controller_disable() {
        let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::LinearDelay);

        controller.check_backpressure(0.9);
        assert!(controller.is_backpressure_active());

        controller.set_enabled(false);
        assert!(!controller.is_backpressure_active());

        // Should not detect backpressure when disabled
        controller.check_backpressure(0.95);
        assert!(!controller.is_backpressure_active());
    }

    #[test]
    fn test_adaptive_configuration() {
        let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::Adaptive);

        controller.configure_adaptive(0.3, 2.5);
        
        // Test that adaptive strategy works
        controller.check_backpressure(0.85);
        controller.check_backpressure(0.9);
        controller.check_backpressure(0.95);

        assert!(controller.current_delay().as_micros() >= 0);
    }

    #[test]
    fn test_utility_controllers() {
        let conservative = conservative_flow_controller();
        assert_eq!(conservative.threshold(), 0.7);

        let aggressive = aggressive_flow_controller();
        assert_eq!(aggressive.threshold(), 0.9);

        let low_latency = low_latency_flow_controller();
        assert_eq!(low_latency.strategy(), BackpressureStrategy::DropOldest);

        let high_throughput = high_throughput_flow_controller();
        assert_eq!(high_throughput.strategy(), BackpressureStrategy::Adaptive);

        let disabled = disabled_flow_controller();
        assert!(!disabled.is_enabled());
    }

    #[test]
    fn test_config_validation() {
        assert!(validate_flow_config(0.8, BackpressureStrategy::None).is_ok());
        assert!(validate_flow_config(-0.1, BackpressureStrategy::None).is_err());
        assert!(validate_flow_config(1.1, BackpressureStrategy::None).is_err());

        let invalid_custom = BackpressureStrategy::Custom {
            base_delay_ms: 100,
            multiplier: -1.0,
            max_delay_ms: 50,
        };
        assert!(validate_flow_config(0.8, invalid_custom).is_err());
    }

    #[test]
    fn test_rate_limit_validation() {
        assert!(validate_rate_limit(10.0).is_ok());
        assert!(validate_rate_limit(0.0).is_ok());
        assert!(validate_rate_limit(-1.0).is_err());
    }

    #[test]
    fn test_specialized_rate_limiters() {
        let streaming = streaming_rate_limiter(100.0);
        assert_eq!(streaming.window_duration_ms(), 1000);

        let burst_tolerant = burst_tolerant_rate_limiter(100.0);
        assert_eq!(burst_tolerant.window_duration_ms(), 5000);
    }

    #[test]
    fn test_rate_limited_controller() {
        let controller = rate_limited_flow_controller(
            0.8,
            BackpressureStrategy::Exponential,
            50.0,
        );
        
        assert_eq!(controller.threshold(), 0.8);
        assert_eq!(controller.strategy(), BackpressureStrategy::Exponential);
        assert_eq!(controller.rate_limiter().rate_limit(), 50.0);
    }
}