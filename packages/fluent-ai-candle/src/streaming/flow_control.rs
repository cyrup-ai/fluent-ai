//! Flow control and backpressure management for streaming
//!
//! Provides sophisticated flow control mechanisms to handle streaming backpressure:
//! - Adaptive backpressure detection and response
//! - Multiple backpressure strategies (linear, exponential, adaptive)
//! - Token rate limiting and throttling
//! - Buffer management and congestion control
//! - Real-time flow statistics and monitoring

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use crate::streaming::StreamingError;

/// Backpressure strategies for handling flow control
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureStrategy {
    /// No backpressure handling (fire and forget)
    None,
    /// Drop oldest tokens when buffer is full
    DropOldest,
    /// Drop newest tokens when buffer is full
    DropNewest,
    /// Linear delay increase with buffer pressure
    LinearDelay,
    /// Exponential delay increase with buffer pressure
    Exponential,
    /// Adaptive strategy based on consumer behavior
    Adaptive,
    /// Custom strategy with user-defined parameters
    Custom { 
        base_delay_ms: u64, 
        multiplier: f32, 
        max_delay_ms: u64 
    },
}

impl Default for BackpressureStrategy {
    fn default() -> Self {
        BackpressureStrategy::Exponential
    }
}

/// Flow control statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct FlowStats {
    /// Number of backpressure events detected
    pub backpressure_events: u64,
    /// Total tokens dropped due to backpressure
    pub tokens_dropped: u64,
    /// Total delay time applied (microseconds)
    pub total_delay_us: u64,
    /// Average token processing rate (tokens per second)
    pub avg_token_rate: f64,
    /// Peak token rate observed
    pub peak_token_rate: f64,
    /// Current buffer utilization percentage
    pub buffer_utilization: f32,
    /// Number of flow control adjustments made
    pub flow_adjustments: u64,
    /// Last adjustment timestamp
    pub last_adjustment: Option<Instant>,
}

/// Flow controller for managing streaming backpressure
pub struct FlowController {
    /// Backpressure strategy in use
    strategy: BackpressureStrategy,
    /// Enable flow control
    enabled: bool,
    /// Backpressure threshold (0.0 to 1.0)
    threshold: f32,
    /// Current backpressure state
    is_backpressure_active: bool,
    /// Statistics tracking
    stats: FlowStats,
    /// Token rate tracking
    token_timestamps: Vec<Instant>,
    /// Maximum rate history to keep
    max_rate_history: usize,
    /// Last backpressure check
    last_check: Instant,
    /// Current delay duration
    current_delay: Duration,
    /// Adaptive parameters
    adaptive_params: AdaptiveParams,
    /// Rate limiter
    rate_limiter: TokenRateLimiter,
}

/// Adaptive backpressure parameters
#[derive(Debug, Clone)]
struct AdaptiveParams {
    /// Learning rate for adaptation
    learning_rate: f32,
    /// Sensitivity to buffer changes
    buffer_sensitivity: f32,
    /// Minimum delay (microseconds)
    min_delay_us: u64,
    /// Maximum delay (microseconds)  
    max_delay_us: u64,
    /// Recent buffer utilization history
    buffer_history: Vec<f32>,
    /// History size to maintain
    history_size: usize,
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            buffer_sensitivity: 2.0,
            min_delay_us: 100,
            max_delay_us: 100_000,
            buffer_history: Vec::with_capacity(100),
            history_size: 100,
        }
    }
}

/// Token rate limiter for controlling generation speed
pub struct TokenRateLimiter {
    /// Maximum tokens per second (0 = unlimited)
    max_tokens_per_second: f64,
    /// Token count in current window
    current_window_tokens: AtomicUsize,
    /// Current time window start
    window_start: AtomicU64,
    /// Window duration in nanoseconds
    window_duration_ns: u64,
    /// Enable rate limiting
    enabled: bool,
}

impl TokenRateLimiter {
    /// Create new rate limiter
    #[inline]
    pub fn new(max_tokens_per_second: f64) -> Self {
        Self {
            max_tokens_per_second,
            current_window_tokens: AtomicUsize::new(0),
            window_start: AtomicU64::new(0),
            window_duration_ns: 1_000_000_000, // 1 second
            enabled: max_tokens_per_second > 0.0,
        }
    }

    /// Check if token generation should be allowed
    #[inline]
    pub fn should_allow_token(&self) -> bool {
        if !self.enabled || self.max_tokens_per_second <= 0.0 {
            return true;
        }

        let now = Instant::now().duration_since(Instant::now()).as_nanos() as u64;
        let current_window_start = self.window_start.load(Ordering::Relaxed);

        // Check if we need a new time window
        if now - current_window_start >= self.window_duration_ns {
            // Start new window
            self.window_start.store(now, Ordering::Relaxed);
            self.current_window_tokens.store(0, Ordering::Relaxed);
            return true;
        }

        // Check current window token count
        let current_tokens = self.current_window_tokens.load(Ordering::Relaxed);
        let max_tokens_in_window = self.max_tokens_per_second as usize;

        if current_tokens < max_tokens_in_window {
            self.current_window_tokens.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Update rate limit
    #[inline]
    pub fn update_rate_limit(&mut self, max_tokens_per_second: f64) {
        self.max_tokens_per_second = max_tokens_per_second;
        self.enabled = max_tokens_per_second > 0.0;
        
        // Reset window to apply new rate immediately
        self.window_start.store(0, Ordering::Relaxed);
        self.current_window_tokens.store(0, Ordering::Relaxed);
    }
}

impl FlowController {
    /// Create new flow controller
    #[inline]
    pub fn new(enabled: bool, threshold: f32, strategy: BackpressureStrategy) -> Self {
        Self {
            strategy,
            enabled,
            threshold: threshold.clamp(0.0, 1.0),
            is_backpressure_active: false,
            stats: FlowStats::default(),
            token_timestamps: Vec::with_capacity(1000),
            max_rate_history: 1000,
            last_check: Instant::now(),
            current_delay: Duration::from_micros(0),
            adaptive_params: AdaptiveParams::default(),
            rate_limiter: TokenRateLimiter::new(0.0), // Disabled by default
        }
    }

    /// Check for backpressure conditions
    #[inline]
    pub fn check_backpressure(&mut self, buffer_utilization: f32) -> Result<(), StreamingError> {
        if !self.enabled {
            return Ok(());
        }

        let now = Instant::now();
        self.update_token_rate(now);
        
        let was_active = self.is_backpressure_active;
        self.is_backpressure_active = buffer_utilization >= self.threshold;

        if self.is_backpressure_active {
            self.stats.backpressure_events += 1;
            
            // Apply backpressure strategy
            match self.strategy {
                BackpressureStrategy::None => {},
                BackpressureStrategy::DropOldest | BackpressureStrategy::DropNewest => {
                    // Token dropping is handled by the buffer manager
                    return Err(StreamingError::BackpressureError(
                        format!("Buffer overflow - {} strategy active", 
                            if matches!(self.strategy, BackpressureStrategy::DropOldest) { "drop oldest" } else { "drop newest" }
                        )
                    ));
                },
                BackpressureStrategy::LinearDelay => {
                    self.apply_linear_delay(buffer_utilization);
                },
                BackpressureStrategy::Exponential => {
                    self.apply_exponential_delay(buffer_utilization);
                },
                BackpressureStrategy::Adaptive => {
                    self.apply_adaptive_delay(buffer_utilization);
                },
                BackpressureStrategy::Custom { base_delay_ms, multiplier, max_delay_ms } => {
                    self.apply_custom_delay(buffer_utilization, base_delay_ms, multiplier, max_delay_ms);
                },
            }

            if !was_active {
                self.stats.flow_adjustments += 1;
                self.stats.last_adjustment = Some(now);
            }
        } else {
            // Reduce delay when backpressure subsides
            if was_active {
                self.current_delay = Duration::from_micros(
                    (self.current_delay.as_micros() as u64).saturating_sub(1000).max(0)
                );
            }
        }

        self.stats.buffer_utilization = buffer_utilization;
        self.last_check = now;

        Ok(())
    }

    /// Update token generation rate tracking
    #[inline]
    fn update_token_rate(&mut self, now: Instant) {
        self.token_timestamps.push(now);
        
        // Keep only recent timestamps (last second)
        let cutoff = now - Duration::from_secs(1);
        self.token_timestamps.retain(|&timestamp| timestamp > cutoff);
        
        // Limit history size
        if self.token_timestamps.len() > self.max_rate_history {
            let excess = self.token_timestamps.len() - self.max_rate_history;
            self.token_timestamps.drain(0..excess);
        }

        // Calculate current rate
        let current_rate = self.token_timestamps.len() as f64;
        self.stats.avg_token_rate = (self.stats.avg_token_rate * 0.9) + (current_rate * 0.1);
        
        if current_rate > self.stats.peak_token_rate {
            self.stats.peak_token_rate = current_rate;
        }
    }

    /// Apply linear delay based on buffer pressure
    #[inline]
    fn apply_linear_delay(&mut self, buffer_utilization: f32) {
        let delay_factor = (buffer_utilization - self.threshold) / (1.0 - self.threshold);
        let delay_us = (delay_factor * 10_000.0) as u64; // Up to 10ms
        
        self.current_delay = Duration::from_micros(delay_us);
        self.stats.total_delay_us += delay_us;
    }

    /// Apply exponential delay based on buffer pressure
    #[inline]
    fn apply_exponential_delay(&mut self, buffer_utilization: f32) {
        let pressure = buffer_utilization - self.threshold;
        let delay_factor = pressure * pressure * 10.0; // Exponential growth
        let delay_us = (delay_factor * 10_000.0) as u64; // Base up to 10ms
        
        self.current_delay = Duration::from_micros(delay_us.min(100_000)); // Cap at 100ms
        self.stats.total_delay_us += delay_us;
    }

    /// Apply adaptive delay that learns from consumer behavior
    #[inline]
    fn apply_adaptive_delay(&mut self, buffer_utilization: f32) {
        // Update buffer history
        self.adaptive_params.buffer_history.push(buffer_utilization);
        if self.adaptive_params.buffer_history.len() > self.adaptive_params.history_size {
            self.adaptive_params.buffer_history.remove(0);
        }

        // Calculate trend
        let trend = if self.adaptive_params.buffer_history.len() >= 2 {
            let recent_avg = self.adaptive_params.buffer_history.iter()
                .rev().take(5).sum::<f32>() / 5.0.min(self.adaptive_params.buffer_history.len() as f32);
            let older_avg = self.adaptive_params.buffer_history.iter()
                .rev().skip(5).take(5).sum::<f32>() / 5.0.min((self.adaptive_params.buffer_history.len().saturating_sub(5)) as f32);
            recent_avg - older_avg
        } else {
            0.0
        };

        // Adaptive delay calculation
        let base_pressure = buffer_utilization - self.threshold;
        let trend_factor = trend * self.adaptive_params.buffer_sensitivity;
        let adaptive_factor = base_pressure + trend_factor;
        
        let delay_us = (adaptive_factor * 20_000.0) as u64; // More aggressive than linear
        let clamped_delay = delay_us.clamp(
            self.adaptive_params.min_delay_us,
            self.adaptive_params.max_delay_us
        );
        
        self.current_delay = Duration::from_micros(clamped_delay);
        self.stats.total_delay_us += clamped_delay;
    }

    /// Apply custom delay strategy
    #[inline]
    fn apply_custom_delay(&mut self, buffer_utilization: f32, base_delay_ms: u64, multiplier: f32, max_delay_ms: u64) {
        let pressure = buffer_utilization - self.threshold;
        let delay_ms = (base_delay_ms as f32 * (1.0 + pressure * multiplier)) as u64;
        let clamped_delay_ms = delay_ms.min(max_delay_ms);
        
        self.current_delay = Duration::from_millis(clamped_delay_ms);
        self.stats.total_delay_us += clamped_delay_ms * 1000;
    }

    /// Get current delay to apply
    #[inline]
    pub fn current_delay(&self) -> Duration {
        self.current_delay
    }

    /// Apply the current delay
    #[inline]
    pub async fn apply_delay(&self) -> Result<(), StreamingError> {
        if self.current_delay.as_micros() > 0 {
            tokio::time::sleep(self.current_delay).await;
        }
        Ok(())
    }

    /// Check rate limiter
    #[inline]
    pub fn should_allow_token(&self) -> bool {
        self.rate_limiter.should_allow_token()
    }

    /// Update backpressure strategy
    #[inline]
    pub fn update_strategy(&mut self, strategy: BackpressureStrategy) {
        self.strategy = strategy;
        
        // Reset delay when strategy changes
        self.current_delay = Duration::from_micros(0);
        self.stats.flow_adjustments += 1;
        self.stats.last_adjustment = Some(Instant::now());
    }

    /// Update backpressure threshold
    #[inline]
    pub fn update_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
        self.stats.flow_adjustments += 1;
        self.stats.last_adjustment = Some(Instant::now());
    }

    /// Enable/disable flow control
    #[inline]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.is_backpressure_active = false;
            self.current_delay = Duration::from_micros(0);
        }
    }

    /// Set rate limit (tokens per second, 0 = unlimited)
    #[inline]
    pub fn set_rate_limit(&mut self, max_tokens_per_second: f64) {
        self.rate_limiter.update_rate_limit(max_tokens_per_second);
    }

    /// Get current flow statistics
    #[inline]
    pub fn get_stats(&self) -> &FlowStats {
        &self.stats
    }

    /// Reset flow statistics
    #[inline]
    pub fn reset_stats(&mut self) {
        self.stats = FlowStats::default();
        self.token_timestamps.clear();
    }

    /// Check if currently under backpressure
    #[inline]
    pub fn is_backpressure_active(&self) -> bool {
        self.is_backpressure_active
    }

    /// Get current strategy
    #[inline]
    pub fn strategy(&self) -> BackpressureStrategy {
        self.strategy
    }

    /// Get current threshold
    #[inline]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Configure adaptive parameters
    #[inline]
    pub fn configure_adaptive(&mut self, learning_rate: f32, buffer_sensitivity: f32) {
        self.adaptive_params.learning_rate = learning_rate.clamp(0.0, 1.0);
        self.adaptive_params.buffer_sensitivity = buffer_sensitivity.clamp(0.0, 10.0);
    }
}

/// Utility functions for flow control configuration
pub mod flow_utils {
    use super::*;

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

    /// Validate flow control configuration
    #[inline]
    pub fn validate_flow_config(
        threshold: f32,
        strategy: BackpressureStrategy,
    ) -> Result<(), StreamingError> {
        if threshold < 0.0 || threshold > 1.0 {
            return Err(StreamingError::FlowControlError(
                "Threshold must be between 0.0 and 1.0".to_string()
            ));
        }

        if let BackpressureStrategy::Custom { base_delay_ms, multiplier, max_delay_ms } = strategy {
            if base_delay_ms > max_delay_ms {
                return Err(StreamingError::FlowControlError(
                    "Base delay cannot exceed max delay".to_string()
                ));
            }
            if multiplier < 0.0 {
                return Err(StreamingError::FlowControlError(
                    "Multiplier must be non-negative".to_string()
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration as TokioDuration};

    #[test]
    fn test_flow_controller_creation() {
        let controller = FlowController::new(true, 0.8, BackpressureStrategy::Exponential);
        assert!(controller.enabled);
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
        assert!(controller.check_backpressure(0.7).is_ok());
        assert!(!controller.is_backpressure_active());
        
        // Above threshold - backpressure detected
        assert!(controller.check_backpressure(0.9).is_ok());
        assert!(controller.is_backpressure_active());
        
        // Back below threshold
        assert!(controller.check_backpressure(0.7).is_ok());
        assert!(!controller.is_backpressure_active());
    }

    #[test]
    fn test_drop_strategies() {
        let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::DropOldest);
        
        // Should return error when backpressure triggers drop strategy
        let result = controller.check_backpressure(0.9);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("drop oldest"));
        
        controller.update_strategy(BackpressureStrategy::DropNewest);
        let result = controller.check_backpressure(0.9);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("drop newest"));
    }

    #[test]
    fn test_delay_strategies() {
        let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::LinearDelay);
        
        controller.check_backpressure(0.9).unwrap();
        assert!(controller.current_delay().as_micros() > 0);
        
        controller.update_strategy(BackpressureStrategy::Exponential);
        controller.check_backpressure(0.95).unwrap();
        assert!(controller.current_delay().as_micros() > 0);
    }

    #[test]
    fn test_custom_delay_strategy() {
        let custom_strategy = BackpressureStrategy::Custom { 
            base_delay_ms: 10, 
            multiplier: 2.0, 
            max_delay_ms: 100 
        };
        let mut controller = FlowController::new(true, 0.8, custom_strategy);
        
        controller.check_backpressure(0.9).unwrap();
        assert!(controller.current_delay().as_millis() > 0);
        assert!(controller.current_delay().as_millis() <= 100);
    }

    #[test]
    fn test_stats_tracking() {
        let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::None);
        
        // Trigger backpressure multiple times
        controller.check_backpressure(0.9).unwrap();
        controller.check_backpressure(0.95).unwrap();
        
        let stats = controller.get_stats();
        assert_eq!(stats.backpressure_events, 2);
        assert_eq!(stats.buffer_utilization, 0.95);
        
        controller.reset_stats();
        let stats = controller.get_stats();
        assert_eq!(stats.backpressure_events, 0);
    }

    #[test]
    fn test_rate_limiter() {
        let mut rate_limiter = TokenRateLimiter::new(10.0); // 10 tokens per second
        
        // Should allow initial tokens
        assert!(rate_limiter.should_allow_token());
        
        // Test rate limiting by consuming tokens rapidly
        let mut allowed_count = 0;
        for _ in 0..20 {
            if rate_limiter.should_allow_token() {
                allowed_count += 1;
            }
        }
        
        // Should be limited to approximately the rate limit
        assert!(allowed_count <= 11); // Allow some variance
    }

    #[test]
    fn test_flow_controller_disable() {
        let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::LinearDelay);
        
        controller.check_backpressure(0.9).unwrap();
        assert!(controller.is_backpressure_active());
        
        controller.set_enabled(false);
        assert!(!controller.is_backpressure_active());
        
        // Should not detect backpressure when disabled
        controller.check_backpressure(0.95).unwrap();
        assert!(!controller.is_backpressure_active());
    }

    #[test]
    fn test_adaptive_configuration() {
        let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::Adaptive);
        
        controller.configure_adaptive(0.3, 2.5);
        // Adaptive parameters should be updated but not directly testable without internals access
        
        // Test that adaptive strategy works
        controller.check_backpressure(0.85).unwrap();
        controller.check_backpressure(0.9).unwrap();
        controller.check_backpressure(0.95).unwrap();
        
        assert!(controller.current_delay().as_micros() > 0);
    }

    #[tokio::test]
    async fn test_delay_application() {
        let mut controller = FlowController::new(true, 0.8, BackpressureStrategy::LinearDelay);
        
        controller.check_backpressure(0.9).unwrap();
        
        let start = Instant::now();
        controller.apply_delay().await.unwrap();
        let elapsed = start.elapsed();
        
        // Should have some delay
        assert!(elapsed >= controller.current_delay());
    }

    #[test]
    fn test_utility_controllers() {
        let conservative = flow_utils::conservative_flow_controller();
        assert_eq!(conservative.threshold(), 0.7);
        
        let aggressive = flow_utils::aggressive_flow_controller();
        assert_eq!(aggressive.threshold(), 0.9);
        
        let low_latency = flow_utils::low_latency_flow_controller();
        assert_eq!(low_latency.strategy(), BackpressureStrategy::DropOldest);
        
        let high_throughput = flow_utils::high_throughput_flow_controller();
        assert_eq!(high_throughput.strategy(), BackpressureStrategy::Adaptive);
    }

    #[test]
    fn test_config_validation() {
        assert!(flow_utils::validate_flow_config(0.8, BackpressureStrategy::None).is_ok());
        assert!(flow_utils::validate_flow_config(-0.1, BackpressureStrategy::None).is_err());
        assert!(flow_utils::validate_flow_config(1.1, BackpressureStrategy::None).is_err());
        
        let invalid_custom = BackpressureStrategy::Custom { 
            base_delay_ms: 100, 
            multiplier: -1.0, 
            max_delay_ms: 50 
        };
        assert!(flow_utils::validate_flow_config(0.8, invalid_custom).is_err());
    }
}