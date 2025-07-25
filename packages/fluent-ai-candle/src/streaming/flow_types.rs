//! Flow control types and data structures
//!
//! Defines core types for streaming backpressure management:
//! - BackpressureStrategy enum with multiple strategies
//! - FlowStats for monitoring flow control performance
//! - AdaptiveParams for adaptive backpressure tuning

use std::time::Instant;

/// Backpressure strategies for handling flow control
#[derive(Debug, Clone, Copy, PartialEq)]
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
        max_delay_ms: u64}}

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
    pub last_adjustment: Option<Instant>}

impl FlowStats {
    /// Create new empty flow statistics
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all statistics to default values
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Record a backpressure event
    #[inline]
    pub fn record_backpressure_event(&mut self) {
        self.backpressure_events += 1;
    }

    /// Record tokens dropped due to backpressure
    #[inline]
    pub fn record_tokens_dropped(&mut self, count: u64) {
        self.tokens_dropped += count;
    }

    /// Record delay time applied
    #[inline]
    pub fn record_delay(&mut self, delay_us: u64) {
        self.total_delay_us += delay_us;
    }

    /// Update token processing rate
    #[inline]
    pub fn update_token_rate(&mut self, rate: f64) {
        self.avg_token_rate = rate;
        if rate > self.peak_token_rate {
            self.peak_token_rate = rate;
        }
    }

    /// Update buffer utilization percentage
    #[inline]
    pub fn update_buffer_utilization(&mut self, utilization: f32) {
        self.buffer_utilization = utilization.clamp(0.0, 1.0);
    }

    /// Record a flow control adjustment
    #[inline]
    pub fn record_adjustment(&mut self) {
        self.flow_adjustments += 1;
        self.last_adjustment = Some(Instant::now());
    }
}

/// Adaptive backpressure parameters
#[derive(Debug, Clone)]
pub struct AdaptiveParams {
    /// Learning rate for adaptation
    pub learning_rate: f32,
    /// Sensitivity to buffer changes
    pub buffer_sensitivity: f32,
    /// Minimum delay (microseconds)
    pub min_delay_us: u64,
    /// Maximum delay (microseconds)
    pub max_delay_us: u64,
    /// Exponential smoothing factor
    pub smoothing_factor: f32}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            buffer_sensitivity: 2.0,
            min_delay_us: 10,
            max_delay_us: 10_000,
            smoothing_factor: 0.9}
    }
}

impl AdaptiveParams {
    /// Create new adaptive parameters with validation
    #[inline]
    pub fn new(
        learning_rate: f32,
        buffer_sensitivity: f32,
        min_delay_us: u64,
        max_delay_us: u64,
    ) -> Self {
        Self {
            learning_rate: learning_rate.clamp(0.0, 1.0),
            buffer_sensitivity: buffer_sensitivity.clamp(0.0, 10.0),
            min_delay_us,
            max_delay_us: max_delay_us.max(min_delay_us),
            smoothing_factor: 0.9}
    }

    /// Update learning parameters with validation
    #[inline]
    pub fn update_learning(&mut self, learning_rate: f32, buffer_sensitivity: f32) {
        self.learning_rate = learning_rate.clamp(0.0, 1.0);
        self.buffer_sensitivity = buffer_sensitivity.clamp(0.0, 10.0);
    }

    /// Update delay bounds with validation
    #[inline]
    pub fn update_delay_bounds(&mut self, min_delay_us: u64, max_delay_us: u64) {
        self.min_delay_us = min_delay_us;
        self.max_delay_us = max_delay_us.max(min_delay_us);
    }
}